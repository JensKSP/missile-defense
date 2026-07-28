# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: OpenAI Codex
"""Run and compare several genuinely fresh PPO training seeds.

This module deliberately stays one process outside :mod:`missile_defense.train`.  Each seed
gets a separate output directory and a separate trainer process, which keeps
checkpoints, optimizer state, STOP/PAUSE controls, and console logs isolated.
The winner is chosen only from the trainer's validation rows; the held-out
canonical benchmark remains untouched until the experiment has selected one
checkpoint.

Reached as ``missile-defense-train --multiseed`` rather than by a command of its
own: it is the trainer run several times over, and a second name on ``PATH`` for
that was a name to explain rather than a distinction to make.  Typical use from a
checkout::

    missile-defense-train --multiseed \
        --out-dir runs/entity-3seed --num-seeds 3 --seed-start 1000 \
        -- --architecture entity --updates 750 --envs 4096

Everything after ``--`` is passed to ``missile_defense.train``.  ``--seed``, ``--out-dir``,
``--resume`` and ``--load`` are intentionally forbidden there: the runner owns
those arguments so every run is fresh and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from .benchmark import VALIDATION_SPLIT

SCHEMA_VERSION = 1
SUMMARY_COLUMNS = (
    "train_seed",
    "status",
    "validation_score",
    "update",
    "run_dir",
    "checkpoint",
    "error",
)
OWNED_TRAINER_OPTIONS = ("--seed", "--out-dir", "--resume", "--load")


class ExperimentError(ValueError):
    """The experiment cannot be launched or compared without ambiguity."""


@dataclass(frozen=True)
class ValidationProtocol:
    """The evaluation fields that must match before scores are comparable."""

    seed_split: str
    seed_offset: int
    seed_count: int
    frame_skip: int
    max_ticks: int
    inference_device: str


@dataclass(frozen=True)
class SeedResult:
    """The best validation result available for one independent training seed."""

    train_seed: int
    status: str
    run_dir: Path
    checkpoint: Path
    validation_score: float | None = None
    update: int | None = None
    protocol: ValidationProtocol | None = None
    error: str | None = None


def seed_values(seed_start: int, num_seeds: int) -> list[int]:
    """Return the deterministic contiguous training-seed schedule."""

    if num_seeds < 1:
        raise ExperimentError("--num-seeds must be at least 1")
    if seed_start < 0:
        raise ExperimentError("--seed-start must not be negative")
    return list(range(seed_start, seed_start + num_seeds))


def trainer_arguments(arguments: Sequence[str]) -> list[str]:
    """Validate and normalize arguments forwarded to each trainer process."""

    forwarded = list(arguments)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    for argument in forwarded:
        option = argument.split("=", 1)[0]
        if option in OWNED_TRAINER_OPTIONS:
            raise ExperimentError(
                f"{option} is owned by the multi-seed runner; "
                "set experiment seeds and output directories on the runner itself"
            )
    return forwarded


def run_directory(experiment_dir: Path, seed: int) -> Path:
    """Stable, lexically sortable directory for one training seed."""

    return experiment_dir / f"seed-{seed:06d}"


def training_command(
    python: str,
    experiment_dir: Path,
    seed: int,
    arguments: Sequence[str],
) -> list[str]:
    """Construct one from-zero trainer invocation."""

    return [
        python,
        "-u",
        "-m",
        "missile_defense.train",
        "--seed",
        str(seed),
        "--out-dir",
        str(run_directory(experiment_dir, seed)),
        *trainer_arguments(arguments),
    ]


def _integer(row: Mapping[str, str], field: str, path: Path) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ExperimentError(f"{path} has no valid {field!r} field") from error


def _protocol(row: Mapping[str, str], path: Path) -> ValidationProtocol:
    split = row.get("seed_split", "")
    if split != VALIDATION_SPLIT:
        raise ExperimentError(
            f"{path} row is {split or 'unlabelled'} rather than {VALIDATION_SPLIT}; "
            "canonical results must not select an experiment winner"
        )
    inference_device = row.get("inference_device", "")
    if not inference_device:
        raise ExperimentError(f"{path} has no inference_device protocol field")
    return ValidationProtocol(
        seed_split=split,
        seed_offset=_integer(row, "seed_offset", path),
        seed_count=_integer(row, "seed_count", path),
        frame_skip=_integer(row, "frame_skip", path),
        max_ticks=_integer(row, "max_ticks", path),
        inference_device=inference_device,
    )


def best_validation_result(train_seed: int, run_dir: Path) -> SeedResult:
    """Read one run's validation-selected checkpoint and its best eval row."""

    checkpoint = run_dir / "checkpoints" / "policy-best.pt"
    eval_path = run_dir / "evals.csv"
    try:
        if not checkpoint.is_file():
            raise ExperimentError(f"missing validation-selected checkpoint {checkpoint}")
        if not eval_path.is_file():
            raise ExperimentError(f"missing evaluation log {eval_path}")
        with eval_path.open(newline="", encoding="utf-8") as handle:
            rows = [
                row for row in csv.DictReader(handle) if row.get("seed_split") == VALIDATION_SPLIT
            ]
        if not rows:
            raise ExperimentError(
                f"{eval_path} has no {VALIDATION_SPLIT} rows; "
                "do not substitute a canonical benchmark row"
            )

        def score(row: Mapping[str, str]) -> float:
            try:
                return float(row["mean_score"])
            except (KeyError, TypeError, ValueError) as error:
                raise ExperimentError(f"{eval_path} has an invalid mean_score") from error

        best = max(rows, key=score)
        return SeedResult(
            train_seed=train_seed,
            status="complete",
            run_dir=run_dir,
            checkpoint=checkpoint,
            validation_score=score(best),
            update=_integer(best, "update", eval_path),
            protocol=_protocol(best, eval_path),
        )
    except ExperimentError as error:
        return SeedResult(
            train_seed=train_seed,
            status="incomplete",
            run_dir=run_dir,
            checkpoint=checkpoint,
            error=str(error),
        )


def _validate_protocols(results: Sequence[SeedResult]) -> ValidationProtocol:
    complete = [result for result in results if result.protocol is not None]
    if not complete:
        raise ExperimentError("no run has a comparable validation result")
    expected = complete[0].protocol
    assert expected is not None
    mismatches = [result.train_seed for result in complete[1:] if result.protocol != expected]
    if mismatches:
        seeds = ", ".join(str(seed) for seed in mismatches)
        raise ExperimentError(
            "validation protocols differ across runs; refusing to compare scores "
            f"(mismatching training seeds: {seeds})"
        )
    return expected


def select_winner(results: Sequence[SeedResult]) -> SeedResult:
    """Select the highest score after proving all validation protocols match."""

    _validate_protocols(results)
    candidates = [result for result in results if result.validation_score is not None]

    def ranking(result: SeedResult) -> tuple[float, int]:
        score = result.validation_score
        assert score is not None
        return score, -result.train_seed

    # Ties deliberately prefer the lower training seed, making reruns stable.
    return max(candidates, key=ranking)


def collect_results(experiment_dir: Path, seeds: Sequence[int]) -> list[SeedResult]:
    """Aggregate the current artifacts without running or importing PyTorch."""

    return [best_validation_result(seed, run_directory(experiment_dir, seed)) for seed in seeds]


def _result_payload(result: SeedResult, winner: SeedResult | None) -> dict[str, Any]:
    return {
        "train_seed": result.train_seed,
        "status": "selected" if winner == result else result.status,
        "validation_score": result.validation_score,
        "update": result.update,
        "run_dir": str(result.run_dir),
        "checkpoint": str(result.checkpoint),
        "protocol": asdict(result.protocol) if result.protocol is not None else None,
        "error": result.error,
    }


def write_summary(experiment_dir: Path, results: Sequence[SeedResult]) -> SeedResult | None:
    """Write machine-readable JSON/CSV summaries and return the winner, if any."""

    try:
        winner = select_winner(results)
        selection_error = None
    except ExperimentError as error:
        winner = None
        selection_error = str(error)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "selection": "highest mean_score on matching validation protocol",
        "canonical_benchmark_used": False,
        "winner": _result_payload(winner, winner) if winner is not None else None,
        "selection_error": selection_error,
        "runs": [_result_payload(result, winner) for result in results],
    }
    _write_json(experiment_dir / "summary.json", payload)
    csv_path = experiment_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for result in results:
            row = _result_payload(result, winner)
            writer.writerow({column: row[column] for column in SUMMARY_COLUMNS})
    return winner


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prepare(experiment_dir: Path) -> None:
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise ExperimentError(
            f"{experiment_dir} is not empty; use a new directory so every seed starts from zero"
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)


def _manifest(
    experiment_dir: Path,
    python: str,
    seeds: Sequence[int],
    arguments: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "python": python,
        "seeds": list(seeds),
        "trainer_arguments": list(arguments),
        "commands": [training_command(python, experiment_dir, seed, arguments) for seed in seeds],
        "exit_codes": {},
    }


def _print_summary(results: Sequence[SeedResult], winner: SeedResult | None) -> None:
    print("\nvalidation selection (canonical benchmark not used)")
    for result in results:
        if result.validation_score is None:
            print(f"  seed {result.train_seed:>6}: incomplete — {result.error}")
            continue
        marker = "  <- selected" if result == winner else ""
        print(
            f"  seed {result.train_seed:>6}: {result.validation_score:>10,.1f} "
            f"at update {result.update}{marker}"
        )
    if winner is not None:
        print(f"selected checkpoint -> {winner.checkpoint}")
        print(
            "Run the held-out canonical benchmark once, after all experiment "
            "decisions are final:\n  "
            f"{sys.executable} -m missile_defense.train --load {winner.checkpoint}"
        )


def run_experiment(
    experiment_dir: Path,
    seeds: Sequence[int],
    arguments: Sequence[str],
    *,
    python: str = sys.executable,
) -> int:
    """Launch every seed serially, then aggregate all validation results."""

    normalized = trainer_arguments(arguments)
    _prepare(experiment_dir)
    manifest = _manifest(experiment_dir, python, seeds, normalized)
    manifest_path = experiment_dir / "experiment.json"
    _write_json(manifest_path, manifest)
    exit_codes = cast(dict[str, int], manifest["exit_codes"])

    interrupted = False
    for seed in seeds:
        command = training_command(python, experiment_dir, seed, normalized)
        print(f"\n=== training seed {seed} -> {run_directory(experiment_dir, seed)} ===")
        print("+", " ".join(command))
        try:
            completed = subprocess.run(command, check=False)
            exit_code = completed.returncode
        except KeyboardInterrupt:
            interrupted = True
            exit_code = 130
        exit_codes[str(seed)] = exit_code
        _write_json(manifest_path, manifest)
        results = collect_results(experiment_dir, seeds)
        write_summary(experiment_dir, results)
        if interrupted:
            break

    results = collect_results(experiment_dir, seeds)
    winner = write_summary(experiment_dir, results)
    _print_summary(results, winner)
    failed = any(code != 0 for code in exit_codes.values())
    incomplete = len(exit_codes) != len(seeds)
    return 130 if interrupted else int(failed or incomplete or winner is None)


def _parser() -> argparse.ArgumentParser:
    # `prog` is spelled out because argparse would otherwise derive it from
    # sys.argv[0] and print `usage: missile-defense-train ...` — the command
    # without the flag that got here, which is a usage line that does not do what
    # it says. The flag is consumed in missile_defense.cli before this parser ever sees it.
    parser = argparse.ArgumentParser(
        prog="missile-defense-train --multiseed",
        description=(
            "Train independent policies from zero and select one using only "
            "the validation seed split."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="New experiment directory; it must be absent or empty.",
    )
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for each missile_defense.train child.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild summaries for an existing experiment without starting training.",
    )
    parser.add_argument(
        "trainer_arguments",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed to every missile_defense.train process.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        seeds = seed_values(args.seed_start, args.num_seeds)
        if args.aggregate_only:
            if not args.out_dir.is_dir():
                raise ExperimentError(f"experiment directory does not exist: {args.out_dir}")
            results = collect_results(args.out_dir, seeds)
            winner = write_summary(args.out_dir, results)
            _print_summary(results, winner)
            return int(winner is None)
        return run_experiment(
            args.out_dir,
            seeds,
            args.trainer_arguments,
            python=args.python,
        )
    except ExperimentError as error:
        print(f"multi-seed experiment: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
