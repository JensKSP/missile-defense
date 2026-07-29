# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""One trainer window per library (`missile_defense.ui.instance`).

What is tested is the *decision and the round trip*, not Qt: the identity two
launches derive, whether a claim is exclusive, and that an "activate" handed to
a twin actually reaches its callback. The module is standard library on
purpose, so all of this runs on the CI gate that installs no PySide6 — which is
also why the raise itself (a Qt signal in `app.main`) is not here.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
from multiprocessing import connection
from pathlib import Path

import pytest
from missile_defense.ui import instance


def private_address(tag: str) -> str:
    """An endpoint no real trainer — and no parallel test shard — can be on."""
    return instance.endpoint(f"test\n{os.getpid()}\n{tag}")


# ---- the identity ------------------------------------------------------------


def test_the_same_library_is_the_same_identity_however_it_is_spelled(tmp_path: Path) -> None:
    plain = instance.trainer_key(tmp_path / "runs")
    assert instance.trainer_key(tmp_path / "elsewhere" / ".." / "runs") == plain
    assert instance.trainer_key(tmp_path / "other") != plain


def test_endpoints_separate_users_and_identities_deterministically() -> None:
    mine = instance.endpoint("trainer\n/data/runs", user="jens", runtime_dir="/run/user/1000")
    assert (
        instance.endpoint("trainer\n/data/runs", user="jens", runtime_dir="/run/user/1000") == mine
    )
    assert (
        instance.endpoint("trainer\n/data/runs", user="else", runtime_dir="/run/user/1001") != mine
    )
    assert instance.endpoint("trainer\n/other", user="jens", runtime_dir="/run/user/1000") != mine


def test_endpoints_live_where_the_platform_puts_cheap_named_channels() -> None:
    piped = instance.endpoint("trainer\n/data/runs", user="jens", platform="win32")
    assert piped.startswith(r"\\.\pipe\missile-defense-trainer-")
    socketed = instance.endpoint(
        "trainer\n/data/runs", user="jens", runtime_dir="/run/user/1000", platform="linux"
    )
    assert socketed.startswith("/run/user/1000/missile-defense-trainer-")
    assert socketed.endswith(".sock")


def test_engaged_until_someone_says_no() -> None:
    assert instance.engaged({})
    assert instance.engaged({instance.DISABLE_VARIABLE: "1"})
    assert not instance.engaged({instance.DISABLE_VARIABLE: "0"})


# ---- the round trip ----------------------------------------------------------


def test_forward_with_nobody_serving_changes_nothing() -> None:
    assert instance.forward(private_address("nobody")) is False


def test_an_activation_reaches_the_twin_and_the_claim_is_exclusive() -> None:
    address = private_address("round-trip")
    raised = threading.Event()
    guard = instance.SingleInstance(address, raised.set)
    assert guard.claim()
    try:
        # The duplicate's whole life: find the twin, hand over, exit.
        assert instance.forward(address) is True
        assert raised.wait(5), "the activation never reached the callback"

        # A second server is refused — which is what sends a real duplicate
        # down the forward path instead of quietly sharing the name.
        second = instance.SingleInstance(address, lambda: None)
        assert second.claim() is False
    finally:
        guard.release()
    # Released means gone: the next launch becomes the window, not a client.
    assert instance.forward(address) is False


def test_release_makes_the_endpoint_claimable_again() -> None:
    address = private_address("second-life")
    first = instance.SingleInstance(address, lambda: None)
    assert first.claim()
    first.release()
    first.release()  # idempotent, promised by the docstring

    reborn = threading.Event()
    second = instance.SingleInstance(address, reborn.set)
    assert second.claim()
    try:
        assert instance.forward(address) is True
        assert reborn.wait(5)
    finally:
        second.release()


def test_a_stranger_on_the_endpoint_is_left_alone() -> None:
    # The name is derived, not reserved: whatever answers with the wrong
    # greeting is some other program, and gets no "activate" from us.
    address = private_address("stranger")
    listener = connection.Listener(address)

    def stranger() -> None:
        peer = listener.accept()
        try:
            peer.send_bytes(b"not-this-protocol 123")
            peer.poll(5)  # linger so the client can read before the close
        finally:
            peer.close()

    impostor = threading.Thread(target=stranger, daemon=True)
    impostor.start()
    try:
        assert instance.forward(address) is False
    finally:
        listener.close()
        impostor.join(timeout=5)


@pytest.mark.skipif(sys.platform == "win32", reason="pipes die with their process; no corpses")
def test_a_crashed_twins_socket_is_swept_not_obeyed() -> None:
    # A corpse: the path exists, nobody listens. A crash cannot be staged in a
    # test, but what it leaves behind can — anything on the path that refuses
    # connections looks exactly like a dead socket to the claim.
    #
    # A scratch directory of its own rather than pytest's tmp_path: sun_path
    # holds ~104 bytes on macOS, and tmp_path nests this test's whole name
    # under the runner's already-deep /var/folders — the bind then fails for
    # *length*, which reads as an unsweepable corpse. The real endpoints never
    # meet this: endpoint() builds short names directly under the runtime dir.
    scratch = Path(tempfile.mkdtemp(prefix="md-corpse-"))
    try:
        address = str(scratch / "corpse.sock")
        Path(address).touch()

        raised = threading.Event()
        guard = instance.SingleInstance(address, raised.set)
        assert guard.claim(), "a corpse must be swept, not treated as a twin"
        try:
            assert instance.forward(address) is True
            assert raised.wait(5)
        finally:
            guard.release()
        assert not Path(address).exists(), "release leaves the endpoint clean"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
