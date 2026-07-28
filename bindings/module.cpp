// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
// nanobind module: the simulation as a vectorised RL environment.
//
// Two things matter for throughput here, and both are deliberate:
//   * observations are written directly into caller-owned NumPy arrays, so a
//     rollout never copies a batch;
//   * the batch step releases the GIL, so the C++ worker pool actually runs in
//     parallel instead of taking turns.
#include "md/agent/eval.hpp"
#include "md/agent/handicap.hpp"
#include "md/agent/policy.hpp"
#include "md/config.hpp"
#include "md/observation.hpp"
#include "md/replay/recording.hpp"
#include "md/version.hpp"
#include "vec_env.hpp"

#include <cstdint>
#include <filesystem>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

using FloatArray = nb::ndarray<float, nb::numpy, nb::c_contig>;
using ConstFloatArray = nb::ndarray<const float, nb::numpy, nb::c_contig>;
using IntArray = nb::ndarray<const std::int32_t, nb::numpy, nb::c_contig>;
using BoolArray = nb::ndarray<bool, nb::numpy, nb::c_contig>;
using ConstBoolArray = nb::ndarray<const bool, nb::numpy, nb::c_contig>;
using OutIntArray = nb::ndarray<std::int32_t, nb::numpy, nb::c_contig>;

/// One `.mdp`, loaded once and played many times.
///
/// Scoring a model over the canonical 32 seeds is 32 episodes against the same
/// weights, and six megabytes re-read per episode would be most of the run.
///
/// It exists at all because inference has to happen **in C++**. A reference
/// forward pass in NumPy (`missile_defense.sim.export_policy.evaluate`) is what defines the
/// format and what the parity test holds the two implementations to, but it
/// runs one observation at a time from Python — measured at ~17 ms for the
/// relational architecture, which is four hours for one contestant's canonical
/// block. The same policy through `md::agent::Policy` is the code the game runs
/// at sixty frames a second, so a league contest takes minutes and a league
/// score is a number somebody will actually wait for.
class LoadedPolicy {
  public:
    explicit LoadedPolicy(const std::filesystem::path& path)
        : policy_(md::agent::Policy::load(path)) {
        // Built and thrown away: `PolicyDriver`'s constructor is where an
        // observation or action-count mismatch is caught, and that refusal
        // belongs at load time rather than part-way through the first episode
        // of a contest somebody is waiting on.
        const md::agent::PolicyDriver check{policy_, md::ObsSpec{}};
        (void) check;
    }

    /// Play one seed to termination or ``max_ticks``. Same `run_episode` the
    /// scripted baseline and `md_agent_eval` use, so the two are comparable by
    /// construction rather than by agreement.
    ///
    /// Comparable *including the handicap*, which is the half this used to leave
    /// out: it built a bare `PolicyDriver` while `md_agent_eval` wrapped its own
    /// in `HandicappedDriver` by default, so the league scored every contestant
    /// on an easier game than the one it recorded having played. The two
    /// disagreed by 340 points on the first canonical seed and nothing said so.
    [[nodiscard]] md::agent::EpisodeResult play(std::uint64_t seed, std::uint64_t max_ticks,
                                                unsigned decision_interval,
                                                md::agent::Handicap handicap) const {
        md::Config config{};
        config.decision_interval = decision_interval;
        md::agent::PolicyDriver driver{policy_, md::ObsSpec{}};
        if (!handicap.active()) {
            return md::agent::run_episode(config, seed, driver, max_ticks);
        }
        md::agent::HandicappedDriver handicapped{driver, handicap};
        return md::agent::run_episode(config, seed, handicapped, max_ticks);
    }

    [[nodiscard]] const md::agent::Policy& policy() const noexcept { return policy_; }

  private:
    md::agent::Policy policy_;
};

void require(bool ok, const char* what) {
    if (!ok) {
        throw std::invalid_argument(what);
    }
}

} // namespace

NB_MODULE(_md_native, m) {
    m.doc() = "missile-defense — deterministic simulation as a vectorised RL environment";
    m.attr("__version__") = std::string(md::version());

    nb::class_<md::Config>(m, "Config", "Tunable simulation constants (see DESIGN.md).")
        .def(nb::init<>())
        .def_rw("world_width", &md::Config::world_width)
        .def_rw("world_height", &md::Config::world_height)
        .def_rw("dt", &md::Config::dt)
        .def_rw("ammo_per_base", &md::Config::ammo_per_base)
        .def_rw("base_cooldown", &md::Config::base_cooldown)
        .def_rw("aim_max_speed", &md::Config::aim_max_speed)
        .def_rw("fire_interval", &md::Config::fire_interval)
        .def_rw("decision_interval", &md::Config::decision_interval)
        .def_rw("bonus_city_score", &md::Config::bonus_city_score)
        .def_rw("score_per_kill", &md::Config::score_per_kill)
        .def_rw("score_per_unused_interceptor", &md::Config::score_per_unused_interceptor)
        .def_rw("score_per_surviving_city", &md::Config::score_per_surviving_city);

    nb::class_<md::ObsSpec>(m, "ObsSpec", "How much of the field the observation exposes.")
        .def(nb::init<>())
        .def_rw("threats", &md::ObsSpec::threats)
        .def_rw("interceptors", &md::ObsSpec::interceptors)
        .def_rw("blasts", &md::ObsSpec::blasts)
        .def_prop_ro(
            "size", [](const md::ObsSpec& s) { return s.size(); }, "Floats per observation.");

    // ---- The M4 evaluation protocol, shared with the scripted baseline --------
    nb::class_<md::agent::EpisodeResult>(m, "EpisodeResult", "Outcome of one episode.")
        .def_ro("seed", &md::agent::EpisodeResult::seed)
        .def_ro("score", &md::agent::EpisodeResult::score)
        .def_ro("wave_reached", &md::agent::EpisodeResult::wave_reached)
        .def_ro("waves_cleared", &md::agent::EpisodeResult::waves_cleared)
        .def_ro("ticks", &md::agent::EpisodeResult::ticks)
        .def_ro("cities_left", &md::agent::EpisodeResult::cities_left)
        .def_ro("cities_lost", &md::agent::EpisodeResult::cities_lost)
        .def_ro("bases_left", &md::agent::EpisodeResult::bases_left)
        .def_ro("bases_lost", &md::agent::EpisodeResult::bases_lost)
        .def_ro("ammo_left", &md::agent::EpisodeResult::ammo_left)
        .def_ro("bonus_cities", &md::agent::EpisodeResult::bonus_cities)
        .def_ro("mirv_splits", &md::agent::EpisodeResult::mirv_splits)
        .def_ro("shots", &md::agent::EpisodeResult::shots)
        .def_ro("kills", &md::agent::EpisodeResult::kills)
        .def_ro("kills_per_shot", &md::agent::EpisodeResult::kills_per_shot)
        // The score, decomposed. These three sum to `score` exactly.
        .def_ro("kill_credit", &md::agent::EpisodeResult::kill_credit)
        .def_ro("city_credit", &md::agent::EpisodeResult::city_credit)
        .def_ro("ammo_credit", &md::agent::EpisodeResult::ammo_credit)
        .def_ro("terminated", &md::agent::EpisodeResult::terminated)
        .def_prop_ro("wasted", &md::agent::EpisodeResult::wasted)
        .def_prop_ro("hits", &md::agent::EpisodeResult::hits)
        .def_prop_ro("accuracy", &md::agent::EpisodeResult::accuracy)
        .def_prop_ro("hit_rate", &md::agent::EpisodeResult::hit_rate);

    nb::class_<md::agent::Summary>(m, "Summary", "Aggregate over a seed set.")
        .def_ro("episodes", &md::agent::Summary::episodes)
        .def_ro("mean_score", &md::agent::Summary::mean_score)
        .def_ro("mean_wave", &md::agent::Summary::mean_wave)
        .def_ro("mean_waves_cleared", &md::agent::Summary::mean_waves_cleared)
        .def_ro("mean_ticks", &md::agent::Summary::mean_ticks)
        .def_ro("mean_cities_left", &md::agent::Summary::mean_cities_left)
        .def_ro("mean_cities_lost", &md::agent::Summary::mean_cities_lost)
        .def_ro("mean_bases_left", &md::agent::Summary::mean_bases_left)
        .def_ro("mean_bases_lost", &md::agent::Summary::mean_bases_lost)
        .def_ro("mean_ammo_left", &md::agent::Summary::mean_ammo_left)
        .def_ro("mean_bonus_cities", &md::agent::Summary::mean_bonus_cities)
        .def_ro("mean_mirv_splits", &md::agent::Summary::mean_mirv_splits)
        .def_ro("mean_shots", &md::agent::Summary::mean_shots)
        .def_ro("mean_kills", &md::agent::Summary::mean_kills)
        .def_ro("mean_hits", &md::agent::Summary::mean_hits)
        .def_ro("mean_accuracy", &md::agent::Summary::mean_accuracy)
        .def_ro("mean_hit_rate", &md::agent::Summary::mean_hit_rate)
        // What the score was made of, and how fast the magazines went. The
        // shares are computed here rather than in Python so the scripted
        // baseline and a learned policy are described by one implementation.
        .def_ro("mean_kill_credit", &md::agent::Summary::mean_kill_credit)
        .def_ro("mean_city_credit", &md::agent::Summary::mean_city_credit)
        .def_ro("mean_ammo_credit", &md::agent::Summary::mean_ammo_credit)
        .def_ro("mean_shots_per_wave", &md::agent::Summary::mean_shots_per_wave)
        .def_prop_ro("kill_share", &md::agent::Summary::kill_share)
        .def_prop_ro("city_share", &md::agent::Summary::city_share)
        .def_prop_ro("ammo_share", &md::agent::Summary::ammo_share)
        .def_ro("min_score", &md::agent::Summary::min_score)
        .def_ro("max_score", &md::agent::Summary::max_score)
        .def_ro("survived", &md::agent::Summary::survived)
        .def_ro("kills_per_shot", &md::agent::Summary::kills_per_shot);

    m.def("default_seeds", &md::agent::default_seeds, nb::arg("count") = 32u,
          "The fixed deterministic seed stream; evaluation protocols select disjoint blocks.");
    m.def(
        "summarize",
        [](const std::vector<md::agent::EpisodeResult>& episodes) {
            return md::agent::summarize(episodes);
        },
        nb::arg("episodes"),
        "Aggregate episode outcomes with the same function the scripted baseline uses.");

    nb::class_<LoadedPolicy>(m, "LoadedPolicy", R"doc(
A promoted `.mdp`, ready to play, with inference in C++.

`missile_defense.sim.export_policy.evaluate` is the reference forward pass and stays the
definition of the format; this is the implementation the *game* runs, and the
only one fast enough to score a model over a whole seed block. The parity test
holds the two to the same action, decision for decision.
)doc")
        // A `str`, not a `std::filesystem::path`: nanobind's path caster needs
        // `nanobind/stl/filesystem.h`, and every caller here already has the
        // string form. One conversion, in one place.
        .def(
            "__init__",
            [](LoadedPolicy* self, const std::string& path) {
                new (self) LoadedPolicy{std::filesystem::path{path}};
            },
            nb::arg("path"), "Read and validate a .mdp. Raises if this build cannot run it.")
        .def_prop_ro("observation_size",
                     [](const LoadedPolicy& self) { return self.policy().observation_size(); })
        .def_prop_ro("action_count",
                     [](const LoadedPolicy& self) { return self.policy().action_count(); })
        .def_prop_ro(
            "architecture",
            [](const LoadedPolicy& self) { return std::string(self.policy().architecture()); })
        .def_prop_ro(
            "display_name",
            [](const LoadedPolicy& self) { return std::string(self.policy().display_name()); })
        .def(
            "play",
            [](const LoadedPolicy& self, std::uint64_t seed, std::uint64_t max_ticks,
               unsigned decision_interval, float aim_trail, std::uint32_t reaction_delay) {
                require(max_ticks > 0, "max_ticks must be positive");
                require(decision_interval > 0, "decision_interval must be positive");
                require(aim_trail >= 0.0F && aim_trail < 1.0F, "aim_trail must be in [0, 1)");
                nb::gil_scoped_release release; // minutes of simulation, no Python in it
                return self.play(
                    seed, max_ticks, decision_interval,
                    md::agent::Handicap{.reaction_delay = reaction_delay, .aim_trail = aim_trail});
            },
            nb::arg("seed"), nb::arg("max_ticks") = 120000U, nb::arg("decision_interval") = 4U,
            // The published handicap, exactly as `md_agent_eval` defaults to it:
            // an evaluation without it is not a comparable one, so opting out is
            // something a caller has to type.
            nb::arg("aim_trail") = md::agent::canonical_handicap.aim_trail,
            nb::arg("reaction_delay") = md::agent::canonical_handicap.reaction_delay,
            "Play one seed to termination or the cap, under the published handicap.")
        .def(
            "act",
            [](const LoadedPolicy& self, ConstFloatArray observation, ConstBoolArray legal) {
                require(observation.size() == self.policy().observation_size(),
                        "observation must be (observation_size,) float32");
                require(legal.size() == self.policy().action_count(),
                        "legal must be (action_count,) bool");
                // Copied rather than reinterpreted: `bool` and `std::uint8_t`
                // are the same width here and are still different types, and
                // `md::agent::PolicyDriver` copies for the same reason.
                std::vector<std::uint8_t> mask(legal.size());
                for (std::size_t i = 0; i < legal.size(); ++i) {
                    mask[i] = static_cast<std::uint8_t>(legal.data()[i]);
                }
                const std::span<const float> features{observation.data(), observation.size()};
                nb::gil_scoped_release release;
                return self.policy().act(features, mask).action;
            },
            nb::arg("observation"), nb::arg("legal"),
            "The action this policy would take, masked to the legal ones.");

    m.attr("MAX_CITIES") = md::max_cities;
    m.attr("BASE_COUNT") = md::base_count;
    m.attr("MAX_THREATS") = md::max_threats;

    nb::class_<md::rl::VecEnv>(m, "VecEnv", R"doc(
A batch of independent simulations with RL conventions applied.

The arrays passed to `step` are written in place; nothing is copied and nothing is
allocated per step. `step` releases the GIL, so the worker pool runs in parallel.
)doc")
        .def(nb::init<std::size_t, const md::Config&, const md::ObsSpec&, unsigned, unsigned,
                      std::uint64_t, float, unsigned>(),
             nb::arg("num_envs"), nb::arg("config") = md::Config{},
             nb::arg("obs_spec") = md::ObsSpec{}, nb::arg("threads") = 0u,
             nb::arg("frame_skip") = 4u, nb::arg("max_ticks") = 120000u,
             nb::arg("aim_trail") = 0.0f, nb::arg("reaction_delay") = 0u)
        .def_prop_ro("num_envs", &md::rl::VecEnv::num_envs)
        .def_prop_ro("obs_size", &md::rl::VecEnv::obs_size)
        .def_prop_ro("action_count", &md::rl::VecEnv::action_count)
        .def_prop_ro("threads", &md::rl::VecEnv::threads)
        .def_prop_ro("frame_skip", &md::rl::VecEnv::frame_skip)
        .def_prop_ro("aim_trail", &md::rl::VecEnv::aim_trail)
        .def_prop_ro("reaction_delay", &md::rl::VecEnv::reaction_delay)
        .def(
            "reset",
            [](md::rl::VecEnv& env, std::uint64_t seed, FloatArray obs) {
                require(obs.size() == env.num_envs() * env.obs_size(),
                        "obs must be (num_envs, obs_size)");
                float* data = obs.data();
                nb::gil_scoped_release release;
                env.reset(seed, data);
            },
            nb::arg("seed"), nb::arg("obs"), "Seed every env and fill `obs` in place.")
        .def(
            "reset_seeds",
            [](md::rl::VecEnv& env, const std::vector<std::uint64_t>& seeds, FloatArray obs) {
                require(obs.size() == env.num_envs() * env.obs_size(),
                        "obs must be (num_envs, obs_size)");
                require(!seeds.empty(), "seeds must not be empty");
                float* data = obs.data();
                nb::gil_scoped_release release;
                env.reset(std::span<const std::uint64_t>{seeds}, data);
            },
            nb::arg("seeds"), nb::arg("obs"),
            "Seed each env explicitly — for evaluating on the canonical seed set.")
        .def(
            "take_episode_result",
            [](md::rl::VecEnv& env, std::size_t index) { return env.take_episode_result(index); },
            nb::arg("index"), "The outcome of the last episode this env finished, or None.")
        .def(
            "step",
            [](md::rl::VecEnv& env, IntArray actions, FloatArray obs, FloatArray final_obs,
               FloatArray rewards, BoolArray terminated, BoolArray truncated) {
                const std::size_t n = env.num_envs();
                require(actions.size() == n, "actions must be (num_envs,) int32");
                require(obs.size() == n * env.obs_size(), "obs must be (num_envs, obs_size)");
                require(final_obs.size() == n * env.obs_size(),
                        "final_obs must be (num_envs, obs_size)");
                require(rewards.size() == n, "rewards must be (num_envs,) float32");
                require(terminated.size() == n, "terminated must be (num_envs,) bool");
                require(truncated.size() == n, "truncated must be (num_envs,) bool");

                const std::int32_t* a = actions.data();
                float* o = obs.data();
                float* f = final_obs.data();
                float* r = rewards.data();
                bool* term = terminated.data();
                bool* trunc = truncated.data();

                nb::gil_scoped_release release; // the whole point: real parallelism
                env.step(a, o, f, r, term, trunc);
            },
            nb::arg("actions"), nb::arg("obs"), nb::arg("final_obs"), nb::arg("rewards"),
            nb::arg("terminated"), nb::arg("truncated"),
            "Advance every env by up to frame_skip ticks without crossing max_ticks; "
            "rewards and event features cover the whole window.")
        .def(
            "shot_stats",
            [](const md::rl::VecEnv& env, OutIntArray wasted, OutIntArray multi_kills) {
                require(wasted.size() == env.num_envs(), "wasted must be (num_envs,)");
                require(multi_kills.size() == env.num_envs(), "multi_kills must be (num_envs,)");
                std::int32_t* w = wasted.data();
                std::int32_t* m = multi_kills.data();
                nb::gil_scoped_release release;
                env.shot_stats(w, m);
            },
            nb::arg("wasted"), nb::arg("multi_kills"),
            "How the last step spent its ammunition: blasts that killed nothing, "
            "and kills beyond a blast's first.")
        .def(
            "action_masks",
            [](const md::rl::VecEnv& env, BoolArray mask) {
                require(mask.size() == env.num_envs() * env.action_count(),
                        "mask must be (num_envs, action_count)");
                bool* data = mask.data();
                nb::gil_scoped_release release;
                env.action_masks(data);
            },
            nb::arg("mask"), "Fill `mask` with which actions are legal per env.")
        .def(
            "record",
            [](md::rl::VecEnv& env, std::size_t index, bool on) { env.set_recording(index, on); },
            nb::arg("index"), nb::arg("on") = true,
            "Log this env's actions from episode tick 0 so it can be watched in the app.")
        .def(
            "is_recording",
            [](const md::rl::VecEnv& env, std::size_t index) { return env.is_recording(index); },
            nb::arg("index"))
        .def(
            "save_recording",
            [](md::rl::VecEnv& env, std::size_t index, const std::string& path,
               std::uint64_t update, const std::string& label) {
                // Only whole episodes are handed over, so a False here means "none
                // finished yet", not an error.
                std::optional<md::replay::Recording> recording = env.take_recording(index);
                if (!recording.has_value()) {
                    return false;
                }
                recording->update = update;
                recording->set_label(label);
                return md::replay::save(*recording, std::filesystem::path{path});
            },
            nb::arg("index"), nb::arg("path"), nb::arg("update") = 0u, nb::arg("label") = "",
            "Write the last completed episode for `index`; False if none is ready.");
}
