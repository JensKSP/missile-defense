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
#include "md/config.hpp"
#include "md/observation.hpp"
#include "md/replay/recording.hpp"
#include "md/version.hpp"
#include "vec_env.hpp"

#include <cstdint>
#include <filesystem>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

using FloatArray = nb::ndarray<float, nb::numpy, nb::c_contig>;
using IntArray = nb::ndarray<const std::int32_t, nb::numpy, nb::c_contig>;
using BoolArray = nb::ndarray<bool, nb::numpy, nb::c_contig>;

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

    m.attr("MAX_CITIES") = md::max_cities;
    m.attr("BASE_COUNT") = md::base_count;
    m.attr("MAX_THREATS") = md::max_threats;

    nb::class_<md::rl::VecEnv>(m, "VecEnv", R"doc(
A batch of independent simulations with RL conventions applied.

The arrays passed to `step` are written in place; nothing is copied and nothing is
allocated per step. `step` releases the GIL, so the worker pool runs in parallel.
)doc")
        .def(nb::init<std::size_t, const md::Config&, const md::ObsSpec&, unsigned, unsigned,
                      std::uint64_t>(),
             nb::arg("num_envs"), nb::arg("config") = md::Config{},
             nb::arg("obs_spec") = md::ObsSpec{}, nb::arg("threads") = 0u,
             nb::arg("frame_skip") = 4u, nb::arg("max_ticks") = 120000u)
        .def_prop_ro("num_envs", &md::rl::VecEnv::num_envs)
        .def_prop_ro("obs_size", &md::rl::VecEnv::obs_size)
        .def_prop_ro("action_count", &md::rl::VecEnv::action_count)
        .def_prop_ro("threads", &md::rl::VecEnv::threads)
        .def_prop_ro("frame_skip", &md::rl::VecEnv::frame_skip)
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
            "Advance every env by frame_skip ticks, writing all outputs in place.")
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
            "Log this env's actions so the episode can be watched in the app.")
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
