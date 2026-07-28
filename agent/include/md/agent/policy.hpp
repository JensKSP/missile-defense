// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace md::agent {

/// A learned policy, loaded from an `.mdp` and run without a Python in sight.
///
/// **The whole point is that this needs no torch and no interpreter.** The game
/// is C++ with neither in it — that is the promise `debian/control` keeps — so a
/// trained agent can only reach a player through a format the game can read on
/// its own. `python/missile_defense/sim/policy_format.py` writes those files and
/// `docs/API.md` §7 states what they promise; this is the reader.
///
/// **It is also a reader of files a person may have downloaded**, which is why
/// `.pt` is not the import format: loading a pickle runs its author's code. Every
/// offset in an `.mdp` is bounds-checked against the payload before a byte is
/// read, the payload is checksummed, and a file that fails any check throws
/// rather than being partly believed.
///
/// Only the `mlp` architecture exists here, deliberately. An interpreter for
/// arbitrary computation graphs is a far larger thing to get right than the one
/// network this project trains, and the format names its architecture precisely
/// so that both sides can refuse what the other cannot run.
class Policy {
  public:
    /// One forward pass: what the network would do, and what it thinks of here.
    struct Decision {
        /// Index into the discrete action space (`md::action_count`). Always a
        /// *legal* one — see `act`.
        std::uint32_t action = 0;
        /// The critic's estimate. Carried because an evaluator that logs it can
        /// tell "played badly" from "knew it was losing", and it costs one dot
        /// product that has already been computed.
        float value = 0.0F;
    };

    /// Parse `path`. Throws `Policy::Error` with a message naming the file and
    /// the failed check — this is an error a person meets when a download went
    /// wrong, so it has to explain itself.
    [[nodiscard]] static Policy load(const std::filesystem::path& path);

    /// What a policy file says about itself, without its weights.
    struct Description {
        std::string display_name;
        std::uint32_t schema = 0;
        std::size_t observation_size = 0;
        std::size_t action_count = 0;
        std::string architecture;
    };

    /// Read only the manifest of `path`.
    ///
    /// For *listing* models: a browser with eight entries would otherwise read
    /// eight multi-megabyte tensor blocks and verify eight checksums to put
    /// eight names on a screen. Throws `Policy::Error` the same way `load` does
    /// — a file this cannot describe is one `load` would refuse anyway.
    [[nodiscard]] static Description describe(const std::filesystem::path& path);

    /// The action this policy would take. `legal` is `md::action_mask`'s output.
    ///
    /// The masking rule, which `missile_defense.sim.export_policy` states identically because the
    /// two must not drift: compute the logits, overwrite every illegal action's
    /// with `masked_logit`, then take the **first** maximum. Ties going to the
    /// lowest index is a promise, not an accident of `std::max_element` — the
    /// parity fixture would be a coin flip otherwise.
    ///
    /// Allocation-free after the first call on a thread, and `const`: the
    /// scratch is thread-local, so one `Policy` may be shared by a worker pool.
    [[nodiscard]] Decision act(std::span<const float> observation,
                               std::span<const std::uint8_t> legal) const;

    /// The same, and the masked logits as well. For the parity tests, which have
    /// to compare the whole vector rather than only the argmax of it — two
    /// implementations can agree on the chosen action and disagree everywhere.
    [[nodiscard]] Decision act(std::span<const float> observation,
                               std::span<const std::uint8_t> legal,
                               std::span<float> logits_out) const;

    /// What an illegal action's logit becomes. Matches `missile_defense.sim.export_policy`.
    static constexpr float masked_logit = -1.0e8F;

    [[nodiscard]] std::uint32_t schema() const noexcept { return schema_; }

    [[nodiscard]] std::size_t observation_size() const noexcept { return observation_size_; }

    [[nodiscard]] std::size_t action_count() const noexcept { return action_count_; }

    [[nodiscard]] std::string_view architecture() const noexcept { return architecture_; }

    /// What to call this agent on screen while it plays, or empty when the file
    /// does not say. A path is not a name: `policy-best.pt` says nothing about
    /// which run produced it, so the name travels inside the file.
    [[nodiscard]] std::string_view display_name() const noexcept { return display_name_; }

    /// A `.mdp` that could not be read, and why.
    class Error : public std::runtime_error {
      public:
        using std::runtime_error::runtime_error;
    };

  private:
    Policy() = default;

    /// One layer's parameters, already laid out row-major as the file had them.
    struct Layer {
        std::vector<float> weight; // out * in, row-major
        std::vector<float> bias;   // out
        std::size_t inputs = 0;
        std::size_t outputs = 0;
    };

    /// One cross-attention block. Single-headed, so the head dimension *is*
    /// `width_` and the score scale is `1/sqrt(width_)` — the default
    /// `torch.nn.functional.scaled_dot_product_attention` applies, which is the
    /// number parity depends on. The three projections carry no bias, matching
    /// `missile_defense.training.ppo._CrossAttention`; only `output` has one.
    struct Attention {
        std::vector<float> query; // width * width, row-major
        std::vector<float> key;
        std::vector<float> value;
        Layer output;
    };

    /// The relational forward pass, writing masked-but-not-yet-argmaxed logits.
    /// Split out because it is an order of magnitude more code than the flat one
    /// and shares only the masking tail with it.
    void entity_logits(std::span<const float> observation, std::span<float> logits_out,
                       float& value_out) const;

    std::uint32_t schema_ = 0;
    std::size_t observation_size_ = 0;
    std::size_t action_count_ = 0;
    std::size_t hidden_ = 0;
    std::string architecture_;
    std::string display_name_;

    // --- mlp ---
    Layer trunk0_;
    Layer trunk1_;
    Layer policy_head_;
    Layer value_head_;

    // --- entity. Unused by `mlp`, and vice versa; which set is live is decided
    // once at load time by `architecture_` and never re-examined per decision.
    Layer threat_encoder0_;
    Layer threat_encoder1_;
    Layer interceptor_encoder0_;
    Layer interceptor_encoder1_;
    Layer blast_encoder0_;
    Layer blast_encoder1_;
    Attention interceptor_attention_;
    Attention blast_attention_;
    Layer actor_context0_;
    Layer actor_context1_;
    Layer context_to_threat_;
    Layer relation0_;
    Layer relation1_;
    Layer fire_head_;
    Layer noop_head_;
    Layer critic_trunk0_;
    Layer critic_trunk1_;
    std::size_t width_ = 0;
    std::size_t batteries_ = 0;
    std::size_t threats_ = 0;
    std::size_t interceptors_ = 0;
    std::size_t blasts_ = 0;
    std::size_t globals_width_ = 0;
};

} // namespace md::agent
