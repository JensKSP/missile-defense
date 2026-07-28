// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/agent/policy.hpp"

#include "md/protocol.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <md/observation.hpp>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace md::agent {

namespace {

using json = nlohmann::json;

/// Mirrors `missile_defense.policy_format`. Any disagreement here is a file one side writes
/// and the other refuses, so the constants are stated rather than derived.
// Both from protocol.toml: the trainer writes this container and this
// reads it, in two languages that cannot share a constant.
constexpr std::string_view magic = protocol::policy_magic;
constexpr std::uint32_t container_version = protocol::policy_container_version;
constexpr std::uint32_t supported_schema = 1;
constexpr std::string_view dtype = "<f4";
constexpr std::size_t itemsize = 4;

/// SHA-256, because the manifest carries one and a policy whose weights have
/// quietly rotted plays slightly worse rather than failing. Implemented here
/// rather than pulled in: the alternative is a dependency on OpenSSL for sixty
/// lines, and this is the only hash in the whole program.
class Sha256 {
  public:
    void update(std::span<const std::byte> data) {
        for (const std::byte value : data) {
            buffer_[fill_++] = std::to_integer<std::uint8_t>(value);
            if (fill_ == block) {
                compress();
                fill_ = 0;
            }
            ++length_;
        }
    }

    [[nodiscard]] std::string hex() {
        const std::uint64_t bits = length_ * 8;
        const std::array<std::byte, 1> one{std::byte{0x80}};
        update(one);
        const std::array<std::byte, 1> zero{std::byte{0x00}};
        while (fill_ != block - 8) {
            update(zero);
            --length_; // padding is not message length
        }
        for (int shift = 56; shift >= 0; shift -= 8) {
            buffer_[fill_++] = static_cast<std::uint8_t>((bits >> shift) & 0xFFu);
        }
        compress();

        std::string out;
        out.reserve(64);
        constexpr std::string_view digits = "0123456789abcdef";
        for (const std::uint32_t word : state_) {
            for (int shift = 28; shift >= 0; shift -= 4) {
                out.push_back(digits[(word >> shift) & 0xFu]);
            }
        }
        return out;
    }

  private:
    static constexpr std::size_t block = 64;

    static std::uint32_t rotate(std::uint32_t value, int by) { return std::rotr(value, by); }

    void compress() {
        static constexpr std::array<std::uint32_t, 64> k{
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};

        std::array<std::uint32_t, 64> w{};
        for (std::size_t i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(buffer_[(i * 4) + 0]) << 24) |
                   (static_cast<std::uint32_t>(buffer_[(i * 4) + 1]) << 16) |
                   (static_cast<std::uint32_t>(buffer_[(i * 4) + 2]) << 8) |
                   static_cast<std::uint32_t>(buffer_[(i * 4) + 3]);
        }
        for (std::size_t i = 16; i < 64; ++i) {
            const std::uint32_t s0 =
                rotate(w[i - 15], 7) ^ rotate(w[i - 15], 18) ^ (w[i - 15] >> 3u);
            const std::uint32_t s1 =
                rotate(w[i - 2], 17) ^ rotate(w[i - 2], 19) ^ (w[i - 2] >> 10u);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        std::array<std::uint32_t, 8> v = state_;
        for (std::size_t i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotate(v[4], 6) ^ rotate(v[4], 11) ^ rotate(v[4], 25);
            const std::uint32_t ch = (v[4] & v[5]) ^ (~v[4] & v[6]);
            const std::uint32_t t1 = v[7] + s1 + ch + k[i] + w[i];
            const std::uint32_t s0 = rotate(v[0], 2) ^ rotate(v[0], 13) ^ rotate(v[0], 22);
            const std::uint32_t maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            const std::uint32_t t2 = s0 + maj;
            v[7] = v[6];
            v[6] = v[5];
            v[5] = v[4];
            v[4] = v[3] + t1;
            v[3] = v[2];
            v[2] = v[1];
            v[1] = v[0];
            v[0] = t1 + t2;
        }
        for (std::size_t i = 0; i < 8; ++i) {
            state_[i] += v[i];
        }
    }

    std::array<std::uint32_t, 8> state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<std::uint8_t, block> buffer_{};
    std::size_t fill_ = 0;
    std::uint64_t length_ = 0;
};

std::string sha256_hex(std::span<const std::byte> data) {
    Sha256 hash;
    hash.update(data);
    return hash.hex();
}

[[noreturn]] void fail(const std::filesystem::path& path, const std::string& why) {
    throw Policy::Error{path.string() + ": " + why};
}

std::uint32_t read_u32(std::span<const std::byte> bytes, std::size_t offset) {
    // Assembled byte by byte rather than memcpy'd: the file is little-endian by
    // specification, and this is the same on a big-endian machine.
    return std::to_integer<std::uint32_t>(bytes[offset]) |
           (std::to_integer<std::uint32_t>(bytes[offset + 1]) << 8u) |
           (std::to_integer<std::uint32_t>(bytes[offset + 2]) << 16u) |
           (std::to_integer<std::uint32_t>(bytes[offset + 3]) << 24u);
}

float read_f32(std::span<const std::byte> bytes, std::size_t offset) {
    const std::uint32_t raw = read_u32(bytes, offset);
    // `bit_cast` and not a reinterpret: the latter is an aliasing violation that
    // happens to work, and this file already refuses undefined behaviour it
    // could have got away with.
    return std::bit_cast<float>(raw);
}

/// A tensor named in the manifest, already bounds-checked against the payload.
struct Located {
    std::size_t offset = 0;
    std::vector<std::size_t> shape;
};

std::size_t elements(const std::vector<std::size_t>& shape) {
    std::size_t total = 1;
    for (const std::size_t extent : shape) {
        total *= extent;
    }
    return total;
}

} // namespace

Policy::Description Policy::describe(const std::filesystem::path& path) {
    std::ifstream file{path, std::ios::binary};
    if (!file) {
        fail(path, "could not be opened");
    }
    // Only the front of the file, unlike `load`: this exists so that listing
    // models costs a few hundred bytes each instead of every tensor in them.
    const std::size_t header = magic.size() + 8;
    std::string front(header, '\0');
    if (!file.read(front.data(), static_cast<std::streamsize>(header))) {
        fail(path, "truncated — shorter than the header");
    }
    if (std::memcmp(front.data(), magic.data(), magic.size()) != 0) {
        fail(path, "not a policy file (bad magic)");
    }
    const std::span<const std::byte> bytes = std::as_bytes(std::span{front});
    const std::uint32_t container = read_u32(bytes, magic.size());
    if (container != container_version) {
        fail(path, "container version " + std::to_string(container) + ", this build reads " +
                       std::to_string(container_version));
    }
    const std::uint32_t manifest_length = read_u32(bytes, magic.size() + 4);

    std::string text(manifest_length, '\0');
    if (manifest_length > 0 &&
        !file.read(text.data(), static_cast<std::streamsize>(manifest_length))) {
        fail(path, "truncated — the manifest runs past the end of the file");
    }

    json manifest;
    try {
        manifest = json::parse(text);
    } catch (const json::exception& error) {
        fail(path, std::string{"the manifest is not readable JSON ("} + error.what() + ")");
    }
    if (!manifest.is_object()) {
        fail(path, "the manifest is not an object");
    }

    Description described;
    try {
        described.schema = manifest.at("schema").get<std::uint32_t>();
        described.observation_size = manifest.at("observation_size").get<std::size_t>();
        described.action_count = manifest.at("action_count").get<std::size_t>();
        described.architecture = manifest.at("architecture").get<std::string>();
        // Under `metadata`, exactly where `load` looks for it — a second rule
        // here would show one name in the browser and another in the HUD.
        if (const auto metadata = manifest.find("metadata");
            metadata != manifest.end() && metadata->is_object()) {
            if (const auto name = metadata->find("display_name");
                name != metadata->end() && name->is_string()) {
                described.display_name = name->get<std::string>();
            }
        }
    } catch (const json::exception& error) {
        fail(path,
             std::string{"the manifest is missing or misdescribes a field ("} + error.what() + ")");
    }
    return described;
}

Policy Policy::load(const std::filesystem::path& path) {
    std::ifstream file{path, std::ios::binary};
    if (!file) {
        fail(path, "could not be opened");
    }
    const std::string raw{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
    // `as_bytes` rather than a reinterpret_cast: same view, but it is the
    // standard's own way of saying "these are just bytes now", and this file
    // already declines to rely on undefined behaviour it could have got away
    // with (see `bit_cast` above).
    const std::span<const std::byte> bytes = std::as_bytes(std::span{raw});

    const std::size_t header = magic.size() + 8;
    if (bytes.size() < header) {
        fail(path, "truncated — shorter than the header");
    }
    if (std::memcmp(raw.data(), magic.data(), magic.size()) != 0) {
        fail(path, "not a policy file (bad magic)");
    }
    const std::uint32_t container = read_u32(bytes, magic.size());
    if (container != container_version) {
        fail(path, "container version " + std::to_string(container) + ", this build reads " +
                       std::to_string(container_version));
    }
    const std::uint32_t manifest_length = read_u32(bytes, magic.size() + 4);
    if (bytes.size() < header + manifest_length) {
        fail(path, "truncated — the manifest runs past the end of the file");
    }

    json manifest;
    try {
        manifest = json::parse(raw.begin() + static_cast<std::ptrdiff_t>(header),
                               raw.begin() + static_cast<std::ptrdiff_t>(header) +
                                   static_cast<std::ptrdiff_t>(manifest_length));
    } catch (const json::exception& error) {
        fail(path, std::string{"the manifest is not readable JSON ("} + error.what() + ")");
    }
    if (!manifest.is_object()) {
        fail(path, "the manifest is not an object");
    }

    const std::span<const std::byte> payload = bytes.subspan(header + manifest_length);

    Policy policy;
    try {
        policy.schema_ = manifest.at("schema").get<std::uint32_t>();
        policy.observation_size_ = manifest.at("observation_size").get<std::size_t>();
        policy.action_count_ = manifest.at("action_count").get<std::size_t>();
        policy.architecture_ = manifest.at("architecture").get<std::string>();

        const auto declared = manifest.at("payload_size").get<std::size_t>();
        if (payload.size() != declared) {
            fail(path, "truncated — the manifest declares " + std::to_string(declared) +
                           " bytes of weights and the file holds " +
                           std::to_string(payload.size()));
        }
        if (sha256_hex(payload) != manifest.at("checksum").get<std::string>()) {
            fail(path, "checksum mismatch — the weights are corrupt");
        }
    } catch (const json::exception& error) {
        fail(path, std::string{"the manifest is missing a required field ("} + error.what() + ")");
    }

    if (policy.schema_ != supported_schema) {
        fail(path, "schema " + std::to_string(policy.schema_) + " — this build reads schema " +
                       std::to_string(supported_schema) +
                       ". The observation encoding or action space has changed; re-export the "
                       "checkpoint against this build.");
    }
    if (policy.architecture_ != "mlp" && policy.architecture_ != "entity") {
        fail(path, "architecture '" + policy.architecture_ +
                       "' is not one this build can run (mlp, entity)");
    }
    if (policy.observation_size_ == 0 || policy.action_count_ == 0) {
        fail(path, "observation size and action count must both be positive");
    }

    // Locate every tensor, bounds-checking before anything is read. This is the
    // check that keeps a hand-edited manifest from reading somebody else's
    // memory, and it is done *before* the slice rather than after.
    std::vector<std::pair<std::string, Located>> located;
    for (const json& entry : manifest.at("tensors")) {
        if (!entry.is_object()) {
            fail(path, "a tensor entry is not an object");
        }
        std::string name;
        Located found;
        std::size_t length = 0;
        try {
            name = entry.at("name").get<std::string>();
            if (entry.at("dtype").get<std::string>() != dtype) {
                fail(path, name + " is not " + std::string{dtype});
            }
            for (const json& extent : entry.at("shape")) {
                const auto value = extent.get<std::int64_t>();
                if (value <= 0) {
                    fail(path, name + " has a non-positive extent");
                }
                found.shape.push_back(static_cast<std::size_t>(value));
            }
            found.offset = entry.at("offset").get<std::size_t>();
            length = entry.at("bytes").get<std::size_t>();
        } catch (const json::exception& error) {
            fail(path, std::string{"a tensor entry is malformed ("} + error.what() + ")");
        }
        if (found.shape.empty()) {
            fail(path, name + " has no shape");
        }
        if (length != elements(found.shape) * itemsize) {
            fail(path, name + " declares a shape and reserves a different number of bytes");
        }
        // Overflow-safe: `offset + length` could wrap on a hostile manifest, so
        // the subtraction is done on the side that cannot.
        if (found.offset > payload.size() || length > payload.size() - found.offset) {
            fail(path, name + " claims bytes past the end of the payload");
        }
        for (const auto& [seen, _] : located) {
            if (seen == name) {
                fail(path, "duplicate tensor name '" + name + "'");
            }
        }
        located.emplace_back(name, std::move(found));
    }

    const auto take = [&](const std::string& name, std::size_t rank) -> Located {
        for (const auto& [seen, found] : located) {
            if (seen == name) {
                if (found.shape.size() != rank) {
                    fail(path, name + " has the wrong number of dimensions");
                }
                return found;
            }
        }
        fail(path, "no tensor named '" + name + "', which " + policy.architecture_ + " needs");
    };

    const auto layer = [&](const std::string& prefix) -> Layer {
        const Located weight = take(prefix + ".weight", 2);
        const Located bias = take(prefix + ".bias", 1);
        Layer built;
        built.outputs = weight.shape[0];
        built.inputs = weight.shape[1];
        if (bias.shape[0] != built.outputs) {
            fail(path, prefix + " has a bias that does not match its weight");
        }
        built.weight.resize(built.outputs * built.inputs);
        for (std::size_t i = 0; i < built.weight.size(); ++i) {
            built.weight[i] = read_f32(payload, weight.offset + (i * itemsize));
            if (!std::isfinite(built.weight[i])) {
                // A NaN propagates to every logit, so the policy plays uniformly
                // at random and merely looks bad rather than broken.
                fail(path, prefix + ".weight contains a non-finite value");
            }
        }
        built.bias.resize(built.outputs);
        for (std::size_t i = 0; i < built.bias.size(); ++i) {
            built.bias[i] = read_f32(payload, bias.offset + (i * itemsize));
            if (!std::isfinite(built.bias[i])) {
                fail(path, prefix + ".bias contains a non-finite value");
            }
        }
        return built;
    };

    // A projection with no bias — the three inside a cross-attention block.
    const auto projection = [&](const std::string& name, std::size_t width) -> std::vector<float> {
        const Located found = take(name, 2);
        if (found.shape[0] != width || found.shape[1] != width) {
            fail(path, name + " is not " + std::to_string(width) + "x" + std::to_string(width));
        }
        std::vector<float> values(width * width);
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = read_f32(payload, found.offset + (i * itemsize));
            if (!std::isfinite(values[i])) {
                fail(path, name + " contains a non-finite value");
            }
        }
        return values;
    };

    if (policy.architecture_ == "mlp") {
        policy.trunk0_ = layer("trunk.0");
        policy.trunk1_ = layer("trunk.2");
        policy.policy_head_ = layer("policy_head");
        policy.value_head_ = layer("value_head");
        policy.hidden_ = policy.trunk0_.outputs;

        // The dimensions have to chain, and they have to chain into the sizes the
        // manifest claims. Otherwise a file whose weights are internally consistent
        // but describe a different observation would be accepted and then read a
        // shorter observation than it expects, one row at a time.
        const bool consistent =
            policy.trunk0_.inputs == policy.observation_size_ &&
            policy.trunk1_.inputs == policy.hidden_ && policy.trunk1_.outputs == policy.hidden_ &&
            policy.policy_head_.inputs == policy.hidden_ &&
            policy.policy_head_.outputs == policy.action_count_ &&
            policy.value_head_.inputs == policy.hidden_ && policy.value_head_.outputs == 1;
        if (!consistent) {
            fail(path, "the layer dimensions do not chain into the declared observation size and "
                       "action count");
        }
    } else {
        policy.threat_encoder0_ = layer("threat_encoder.0");
        policy.threat_encoder1_ = layer("threat_encoder.2");
        policy.interceptor_encoder0_ = layer("interceptor_encoder.0");
        policy.interceptor_encoder1_ = layer("interceptor_encoder.2");
        policy.blast_encoder0_ = layer("blast_encoder.0");
        policy.blast_encoder1_ = layer("blast_encoder.2");
        policy.actor_context0_ = layer("actor_context.0");
        policy.actor_context1_ = layer("actor_context.2");
        policy.context_to_threat_ = layer("context_to_threat");
        policy.relation0_ = layer("relation.0");
        policy.relation1_ = layer("relation.2");
        policy.fire_head_ = layer("fire_head");
        policy.noop_head_ = layer("noop_head");
        policy.critic_trunk0_ = layer("critic_trunk.0");
        policy.critic_trunk1_ = layer("critic_trunk.2");
        policy.value_head_ = layer("value_head");

        policy.width_ = policy.threat_encoder0_.outputs;
        policy.hidden_ = policy.actor_context0_.outputs;
        policy.batteries_ = policy.fire_head_.outputs;
        for (Attention* block : {&policy.interceptor_attention_, &policy.blast_attention_}) {
            const std::string prefix = block == &policy.interceptor_attention_
                                           ? "interceptor_attention"
                                           : "blast_attention";
            block->query = projection(prefix + ".query.weight", policy.width_);
            block->key = projection(prefix + ".key.weight", policy.width_);
            block->value = projection(prefix + ".value.weight", policy.width_);
            block->output = layer(prefix + ".output");
        }

        // Which slots the observation holds is a fact about `md::encode`, not about
        // the file, so it comes from the simulation this binary was compiled with.
        // A policy trained against a different spec has a different observation size
        // and is refused here rather than silently sliced at the wrong offsets.
        const ObsSpec spec{};
        policy.threats_ = spec.threats;
        policy.interceptors_ = spec.interceptors;
        policy.blasts_ = spec.blasts;
        const std::size_t entity_floats = (policy.threats_ * ObsSpec::threat_features) +
                                          (policy.interceptors_ * ObsSpec::interceptor_features) +
                                          (policy.blasts_ * ObsSpec::blast_features);
        if (spec.size() != policy.observation_size_ || entity_floats >= policy.observation_size_) {
            fail(path, "this policy expects an observation of " +
                           std::to_string(policy.observation_size_) +
                           " floats and this build encodes " + std::to_string(spec.size()) +
                           ". It was trained against a different simulation; re-export it.");
        }
        policy.globals_width_ = policy.observation_size_ - entity_floats;

        const std::size_t w = policy.width_;
        const bool consistent =
            policy.batteries_ > 0 &&
            policy.action_count_ == 1 + (policy.batteries_ * policy.threats_) &&
            policy.threat_encoder0_.inputs == ObsSpec::threat_features &&
            policy.interceptor_encoder0_.inputs == ObsSpec::interceptor_features &&
            policy.blast_encoder0_.inputs == ObsSpec::blast_features &&
            policy.threat_encoder1_.inputs == w && policy.threat_encoder1_.outputs == w &&
            policy.interceptor_encoder0_.outputs == w && policy.interceptor_encoder1_.inputs == w &&
            policy.interceptor_encoder1_.outputs == w && policy.blast_encoder0_.outputs == w &&
            policy.blast_encoder1_.inputs == w && policy.blast_encoder1_.outputs == w &&
            policy.actor_context0_.inputs == policy.globals_width_ + (2 * w) &&
            policy.actor_context1_.inputs == policy.hidden_ &&
            policy.actor_context1_.outputs == policy.hidden_ &&
            policy.context_to_threat_.inputs == policy.hidden_ &&
            policy.context_to_threat_.outputs == w && policy.relation0_.inputs == 4 * w &&
            policy.relation0_.outputs == w && policy.relation1_.inputs == w &&
            policy.relation1_.outputs == w && policy.fire_head_.inputs == w &&
            policy.noop_head_.inputs == policy.hidden_ && policy.noop_head_.outputs == 1 &&
            policy.critic_trunk0_.inputs == policy.observation_size_ &&
            policy.critic_trunk0_.outputs == policy.hidden_ &&
            policy.critic_trunk1_.inputs == policy.hidden_ &&
            policy.critic_trunk1_.outputs == policy.hidden_ &&
            policy.value_head_.inputs == policy.hidden_ && policy.value_head_.outputs == 1;
        if (!consistent) {
            fail(path, "the layer dimensions do not chain into the declared observation size and "
                       "action count");
        }
    }

    if (const auto metadata = manifest.find("metadata");
        metadata != manifest.end() && metadata->is_object()) {
        if (const auto name = metadata->find("display_name");
            name != metadata->end() && name->is_string()) {
            policy.display_name_ = name->get<std::string>();
        }
    }
    return policy;
}

namespace {

/// Per-thread scratch, so `act` allocates nothing after the first call and a
/// `Policy` stays immutable — one may be shared by a worker pool.
struct Scratch {
    std::vector<float> hidden;
    std::vector<float> features;
    std::vector<float> logits;
    // The relational path. Every one of these is `resize`d to the same extent on
    // every call, so only the first on a thread allocates.
    std::vector<float> encoded_threats;
    std::vector<float> encoded_interceptors;
    std::vector<float> encoded_blasts;
    std::vector<float> slot;
    std::vector<float> context_input;
    std::vector<float> summary;
    std::vector<float> episode;
    std::vector<float> query;
    std::vector<float> scores;
    std::vector<float> attended;
    std::vector<float> relation_input;
    std::vector<float> per_threat;
    std::vector<float> fire;
};

Scratch& scratch() {
    thread_local Scratch buffers;
    return buffers;
}

/// `out = tanh(W x + b)` when `activate`, else the affine part alone.
///
/// float32 throughout, deliberately. Accumulating in double would put this a few
/// ULPs from `missile_defense.export_policy.evaluate`, which is invisible until it flips an
/// argmax between two near-equal logits in one seed out of a hundred — and the
/// parity test that caught it would look flaky rather than wrong.
void forward(const std::vector<float>& weight, const std::vector<float>& bias,
             std::span<const float> in, std::span<float> out, bool activate) {
    const std::size_t inputs = in.size();
    for (std::size_t row = 0; row < out.size(); ++row) {
        float sum = bias[row];
        const std::size_t base = row * inputs;
        for (std::size_t column = 0; column < inputs; ++column) {
            sum += weight[base + column] * in[column];
        }
        out[row] = activate ? std::tanh(sum) : sum;
    }
}

/// `out = W x` for a projection carrying no bias.
void project(const std::vector<float>& weight, std::span<const float> in, std::span<float> out) {
    const std::size_t inputs = in.size();
    for (std::size_t row = 0; row < out.size(); ++row) {
        float sum = 0.0F;
        const std::size_t base = row * inputs;
        for (std::size_t column = 0; column < inputs; ++column) {
            sum += weight[base + column] * in[column];
        }
        out[row] = sum;
    }
}

/// Softmax in place, maximum subtracted first. Torch does the same, and on
/// scores this small the difference would not show — but "the same arithmetic in
/// the same order" is the only claim parity can rest on.
void softmax(std::span<float> values) {
    float largest = values[0];
    for (const float value : values) {
        largest = std::max(largest, value);
    }
    float total = 0.0F;
    for (float& value : values) {
        value = std::exp(value - largest);
        total += value;
    }
    for (float& value : values) {
        value /= total;
    }
}

} // namespace

void Policy::entity_logits(std::span<const float> observation, std::span<float> logits_out,
                           float& value_out) const {
    Scratch& work = scratch();
    const std::size_t w = width_;
    const std::size_t tf = ObsSpec::threat_features;
    const std::size_t inf = ObsSpec::interceptor_features;
    const std::size_t bf = ObsSpec::blast_features;

    work.encoded_threats.resize(threats_ * w);
    work.encoded_interceptors.resize(interceptors_ * w);
    work.encoded_blasts.resize(blasts_ * w);
    work.slot.resize(w);
    work.context_input.resize(globals_width_ + (2 * w));
    work.summary.resize(hidden_);
    work.hidden.resize(hidden_);
    work.episode.resize(w);
    work.query.resize(w);
    work.scores.resize(std::max(interceptors_, blasts_));
    work.attended.resize(w);
    work.relation_input.resize(4 * w);
    work.per_threat.resize(w);
    work.fire.resize(batteries_);

    // Encode every slot through its own two-layer tanh encoder. Padding slots are
    // encoded too, exactly as the batched training pass does; what excludes them
    // is the presence mask at each place they could contribute.
    const auto encode_all = [&](const Layer& first, const Layer& second, std::size_t slots,
                                std::size_t features, std::size_t offset,
                                std::vector<float>& into) {
        for (std::size_t i = 0; i < slots; ++i) {
            const std::span<const float> raw =
                observation.subspan(offset + (i * features), features);
            forward(first.weight, first.bias, raw, work.slot, true);
            forward(second.weight, second.bias, work.slot, std::span<float>{into}.subspan(i * w, w),
                    true);
        }
    };
    const std::size_t interceptors_at = threats_ * tf;
    const std::size_t blasts_at = interceptors_at + (interceptors_ * inf);
    const std::size_t globals_at = blasts_at + (blasts_ * bf);
    encode_all(threat_encoder0_, threat_encoder1_, threats_, tf, 0, work.encoded_threats);
    encode_all(interceptor_encoder0_, interceptor_encoder1_, interceptors_, inf, interceptors_at,
               work.encoded_interceptors);
    encode_all(blast_encoder0_, blast_encoder1_, blasts_, bf, blasts_at, work.encoded_blasts);

    // The leading feature of every slot is its presence flag.
    const auto present = [&](std::size_t offset, std::size_t features, std::size_t i) {
        return observation[offset + (i * features)] != 0.0F;
    };

    // Permutation-invariant mean over the live entities; exactly zero when none.
    const auto pooled = [&](const std::vector<float>& encoded, std::size_t slots,
                            std::size_t offset, std::size_t features, std::span<float> out) {
        std::ranges::fill(out, 0.0F);
        std::size_t live = 0;
        for (std::size_t i = 0; i < slots; ++i) {
            if (!present(offset, features, i)) {
                continue;
            }
            ++live;
            for (std::size_t j = 0; j < w; ++j) {
                out[j] += encoded[(i * w) + j];
            }
        }
        const float count = static_cast<float>(std::max<std::size_t>(live, 1));
        for (float& value : out) {
            value /= count;
        }
    };

    std::ranges::copy(observation.subspan(globals_at, globals_width_), work.context_input.begin());
    pooled(work.encoded_interceptors, interceptors_, interceptors_at, inf,
           std::span<float>{work.context_input}.subspan(globals_width_, w));
    pooled(work.encoded_blasts, blasts_, blasts_at, bf,
           std::span<float>{work.context_input}.subspan(globals_width_ + w, w));

    forward(actor_context0_.weight, actor_context0_.bias, work.context_input, work.hidden, true);
    forward(actor_context1_.weight, actor_context1_.bias, work.hidden, work.summary, true);
    forward(context_to_threat_.weight, context_to_threat_.bias, work.summary, work.episode, false);

    /// One attention block for one threat. An empty entity set yields exactly
    /// zero *including the output bias* — `missile_defense.ppo._CrossAttention` multiplies by
    /// `has_entity`, which suppresses the bias too, and returning `output.bias`
    /// here would disagree with training on every observation with no live blast.
    const auto attend = [&](const Attention& block, std::span<const float> queried,
                            const std::vector<float>& encoded, std::size_t slots,
                            std::size_t offset, std::size_t features, std::span<float> out) {
        std::size_t live = 0;
        for (std::size_t i = 0; i < slots; ++i) {
            live += present(offset, features, i) ? 1u : 0u;
        }
        if (live == 0) {
            std::ranges::fill(out, 0.0F);
            return;
        }
        project(block.query, queried, work.query);
        const float scale = 1.0F / std::sqrt(static_cast<float>(w));
        std::size_t written = 0;
        for (std::size_t i = 0; i < slots; ++i) {
            if (!present(offset, features, i)) {
                continue;
            }
            float score = 0.0F;
            for (std::size_t row = 0; row < w; ++row) {
                float key = 0.0F;
                const std::size_t base = row * w;
                for (std::size_t column = 0; column < w; ++column) {
                    key += block.key[base + column] * encoded[(i * w) + column];
                }
                score += work.query[row] * key;
            }
            work.scores[written++] = score * scale;
        }
        softmax(std::span<float>{work.scores}.first(live));

        std::ranges::fill(work.attended, 0.0F);
        written = 0;
        for (std::size_t i = 0; i < slots; ++i) {
            if (!present(offset, features, i)) {
                continue;
            }
            const float weight = work.scores[written++];
            for (std::size_t row = 0; row < w; ++row) {
                float value = 0.0F;
                const std::size_t base = row * w;
                for (std::size_t column = 0; column < w; ++column) {
                    value += block.value[base + column] * encoded[(i * w) + column];
                }
                work.attended[row] += weight * value;
            }
        }
        forward(block.output.weight, block.output.bias, work.attended, out, false);
    };

    forward(noop_head_.weight, noop_head_.bias, work.summary, logits_out.first(1), false);

    for (std::size_t slot = 0; slot < threats_; ++slot) {
        if (present(0, tf, slot)) {
            const std::span<const float> encoded_threat =
                std::span<const float>{work.encoded_threats}.subspan(slot * w, w);
            std::ranges::copy(encoded_threat, work.relation_input.begin());
            attend(interceptor_attention_, encoded_threat, work.encoded_interceptors, interceptors_,
                   interceptors_at, inf, std::span<float>{work.relation_input}.subspan(w, w));
            attend(blast_attention_, encoded_threat, work.encoded_blasts, blasts_, blasts_at, bf,
                   std::span<float>{work.relation_input}.subspan(2 * w, w));
            std::ranges::copy(work.episode,
                              work.relation_input.begin() + static_cast<std::ptrdiff_t>(3 * w));
            forward(relation0_.weight, relation0_.bias, work.relation_input, work.slot, true);
            forward(relation1_.weight, relation1_.bias, work.slot, work.per_threat, true);
        } else {
            // `per_threat` is multiplied by the presence flag in training, so an
            // absent slot contributes an exact zero and its fire logits are the
            // head's bias alone. Those actions are masked anyway; matching them
            // is what keeps the parity fixture comparing whole logit vectors.
            std::ranges::fill(work.per_threat, 0.0F);
        }
        forward(fire_head_.weight, fire_head_.bias, work.per_threat, work.fire, false);
        for (std::size_t battery = 0; battery < batteries_; ++battery) {
            logits_out[1 + (battery * threats_) + slot] = work.fire[battery];
        }
    }

    forward(critic_trunk0_.weight, critic_trunk0_.bias, observation, work.hidden, true);
    forward(critic_trunk1_.weight, critic_trunk1_.bias, work.hidden, work.summary, true);
    std::array<float, 1> value{};
    forward(value_head_.weight, value_head_.bias, work.summary, value, false);
    value_out = value[0];
}

Policy::Decision Policy::act(std::span<const float> observation,
                             std::span<const std::uint8_t> legal) const {
    return act(observation, legal, {});
}

Policy::Decision Policy::act(std::span<const float> observation,
                             std::span<const std::uint8_t> legal,
                             std::span<float> logits_out) const {
    if (observation.size() != observation_size_) {
        throw Error{"observation has " + std::to_string(observation.size()) +
                    " values, this policy expects " + std::to_string(observation_size_)};
    }
    if (legal.size() != action_count_) {
        throw Error{"legal mask has " + std::to_string(legal.size()) +
                    " entries, this policy expects " + std::to_string(action_count_)};
    }
    if (!logits_out.empty() && logits_out.size() != action_count_) {
        throw Error{"logits buffer has " + std::to_string(logits_out.size()) +
                    " entries, this policy expects " + std::to_string(action_count_)};
    }

    float value = 0.0F;
    if (architecture_ == "entity") {
        // `entity_logits` owns the scratch it needs, including `logits`, so the
        // buffer is sized before the call rather than inside it.
        scratch().logits.resize(action_count_);
        entity_logits(observation, scratch().logits, value);
    } else {
        Scratch& flat = scratch();
        flat.hidden.resize(hidden_);
        flat.features.resize(hidden_);
        flat.logits.resize(action_count_);

        forward(trunk0_.weight, trunk0_.bias, observation, flat.hidden, true);
        forward(trunk1_.weight, trunk1_.bias, flat.hidden, flat.features, true);
        forward(policy_head_.weight, policy_head_.bias, flat.features, flat.logits, false);

        std::array<float, 1> estimate{};
        forward(value_head_.weight, value_head_.bias, flat.features, estimate, false);
        value = estimate[0];
    }
    Scratch& work = scratch();

    Decision decision;
    decision.value = value;
    // Masking, then the *first* maximum. Both halves are promises rather than
    // conveniences: `missile_defense.export_policy` does exactly this, and the parity fixture
    // would be a coin flip on ties if either side chose differently.
    float best = 0.0F;
    bool started = false;
    for (std::size_t i = 0; i < work.logits.size(); ++i) {
        const float logit = legal[i] != 0u ? work.logits[i] : masked_logit;
        work.logits[i] = logit;
        if (!started || logit > best) {
            best = logit;
            decision.action = static_cast<std::uint32_t>(i);
            started = true;
        }
    }
    if (!logits_out.empty()) {
        std::ranges::copy(work.logits, logits_out.begin());
    }
    return decision;
}

} // namespace md::agent
