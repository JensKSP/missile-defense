#include "renderer.hpp"

#include "game_window.hpp"
#include "projection.hpp"

#include <QVulkanDeviceFunctions>
#include <QVulkanInstance>

// Embedded SPIR-V (generated at build time by glslangValidator --vn).
#include "quad_frag_spv.h"
#include "quad_vert_spv.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace md {

namespace {

// Per-instance vertex data: an axis-aligned box (world units), a colour, and a
// shape flag (0 = rectangle, 1 = circle).
struct InstanceData {
    float cx, cy;
    float hx, hy;
    float r, g, b;
    float shape;
};

// Push constants: world -> clip transform, clip.xy = worldPos * a + b.
struct PushConstants {
    float a[2];
    float b[2];
};

constexpr std::size_t max_instances = 1024;

InstanceData rect(float cx, float cy, float hx, float hy, float r, float g, float b) {
    return InstanceData{cx, cy, hx, hy, r, g, b, 0.0f};
}

InstanceData circle(float cx, float cy, float radius, float r, float g, float b) {
    return InstanceData{cx, cy, radius, radius, r, g, b, 1.0f};
}

// A 3x5 pixel font for digits, drawn as small quads (no font textures). Each
// row's low 3 bits are the lit pixels; most-significant bit = leftmost column.
constexpr std::array<std::array<std::uint8_t, 5>, 10> digit_font = {{
    {{7, 5, 5, 5, 7}},
    {{2, 6, 2, 2, 7}},
    {{7, 1, 7, 4, 7}},
    {{7, 1, 7, 1, 7}},
    {{5, 5, 7, 1, 1}},
    {{7, 4, 7, 1, 7}},
    {{7, 4, 7, 5, 7}},
    {{7, 1, 2, 2, 2}},
    {{7, 5, 7, 5, 7}},
    {{7, 5, 7, 1, 7}},
}};

void draw_glyph(std::vector<InstanceData>& inst, const std::array<std::uint8_t, 5>& rows,
                float left_x, float top_y, float px, float r, float g, float b) {
    for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < 3; ++col) {
            if ((rows[static_cast<std::size_t>(row)] & (1u << (2 - col))) != 0u) {
                inst.push_back(rect(left_x + ((static_cast<float>(col) + 0.5f) * px),
                                    top_y - ((static_cast<float>(row) + 0.5f) * px), px * 0.42f,
                                    px * 0.42f, r, g, b));
            }
        }
    }
}

// Draw an unsigned integer as pixel-font digits. right_align anchors to x's right.
void draw_number(std::vector<InstanceData>& inst, std::uint32_t value, float x, float top_y,
                 float px, float r, float g, float b, bool right_align) {
    std::array<int, 12> digits{};
    int count = 0;
    if (value == 0) {
        digits[static_cast<std::size_t>(count++)] = 0;
    } else {
        std::uint32_t v = value;
        while (v > 0 && count < 12) {
            digits[static_cast<std::size_t>(count++)] = static_cast<int>(v % 10);
            v /= 10;
        }
    }
    const float advance = px * 4.0f; // 3 columns + 1 gap
    const float start_x = right_align ? (x - (static_cast<float>(count) * advance)) : x;
    for (int i = 0; i < count; ++i) {
        const auto d = static_cast<std::size_t>(digits[static_cast<std::size_t>(count - 1 - i)]);
        draw_glyph(inst, digit_font[d], start_x + (static_cast<float>(i) * advance), top_y, px, r,
                   g, b);
    }
}

// 3x5 uppercase letters A-Z (same encoding as digit_font). Cramped but readable.
constexpr std::array<std::array<std::uint8_t, 5>, 26> letter_font = {{
    {{2, 5, 7, 5, 5}}, {{6, 5, 6, 5, 6}}, {{3, 4, 4, 4, 3}}, {{6, 5, 5, 5, 6}}, {{7, 4, 6, 4, 7}},
    {{7, 4, 6, 4, 4}}, {{3, 4, 5, 5, 3}}, {{5, 5, 7, 5, 5}}, {{7, 2, 2, 2, 7}}, {{1, 1, 1, 5, 2}},
    {{5, 6, 4, 6, 5}}, {{4, 4, 4, 4, 7}}, {{5, 7, 7, 5, 5}}, {{5, 7, 7, 7, 5}}, {{7, 5, 5, 5, 7}},
    {{6, 5, 6, 4, 4}}, {{7, 5, 5, 7, 3}}, {{6, 5, 6, 5, 5}}, {{3, 4, 2, 1, 6}}, {{7, 2, 2, 2, 2}},
    {{5, 5, 5, 5, 7}}, {{5, 5, 5, 5, 2}}, {{5, 5, 7, 7, 5}}, {{5, 5, 2, 5, 5}}, {{5, 5, 2, 2, 2}},
    {{7, 1, 2, 4, 7}},
}};

std::array<std::uint8_t, 5> glyph_for(char c) {
    if (c >= '0' && c <= '9') {
        return digit_font[static_cast<std::size_t>(c - '0')];
    }
    if (c >= 'A' && c <= 'Z') {
        return letter_font[static_cast<std::size_t>(c - 'A')];
    }
    return {{0, 0, 0, 0, 0}};
}

// Draw a string with the pixel font. `centered` treats x as the horizontal centre.
void draw_text(std::vector<InstanceData>& inst, std::string_view text, float x, float top_y,
               float px, float r, float g, float b, bool centered) {
    const float advance = px * 4.0f;
    float cursor = centered ? (x - (static_cast<float>(text.size()) * advance * 0.5f)) : x;
    for (const char c : text) {
        if (c != ' ') {
            draw_glyph(inst, glyph_for(c), cursor, top_y, px, r, g, b);
        }
        cursor += advance;
    }
}

VkShaderModule make_shader(QVulkanDeviceFunctions* dev, VkDevice device, const uint32_t* code,
                           std::size_t bytes) {
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = bytes;
    info.pCode = code;
    VkShaderModule module = VK_NULL_HANDLE;
    dev->vkCreateShaderModule(device, &info, nullptr, &module);
    return module;
}

void alloc_buffer(QVulkanWindow* win, QVulkanDeviceFunctions* dev, VkDeviceSize size,
                  VkBuffer* buffer, VkDeviceMemory* memory) {
    VkDevice device = win->device();

    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    dev->vkCreateBuffer(device, &bi, nullptr, buffer);

    VkMemoryRequirements req{};
    dev->vkGetBufferMemoryRequirements(device, *buffer, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = win->hostVisibleMemoryIndex();
    dev->vkAllocateMemory(device, &ai, nullptr, memory);
    dev->vkBindBufferMemory(device, *buffer, *memory, 0);
}

} // namespace

Renderer::Renderer(GameWindow* window) noexcept : window_{window} {}

void Renderer::initResources() {
    dev_ = window_->vulkanInstance()->deviceFunctions(window_->device());
    createBuffers();
    createPipeline();
}

void Renderer::createBuffers() {
    // Unit quad (two triangles), corners in [-0.5, 0.5].
    static const std::array<float, 12> quad = {-0.5f, -0.5f, 0.5f, -0.5f, 0.5f,  0.5f,
                                               -0.5f, -0.5f, 0.5f, 0.5f,  -0.5f, 0.5f};
    alloc_buffer(window_, dev_, sizeof(quad), &quad_buf_, &quad_mem_);
    void* mapped = nullptr;
    dev_->vkMapMemory(window_->device(), quad_mem_, 0, sizeof(quad), 0, &mapped);
    std::memcpy(mapped, quad.data(), sizeof(quad));
    dev_->vkUnmapMemory(window_->device(), quad_mem_);

    // One persistently-mapped instance buffer per frame in flight.
    const int frames = window_->concurrentFrameCount();
    const VkDeviceSize inst_bytes = max_instances * sizeof(InstanceData);
    instance_bufs_.resize(static_cast<std::size_t>(frames));
    instance_mems_.resize(static_cast<std::size_t>(frames));
    instance_mapped_.resize(static_cast<std::size_t>(frames));
    for (int i = 0; i < frames; ++i) {
        const auto f = static_cast<std::size_t>(i);
        alloc_buffer(window_, dev_, inst_bytes, &instance_bufs_[f], &instance_mems_[f]);
        dev_->vkMapMemory(window_->device(), instance_mems_[f], 0, inst_bytes, 0,
                          &instance_mapped_[f]);
    }
}

void Renderer::createPipeline() {
    VkDevice device = window_->device();

    VkShaderModule vert = make_shader(dev_, device, quad_vert_spv, sizeof(quad_vert_spv));
    VkShaderModule frag = make_shader(dev_, device, quad_frag_spv, sizeof(quad_frag_spv));

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push.offset = 0;
    push.size = static_cast<std::uint32_t>(sizeof(PushConstants));

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    dev_->vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout_);

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    std::array<VkVertexInputBindingDescription, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].stride = static_cast<std::uint32_t>(sizeof(float) * 2);
    bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindings[1].binding = 1;
    bindings[1].stride = static_cast<std::uint32_t>(sizeof(InstanceData));
    bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    std::array<VkVertexInputAttributeDescription, 5> attrs{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
    attrs[1] = {1, 1, VK_FORMAT_R32G32_SFLOAT,
                static_cast<std::uint32_t>(offsetof(InstanceData, cx))};
    attrs[2] = {2, 1, VK_FORMAT_R32G32_SFLOAT,
                static_cast<std::uint32_t>(offsetof(InstanceData, hx))};
    attrs[3] = {3, 1, VK_FORMAT_R32G32B32_SFLOAT,
                static_cast<std::uint32_t>(offsetof(InstanceData, r))};
    attrs[4] = {4, 1, VK_FORMAT_R32_SFLOAT,
                static_cast<std::uint32_t>(offsetof(InstanceData, shape))};

    VkPipelineVertexInputStateCreateInfo vin{};
    vin.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vin.vertexBindingDescriptionCount = static_cast<std::uint32_t>(bindings.size());
    vin.pVertexBindingDescriptions = bindings.data();
    vin.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attrs.size());
    vin.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = window_->sampleCountFlagBits();

    VkPipelineDepthStencilStateCreateInfo dss{};
    dss.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    dss.depthTestEnable = VK_FALSE;
    dss.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable = VK_FALSE;
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &blend;

    std::array<VkDynamicState, 2> dynamics{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = static_cast<std::uint32_t>(dynamics.size());
    dyn.pDynamicStates = dynamics.data();

    VkGraphicsPipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount = static_cast<std::uint32_t>(stages.size());
    info.pStages = stages.data();
    info.pVertexInputState = &vin;
    info.pInputAssemblyState = &ia;
    info.pViewportState = &vp;
    info.pRasterizationState = &rs;
    info.pMultisampleState = &ms;
    info.pDepthStencilState = &dss;
    info.pColorBlendState = &cb;
    info.pDynamicState = &dyn;
    info.layout = pipeline_layout_;
    info.renderPass = window_->defaultRenderPass();
    info.subpass = 0;
    dev_->vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline_);

    dev_->vkDestroyShaderModule(device, vert, nullptr);
    dev_->vkDestroyShaderModule(device, frag, nullptr);
}

void Renderer::startNextFrame() {
    window_->advance();

    const QSize sz = window_->swapChainImageSize();
    const Sim& sim = window_->sim();
    const float world_w = sim.config().world_width;
    const float world_h = sim.config().world_height;

    const auto state = window_->state();
    const bool in_game = state == GameWindow::State::Playing ||
                         state == GameWindow::State::Paused || state == GameWindow::State::GameOver;
    const float cx = world_w * 0.5f;

    std::vector<InstanceData> inst;
    inst.reserve(max_instances);

    // Field backdrop (drawn in every state).
    inst.push_back(rect(world_w * 0.5f, 1.0f, world_w * 0.5f, 1.0f, 0.10f, 0.11f, 0.18f)); // ground
    for (const auto& city : sim.cities()) {
        if (city.alive) {
            inst.push_back(rect(city.pos.x, 4.0f, 7.0f, 4.0f, 0.25f, 0.75f, 0.95f));
        } else {
            inst.push_back(rect(city.pos.x, 1.5f, 6.0f, 1.5f, 0.22f, 0.20f, 0.24f)); // rubble
        }
    }
    for (const auto& base : sim.bases()) {
        const bool empty = base.ammo == 0;
        inst.push_back(rect(base.pos.x, 6.0f, 6.0f, 6.0f, empty ? 0.40f : 0.95f,
                            empty ? 0.35f : 0.65f, empty ? 0.15f : 0.20f));
    }

    if (in_game) {
        for (const auto& threat : sim.threats()) {
            inst.push_back(circle(threat.pos.x, threat.pos.y, 1.6f, 0.95f, 0.30f, 0.25f));
        }
        for (const auto& it : sim.interceptors()) {
            inst.push_back(circle(it.pos.x, it.pos.y, 1.0f, 0.85f, 0.95f, 1.0f));
        }
        for (const auto& blast : sim.blasts()) {
            inst.push_back(circle(blast.center.x, blast.center.y, blast.radius, 1.0f, 0.6f, 0.15f));
        }
        const float digit_px = world_h * 0.013f;
        const float hud_top = world_h * 0.97f;
        draw_number(inst, static_cast<std::uint32_t>(sim.score() < 0 ? 0 : sim.score()),
                    world_w * 0.02f, hud_top, digit_px, 1.0f, 1.0f, 1.0f, false);
        draw_number(inst, sim.wave(), world_w * 0.98f, hud_top, digit_px, 0.95f, 0.75f, 0.30f,
                    true);
        for (const auto& base : sim.bases()) {
            const float spacing = 1.6f;
            const float base_x0 =
                base.pos.x - ((static_cast<float>(base.ammo) - 1.0f) * spacing * 0.5f);
            for (std::uint32_t k = 0; k < base.ammo; ++k) {
                inst.push_back(rect(base_x0 + (static_cast<float>(k) * spacing), 14.0f, 0.55f,
                                    0.55f, 0.40f, 0.90f, 0.55f));
            }
        }
        if (state == GameWindow::State::Playing) {
            const Vec2 aim = window_->aim();
            inst.push_back(circle(aim.x, aim.y, 2.2f, 1.0f, 1.0f, 1.0f)); // crosshair
        }
    }

    if (state == GameWindow::State::Menu) {
        draw_text(inst, "MISSILE DEFENSE", cx, world_h * 0.86f, world_h * 0.023f, 0.85f, 0.92f,
                  1.0f, true);
        const std::array<std::string_view, 3> items{"START", "HIGHSCORES", "EXIT"};
        for (int i = 0; i < 3; ++i) {
            const bool sel = window_->menu_index() == i;
            const float y = world_h * (0.56f - (static_cast<float>(i) * 0.13f));
            draw_text(inst, items[static_cast<std::size_t>(i)], cx, y, world_h * 0.018f,
                      sel ? 0.95f : 0.45f, sel ? 0.75f : 0.45f, sel ? 0.25f : 0.50f, true);
        }
        draw_text(inst, "ARROWS ENTER", cx, world_h * 0.10f, world_h * 0.011f, 0.4f, 0.45f, 0.5f,
                  true);
    } else if (state == GameWindow::State::GameOver) {
        draw_text(inst, "GAME OVER", cx, world_h * 0.64f, world_h * 0.036f, 0.95f, 0.30f, 0.25f,
                  true);
        draw_text(inst, "SCORE", cx, world_h * 0.46f, world_h * 0.016f, 0.8f, 0.85f, 0.9f, true);
        draw_text(inst, std::to_string(sim.score() < 0 ? 0 : sim.score()), cx, world_h * 0.40f,
                  world_h * 0.022f, 1.0f, 1.0f, 1.0f, true);
        draw_text(inst, "PRESS ENTER", cx, world_h * 0.22f, world_h * 0.013f, 0.6f, 0.65f, 0.7f,
                  true);
    } else if (state == GameWindow::State::Paused) {
        draw_text(inst, "PAUSED", cx, world_h * 0.56f, world_h * 0.032f, 0.95f, 0.9f, 0.4f, true);
    } else if (state == GameWindow::State::Highscores) {
        draw_text(inst, "HIGHSCORES", cx, world_h * 0.80f, world_h * 0.030f, 0.85f, 0.92f, 1.0f,
                  true);
        draw_text(inst, "COMING SOON", cx, world_h * 0.50f, world_h * 0.016f, 0.6f, 0.65f, 0.7f,
                  true);
        draw_text(inst, "PRESS ENTER", cx, world_h * 0.28f, world_h * 0.013f, 0.6f, 0.65f, 0.7f,
                  true);
    }

    if (inst.size() > max_instances) {
        inst.resize(max_instances);
    }

    const auto frame = static_cast<std::size_t>(window_->currentFrame());
    std::memcpy(instance_mapped_[frame], inst.data(), inst.size() * sizeof(InstanceData));
    const auto instance_count = static_cast<std::uint32_t>(inst.size());

    // World -> clip transform (matches the mouse->world mapping in GameWindow).
    const Projection proj = Projection::make(world_w, world_h, static_cast<float>(sz.width()),
                                             static_cast<float>(sz.height()));
    PushConstants pc{};
    pc.a[0] = proj.ax;
    pc.a[1] = proj.ay;
    pc.b[0] = proj.bx;
    pc.b[1] = proj.by;

    std::array<VkClearValue, 3> clear{};
    clear[0].color = {{0.05f, 0.06f, 0.12f, 1.0f}}; // night sky
    clear[1].depthStencil = {1.0f, 0};
    clear[2].color = clear[0].color;
    const bool msaa = window_->sampleCountFlagBits() > VK_SAMPLE_COUNT_1_BIT;

    VkRenderPassBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = window_->defaultRenderPass();
    begin.framebuffer = window_->currentFramebuffer();
    begin.renderArea.extent.width = static_cast<std::uint32_t>(sz.width());
    begin.renderArea.extent.height = static_cast<std::uint32_t>(sz.height());
    begin.clearValueCount = msaa ? 3u : 2u;
    begin.pClearValues = clear.data();

    const VkCommandBuffer cb = window_->currentCommandBuffer();
    dev_->vkCmdBeginRenderPass(cb, &begin, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.width = static_cast<float>(sz.width());
    viewport.height = static_cast<float>(sz.height());
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    dev_->vkCmdSetViewport(cb, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent.width = static_cast<std::uint32_t>(sz.width());
    scissor.extent.height = static_cast<std::uint32_t>(sz.height());
    dev_->vkCmdSetScissor(cb, 0, 1, &scissor);

    dev_->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    dev_->vkCmdPushConstants(cb, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                             static_cast<std::uint32_t>(sizeof(PushConstants)), &pc);

    std::array<VkBuffer, 2> vbufs{quad_buf_, instance_bufs_[frame]};
    std::array<VkDeviceSize, 2> offsets{0, 0};
    dev_->vkCmdBindVertexBuffers(cb, 0, static_cast<std::uint32_t>(vbufs.size()), vbufs.data(),
                                 offsets.data());
    dev_->vkCmdDraw(cb, 6, instance_count, 0, 0);

    dev_->vkCmdEndRenderPass(cb);
    window_->frameReady();
    window_->requestUpdate();
}

void Renderer::releaseResources() {
    VkDevice device = window_->device();
    for (std::size_t i = 0; i < instance_bufs_.size(); ++i) {
        if (instance_mems_[i] != VK_NULL_HANDLE) {
            dev_->vkUnmapMemory(device, instance_mems_[i]);
            dev_->vkFreeMemory(device, instance_mems_[i], nullptr);
        }
        if (instance_bufs_[i] != VK_NULL_HANDLE) {
            dev_->vkDestroyBuffer(device, instance_bufs_[i], nullptr);
        }
    }
    instance_bufs_.clear();
    instance_mems_.clear();
    instance_mapped_.clear();

    if (pipeline_ != VK_NULL_HANDLE) {
        dev_->vkDestroyPipeline(device, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        dev_->vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (quad_buf_ != VK_NULL_HANDLE) {
        dev_->vkDestroyBuffer(device, quad_buf_, nullptr);
        quad_buf_ = VK_NULL_HANDLE;
    }
    if (quad_mem_ != VK_NULL_HANDLE) {
        dev_->vkFreeMemory(device, quad_mem_, nullptr);
        quad_mem_ = VK_NULL_HANDLE;
    }
}

} // namespace md
