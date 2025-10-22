#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <vector>
#include <algorithm>
#include <cstring>
#include <stdexcept>

class Window;

class Engine {
public:
    Engine() = default;
    ~Engine() = default;

    Engine(Window& window_ref);
    void Cleanup();

    void Tick();

private:
    Window* window = nullptr;
    bool frame_buffer_resized = false;

#ifdef NDEBUG
    static constexpr bool enable_validation_layers = false;
#else
    static constexpr bool enable_validation_layers = true;
#endif

    vk::raii::Context        context;
    vk::raii::Instance       instance        = nullptr;
    vk::raii::SurfaceKHR      surface         = nullptr;
    vk::raii::PhysicalDevice physical_device = nullptr;
    vk::raii::Device         device          = nullptr;
    vk::raii::Queue          queue           = nullptr;

    // swapchain
    vk::raii::SwapchainKHR           swapchain = nullptr;
    std::vector<vk::Image>           swapchain_images;
    vk::SurfaceFormatKHR             swapchain_surface_format;
    vk::Extent2D                     swapchain_extent;
    std::vector<vk::raii::ImageView> swapchain_image_views;

    // pipeline
    vk::raii::DescriptorSetLayout descriptor_set_layout = nullptr;
    vk::raii::PipelineLayout      pipeline_layout       = nullptr;
    vk::raii::Pipeline            graphics_pipeline     = nullptr;

    // layers and extensions
    const std::vector<const char*> validation_layers = { "VK_LAYER_KHRONOS_validation" };
    const std::vector<const char*> required_device_extensions = {
        vk::KHRSwapchainExtensionName,
        vk::KHRSpirv14ExtensionName,
        vk::KHRSynchronization2ExtensionName,
        vk::KHRCreateRenderpass2ExtensionName
    };


    // Initialisation
    void CreateInstance();

    // helper methods
    std::vector<const char*> GetRequiredExtensions();
};
