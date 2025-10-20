#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <limits>
#include <assert.h>

#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vk_platform.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<const char*> validation_layers = { "VK_LAYER_KHRONOS_validation" };

#ifdef NDEBUG
    const bool enable_validation_layers = false;
#else
    const bool enable_validation_layers = true;
#endif

class App {
public:
    void Run() {
        InitWindow();
        InitVulkan();
        MainLoop();
        Cleanup();
    }

private:
    GLFWwindow *                     window = nullptr;
    vk::raii::Context                context;
    vk::raii::Instance               instance        = nullptr;
    vk::raii::DebugUtilsMessengerEXT debug_messenger = nullptr;
    vk::raii::SurfaceKHR             surface         = nullptr;
    vk::raii::PhysicalDevice         physical_device = nullptr;
    vk::raii::Device                 device          = nullptr;
    uint32_t                         queue_index     = ~0;
    vk::raii::Queue                  queue           = nullptr;
    vk::raii::SwapchainKHR           swap_chain      = nullptr;
    std::vector<vk::Image>	     swap_chain_images;
    vk::SurfaceFormatKHR	     swap_chain_surface_format;
    vk::Extent2D		     swap_chain_extent;
    std::vector<vk::raii::ImageView> swap_chain_image_views;

    vk::raii::PipelineLayout pipeline_layout   = nullptr;
    vk::raii::Pipeline       graphics_pipeline = nullptr;
    vk::raii::CommandPool    command_pool      = nullptr;
    vk::raii::CommandBuffer command_buffer    = nullptr;

    std::vector<const char*> required_device_extensions = {
        vk::KHRSwapchainExtensionName,
        vk::KHRSpirv14ExtensionName,
        vk::KHRSynchronization2ExtensionName,
        vk::KHRCreateRenderpass2ExtensionName
    };

    void InitWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Engine", nullptr, nullptr);
    }

    void InitVulkan() {
        CreateInstance();
        SetupDebugMessenger();
        CreateSurface();
        PickPhysicalDevice();
        CreateLogicalDevice();
        CreateSwapChain();
        CreateImageViews();
        CreateGraphicsPipeline();
        CreateCommandPool();
        CreateCommandBuffer();
    }

    void MainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void Cleanup() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void CreateInstance() {
        constexpr vk::ApplicationInfo app_info{
            .applicationVersion = VK_MAKE_VERSION(1, 4, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 4, 0),
            .apiVersion = vk::ApiVersion14,
        };
        
        // Get required layers
        std::vector<const char*> required_layers;
        if (enable_validation_layers) {
            required_layers.assign(validation_layers.begin(), validation_layers.end());
        }

        // Check if required layers are supported by Vulkan implementation
        auto layer_properties = context.enumerateInstanceLayerProperties();
        for (auto const& required_layer : required_layers) {
            if (std::ranges::none_of(layer_properties,
                        [required_layer](auto const& layer_property)
                        { return strcmp(layer_property.layerName, required_layer) == 0; }))
            {
                throw std::runtime_error("Required layer not supported: " + std::string(required_layer));
            }
        }

        // Get required extensions
        auto required_extensions = GetRequiredExtensions();

        // Check if required extensions are supported by Vulkan implementation
        auto extension_properties = context.enumerateInstanceExtensionProperties();
        for (auto const & required_extension : required_extensions) {
            if (std::ranges::none_of(extension_properties,
                        [required_extension](auto const& extension_property)
                        { return strcmp(extension_property.extensionName, required_extension) == 0; }))
            {
                throw std::runtime_error("Required extension not supported: " + std::string(required_extension));
            }
        }

        vk::InstanceCreateInfo create_info{
            .pApplicationInfo = &app_info,
            .enabledLayerCount = static_cast<uint32_t>(required_layers.size()),
            .ppEnabledLayerNames = required_layers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(required_extensions.size()),
            .ppEnabledExtensionNames = required_extensions.data(),
        };

        instance = vk::raii::Instance(context, create_info);
    }

    void SetupDebugMessenger() {
        if (!enable_validation_layers) return;

        vk::DebugUtilsMessageSeverityFlagsEXT severity_flags(
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        vk::DebugUtilsMessageTypeFlagsEXT message_type_flags(
                vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

        vk::DebugUtilsMessengerCreateInfoEXT debug_utils_messenger_create_info_EXT{};
        debug_utils_messenger_create_info_EXT.messageSeverity = severity_flags;
        debug_utils_messenger_create_info_EXT.messageType = message_type_flags;
        debug_utils_messenger_create_info_EXT.pfnUserCallback = &DebugCallback;

        debug_messenger = instance.createDebugUtilsMessengerEXT(debug_utils_messenger_create_info_EXT);
    }

    void CreateSurface() {
        VkSurfaceKHR _surface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    void PickPhysicalDevice() {
        std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        const auto dev_iter = std::ranges::find_if(devices,
        [&](auto const & device) {
                auto queue_families = device.getQueueFamilyProperties();
                bool is_suitable = device.getProperties().apiVersion >= VK_API_VERSION_1_4;
                const auto qfp_iter = std::ranges::find_if(queue_families,
                []( vk::QueueFamilyProperties const & qfp )
                        {
                            return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
                        } );
                is_suitable = is_suitable && ( qfp_iter != queue_families.end() );
                auto extensions = device.enumerateDeviceExtensionProperties( );
                bool found = true;
                for (auto const & extension : required_device_extensions) {
                    auto extension_iter = std::ranges::find_if(extensions, [extension](auto const & ext) {return strcmp(ext.extensionName, extension) == 0;});
                    found = found &&  extension_iter != extensions.end();
                }
                is_suitable = is_suitable && found;
                if (is_suitable) {
                    physical_device = device;
                }
                return is_suitable;
        });
        if (dev_iter == devices.end()) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void CreateLogicalDevice() {
        std::vector<vk::QueueFamilyProperties> queue_family_properties = physical_device.getQueueFamilyProperties();

        // get first index into queue_family_properties which supports graphics and present
        for (uint32_t qfp_index = 0; qfp_index < queue_family_properties.size(); qfp_index++) {
            if ((queue_family_properties[qfp_index].queueFlags & vk::QueueFlagBits::eGraphics) &&
                    physical_device.getSurfaceSupportKHR(qfp_index, *surface)) {
                // found queue family that supports graphics and present
                queue_index = qfp_index;
                break;
            }
        }
        if (queue_index == ~0u) {
            throw std::runtime_error("Could not find a queue for grahpics and present -> terminating");
        }

        // query Vulkan 1.3 features
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> feature_chain = {
            vk::PhysicalDeviceFeatures2{},
            vk::PhysicalDeviceVulkan13Features{.dynamicRendering = true},
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT{.extendedDynamicState = true}
        };

        // create a Device
        float                     queue_priority = 0.0f;
        vk::DeviceQueueCreateInfo device_queue_create_info {.queueFamilyIndex = queue_index, .queueCount = 1, .pQueuePriorities = &queue_priority};
        vk::DeviceCreateInfo      device_create_info {.pNext = &feature_chain.get<vk::PhysicalDeviceFeatures2>(),
                                                      .queueCreateInfoCount = 1,
                                                      .pQueueCreateInfos = &device_queue_create_info,
                                                      .enabledLayerCount = 0,
                                                      .ppEnabledLayerNames = nullptr,
                                                      .enabledExtensionCount = static_cast<uint32_t>(required_device_extensions.size()),
                                                      .ppEnabledExtensionNames = required_device_extensions.data()
        };

        device = vk::raii::Device(physical_device, device_create_info);
        queue = vk::raii::Queue(device, queue_index, 0);
    }

    void CreateSwapChain() {
	    auto surface_capabilities = physical_device.getSurfaceCapabilitiesKHR(*surface);
	    swap_chain_extent	      = ChooseSwapExtent(surface_capabilities);
	    swap_chain_surface_format = ChooseSwapSurfaceFormat(physical_device.getSurfaceFormatsKHR(*surface));
	    vk::SwapchainCreateInfoKHR swap_chain_create_info { .surface          = *surface,
		    						.minImageCount    = ChooseSwapMinImageCount(surface_capabilities),
								.imageFormat      = swap_chain_surface_format.format,
								.imageColorSpace  = swap_chain_surface_format.colorSpace,
								.imageExtent      = swap_chain_extent,
								.imageArrayLayers = 1,
								.imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
								.imageSharingMode = vk::SharingMode::eExclusive,
								.preTransform     = surface_capabilities.currentTransform,
								.compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
								.presentMode      = ChooseSwapPresentMode(physical_device.getSurfacePresentModesKHR(*surface)),
								.clipped          = true };

	    swap_chain = vk::raii::SwapchainKHR(device, swap_chain_create_info);
	    swap_chain_images = swap_chain.getImages();
    }

    void CreateImageViews() {
	    assert(swap_chain_image_views.empty());

	    vk::ImageViewCreateInfo image_view_create_info { .viewType = vk::ImageViewType::e2D,
		    					     .format = swap_chain_surface_format.format,
							     .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 } };
	    for (auto image : swap_chain_images) {
		    image_view_create_info.image = image;
		    swap_chain_image_views.emplace_back(device, image_view_create_info);
	    }
    }

    void CreateGraphicsPipeline() {
        vk::raii::ShaderModule vert_shader_module = CreateShaderModule(read_file("shaders/shader.frag.spv"));
        vk::raii::ShaderModule frag_shader_module = CreateShaderModule(read_file("shaders/shader.vert.spv"));

        vk::PipelineShaderStageCreateInfo vert_shader_stage_info { .stage = vk::ShaderStageFlagBits::eVertex, .module = vert_shader_module, .pName = "main" };
        vk::PipelineShaderStageCreateInfo frag_shader_stage_info { .stage = vk::ShaderStageFlagBits::eFragment, .module = frag_shader_module, .pName = "main" };
        vk::PipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info, frag_shader_stage_info};

        vk::PipelineVertexInputStateCreateInfo vertex_input_info;
        vk::PipelineInputAssemblyStateCreateInfo input_assembly{ .topology = vk::PrimitiveTopology::eTriangleList };
        vk::PipelineViewportStateCreateInfo viewport_state{ .viewportCount = 1, .scissorCount = 1 };

        vk::PipelineRasterizationStateCreateInfo rasterizer{ .depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False,
                                                             .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eBack,
                                                             .frontFace = vk::FrontFace::eClockwise, .depthBiasEnable = vk::False,
                                                             .depthBiasSlopeFactor = 1.0f, .lineWidth = 1.0f };

        vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

        vk::PipelineColorBlendAttachmentState color_blend_attachment{ .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        vk::PipelineColorBlendStateCreateInfo color_blending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &color_blend_attachment };

        std::vector dynamic_states = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamic_state{ .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()), .pDynamicStates = dynamic_states.data() };

        vk::PipelineLayoutCreateInfo pipeline_layout_info;

        pipeline_layout = vk::raii::PipelineLayout(device, pipeline_layout_info);

        vk::PipelineRenderingCreateInfo pipeline_rendering_create_info{ .colorAttachmentCount = 1, .pColorAttachmentFormats = &swap_chain_surface_format.format };
        vk::GraphicsPipelineCreateInfo pipeline_info{ .pNext = &pipeline_rendering_create_info,
            .stageCount = 2, .pStages = shader_stages,
            .pVertexInputState = &vertex_input_info, .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_state, .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling, .pColorBlendState = &color_blending,
            .pDynamicState = &dynamic_state, .layout = pipeline_layout, .renderPass = nullptr };

        graphics_pipeline = vk::raii::Pipeline(device, nullptr, pipeline_info);
    }

    void CreateCommandPool() {
        vk::CommandPoolCreateInfo pool_info{ .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                             .queueFamilyIndex = queue_index };
        command_pool = vk::raii::CommandPool(device, pool_info);
    }

    void CreateCommandBuffer() {
        vk::CommandBufferAllocateInfo alloc_info{ .commandPool = command_pool, .level = vk::CommandBufferLevel::ePrimary,
                                                  .commandBufferCount = 1 };
        command_buffer = std::move(vk::raii::CommandBuffers(device, alloc_info).front());
    }

    void RecordCommandBuffer(uint32_t image_index) {
        command_buffer.begin( {} );
        // Before starting rendering, transition swapchain image to COLOR_ATTACHMENT_OPTIMAL
        transition_image_layout(
                image_index,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eColorAttachmentOptimal,
                {},
                vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::PipelineStageFlagBits2::eTopOfPipe,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput
        );
        vk::ClearValue clear_color = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
        vk::RenderingAttachmentInfo attachment_info = {
            .imageView = swap_chain_image_views[image_index],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clear_color
        };
        vk::RenderingInfo rendering_info = {
            .renderArea = { .offset = { 0, 0 }, .extent = swap_chain_extent },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachment_info
        };

        command_buffer.beginRendering(rendering_info);
        command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_pipeline);
        command_buffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swap_chain_extent.width), static_cast<float>(swap_chain_extent.height), 0.0f, 1.0f));
        command_buffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swap_chain_extent));
        command_buffer.draw(3, 1, 0, 0);
        command_buffer.endRendering();

        // After rendering, transition swapchain image to PRESENT_SRC
        transition_image_layout(
                image_index,
                vk::ImageLayout::eColorAttachmentOptimal,
                vk::ImageLayout::ePresentSrcKHR,
                vk::AccessFlagBits2::eColorAttachmentWrite,         // srcAccessMask
                {},                                                 // dstAccessMask
                vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
                vk::PipelineStageFlagBits2::eBottomOfPipe           // dstStage
        );
        command_buffer.end();
    }

    void transition_image_layout(
            uint32_t current_frame,
            vk::ImageLayout old_layout,
            vk::ImageLayout new_layout,
            vk::AccessFlags2 src_access_mask,
            vk::AccessFlags2 dst_access_mask,
            vk::PipelineStageFlags2 src_stage_mask,
            vk::PipelineStageFlags2 dst_stage_mask
            ) {
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask = src_stage_mask,
            .srcAccessMask = src_access_mask,
            .dstStageMask = dst_stage_mask,
            .dstAccessMask = dst_access_mask,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swap_chain_images[current_frame],
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        vk::DependencyInfo dependency_info = {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier
        };
        command_buffer.pipelineBarrier2(dependency_info);
    }

    [[nodiscard]] vk::raii::ShaderModule CreateShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo create_info{ .codeSize = code.size() * sizeof(char), .pCode = reinterpret_cast<const uint32_t*>(code.data()) };
        vk::raii::ShaderModule shader_module{ device, create_info };

        return shader_module;
    }

    static uint32_t ChooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const & surface_capabilities) {
	    auto min_image_count = std::max(3u, surface_capabilities.minImageCount);
	    if ((0 < surface_capabilities.maxImageCount) && (surface_capabilities.maxImageCount < min_image_count)) {
		    min_image_count = surface_capabilities.maxImageCount;
	    }
	    return min_image_count;
    }

    static vk::SurfaceFormatKHR ChooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const & available_formats) {
	    assert(!available_formats.empty());
	    const auto format_it = std::ranges::find_if(
                available_formats,
			    [](const auto & format) { return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; });
	    return format_it != available_formats.end() ? *format_it : available_formats[0];
    }

    static vk::PresentModeKHR ChooseSwapPresentMode(const std::vector<vk::PresentModeKHR> & available_present_modes) {
	    assert(std::ranges::any_of(available_present_modes, [](auto present_mode){ return present_mode == vk::PresentModeKHR::eFifo; }));
	    return std::ranges::any_of(available_present_modes,
			    [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; } ) ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
	    if (capabilities.currentExtent.width != 0xFFFFFFFF) {
		    return capabilities.currentExtent;
	    }
	    int width, height;
	    glfwGetFramebufferSize(window, &width, &height);

	    return {
		    std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
		    std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
	    };
    }

    std::vector<const char*> GetRequiredExtensions() {
        uint32_t glfw_extension_count = 0;
        auto glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

        std::vector extensions(glfw_extensions, glfw_extensions + glfw_extension_count);
        if (enable_validation_layers) {
            extensions.push_back(vk::EXTDebugUtilsExtensionName);
        }

        return extensions;
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL DebugCallback(
            vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
            vk::DebugUtilsMessageTypeFlagsEXT type,
            const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void*) {
        if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
            severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
            std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
        }

        return vk::False;
    }

    static std::vector<char> read_file(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }
        std::vector<char> buffer(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        file.close();
        return buffer;
    }
};

int main() {
    try {
        App app;
        app.Run();
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
