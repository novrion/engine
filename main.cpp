// https://github.com/KhronosGroup/Vulkan-Tutorial/tree/main/attachments
// https://docs.vulkan.org/tutorial/latest/10_Multisampling.html

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <limits>
#include <array>
#include <chrono>
#include <assert.h>

#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vk_platform.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validation_layers = { "VK_LAYER_KHRONOS_validation" };

#ifdef NDEBUG
    const bool enable_validation_layers = false;
#else
    const bool enable_validation_layers = true;
#endif

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 tex_coord;

    static vk::VertexInputBindingDescription GetBindingDescription() {
        return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
    }

    static std::array<vk::VertexInputAttributeDescription, 3> GetAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, tex_coord))
        };
    }

    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && tex_coord == other.tex_coord;
    }
};

template<> struct std::hash<Vertex> {
    size_t operator()(Vertex const& vertex) const noexcept {
        return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.tex_coord) << 1);
    }
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

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
    std::vector<vk::Image>	         swap_chain_images;
    vk::SurfaceFormatKHR	         swap_chain_surface_format;
    vk::Extent2D		             swap_chain_extent;
    std::vector<vk::raii::ImageView> swap_chain_image_views;

    vk::raii::DescriptorSetLayout descriptor_set_layout = nullptr;
    vk::raii::PipelineLayout      pipeline_layout       = nullptr;
    vk::raii::Pipeline            graphics_pipeline     = nullptr;

    vk::raii::Image depth_image               = nullptr;
    vk::raii::DeviceMemory depth_image_memory = nullptr;
    vk::raii::ImageView depth_image_view      = nullptr;

    uint32_t mip_levels = 0;
    vk::raii::Image texture_image               = nullptr;
    vk::raii::DeviceMemory texture_image_memory = nullptr;
    vk::raii::ImageView texture_image_view      = nullptr;
    vk::raii::Sampler texture_sampler           = nullptr;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    vk::raii::Buffer vertex_buffer              = nullptr;
    vk::raii::DeviceMemory vertex_buffer_memory = nullptr;
    vk::raii::Buffer index_buffer               = nullptr;
    vk::raii::DeviceMemory index_buffer_memory  = nullptr;

    std::vector<vk::raii::Buffer> uniform_buffers;
    std::vector<vk::raii::DeviceMemory> uniform_buffers_memory;
    std::vector<void*> uniform_buffers_mapped;

    vk::raii::DescriptorPool descriptor_pool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptor_sets;

    vk::raii::CommandPool command_pool = nullptr;
    std::vector<vk::raii::CommandBuffer> command_buffers;

    std::vector<vk::raii::Semaphore> present_complete_semaphores;
    std::vector<vk::raii::Semaphore> render_finished_semaphores;
    std::vector<vk::raii::Fence>     in_flight_fences;
    uint32_t semaphore_index = 0;
    uint32_t current_frame = 0;

    bool frame_buffer_resized = false;

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
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, FrameBufferResizeCallback);
    }

    static void FrameBufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<App*>(glfwGetWindowUserPointer(window));
        app->frame_buffer_resized = true;
    }

    void InitVulkan() {
        CreateInstance();
        SetupDebugMessenger();
        CreateSurface();
        PickPhysicalDevice();
        CreateLogicalDevice();
        CreateSwapChain();
        CreateImageViews();
        CreateDescriptorSetLayout();
        CreateGraphicsPipeline();
        CreateCommandPool();
        CreateDepthResources();
        CreateTextureImage();
        CreateTextureImageView();
        CreateTextureSampler();
        LoadModel();
        CreateVertexBuffer();
        CreateIndexBuffer();
        CreateUniformBuffers();
        CreateDescriptorPool();
        CreateDescriptorSets();
        CreateCommandBuffers();
        CreateSyncObjects();
    }

    void MainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            DrawFrame();
        }

        device.waitIdle();
    }

    void CleanupSwapChain() {
        swap_chain_image_views.clear();
        swap_chain = nullptr;
    }

    void Cleanup() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void RecreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device.waitIdle();

        CleanupSwapChain();
        CreateSwapChain();
        CreateImageViews();
        CreateDepthResources();
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
                    found = found && extension_iter != extensions.end();
                }

                auto features = device.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
                bool supports_required_features = features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
                                                  features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
                                                  features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                                  features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;
                is_suitable = is_suitable && found && supports_required_features;
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
        vk::StructureChain<vk::PhysicalDeviceFeatures2,
                           vk::PhysicalDeviceVulkan11Features,
                           vk::PhysicalDeviceVulkan13Features,
                           vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
            feature_chain = {
                vk::PhysicalDeviceFeatures2{.features = {.samplerAnisotropy = true }},
                vk::PhysicalDeviceVulkan11Features{.shaderDrawParameters = true},
                vk::PhysicalDeviceVulkan13Features{.synchronization2 = true, .dynamicRendering = true},
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

	    vk::ImageViewCreateInfo image_view_create_info {
            .viewType = vk::ImageViewType::e2D,
		    .format = swap_chain_surface_format.format,
			.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 } };
	    for (auto image : swap_chain_images) {
		    image_view_create_info.image = image;
		    swap_chain_image_views.emplace_back(device, image_view_create_info);
	    }
    }

    void CreateDescriptorSetLayout() {
        std::array bindings = {
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr),
        };
        vk::DescriptorSetLayoutCreateInfo layout_info{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        };
        descriptor_set_layout = vk::raii::DescriptorSetLayout(device, layout_info);
    }

    void CreateGraphicsPipeline() {
        vk::raii::ShaderModule vert_shader_module = CreateShaderModule(read_file("shaders/shader.vert.spv"));
        vk::raii::ShaderModule frag_shader_module = CreateShaderModule(read_file("shaders/shader.frag.spv"));

        vk::PipelineShaderStageCreateInfo vert_shader_stage_info { .stage = vk::ShaderStageFlagBits::eVertex, .module = vert_shader_module, .pName = "main" };
        vk::PipelineShaderStageCreateInfo frag_shader_stage_info { .stage = vk::ShaderStageFlagBits::eFragment, .module = frag_shader_module, .pName = "main" };
        vk::PipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info, frag_shader_stage_info};

        auto binding_description = Vertex::GetBindingDescription();
        auto attribute_descriptions = Vertex::GetAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertex_input_info{ .vertexBindingDescriptionCount = 1,
                                                                  .pVertexBindingDescriptions = &binding_description,
                                                                  .vertexAttributeDescriptionCount = static_cast<uint32_t>(attribute_descriptions.size()),
                                                                  .pVertexAttributeDescriptions = attribute_descriptions.data() };
        vk::PipelineInputAssemblyStateCreateInfo input_assembly{ .topology = vk::PrimitiveTopology::eTriangleList };
        vk::PipelineViewportStateCreateInfo viewport_state{ .viewportCount = 1, .scissorCount = 1 };

        vk::PipelineRasterizationStateCreateInfo rasterizer{ .depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False,
                                                             .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eBack,
                                                             .frontFace = vk::FrontFace::eCounterClockwise, .depthBiasEnable = vk::False,
                                                             .depthBiasSlopeFactor = 1.0f, .lineWidth = 1.0f };

        vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

        vk::PipelineDepthStencilStateCreateInfo depth_stencil{
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False
        };

        vk::PipelineColorBlendAttachmentState color_blend_attachment{ .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        vk::PipelineColorBlendStateCreateInfo color_blending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &color_blend_attachment };

        std::vector dynamic_states = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamic_state{ .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()), .pDynamicStates = dynamic_states.data() };

        vk::PipelineLayoutCreateInfo pipeline_layout_info{ .setLayoutCount = 1, .pSetLayouts = &*descriptor_set_layout, .pushConstantRangeCount = 0 };

        pipeline_layout = vk::raii::PipelineLayout(device, pipeline_layout_info);

        vk::Format depth_format = FindDepthFormat();

        vk::PipelineRenderingCreateInfo pipeline_rendering_create_info{
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &swap_chain_surface_format.format,
            .depthAttachmentFormat = depth_format
        };
        vk::GraphicsPipelineCreateInfo pipeline_info{
            .pNext = &pipeline_rendering_create_info,
            .stageCount = 2,
            .pStages = shader_stages,
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depth_stencil,
            .pColorBlendState = &color_blending,
            .pDynamicState = &dynamic_state,
            .layout = pipeline_layout,
            .renderPass = nullptr
        };

        graphics_pipeline = vk::raii::Pipeline(device, nullptr, pipeline_info);
    }

    void CreateCommandPool() {
        vk::CommandPoolCreateInfo pool_info{ .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                             .queueFamilyIndex = queue_index };
        command_pool = vk::raii::CommandPool(device, pool_info);
    }

    void CreateDepthResources() {
        vk::Format depth_format = FindDepthFormat();

        CreateImage(swap_chain_extent.width, swap_chain_extent.height, 1, depth_format, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depth_image, depth_image_memory);
        depth_image_view = CreateImageView(depth_image, depth_format, vk::ImageAspectFlagBits::eDepth, 1);
    }

    vk::Format FindSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) const {
        for (const auto format : candidates) {
            vk::FormatProperties props = physical_device.getFormatProperties(format);
            if ((tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) ||
                (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features)) {
                    return format;
                }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    [[nodiscard]] vk::Format FindDepthFormat() const {
        return FindSupportedFormat(
            {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment
        );
    }

    bool HasStencilComponent(vk::Format format) {
        return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
    }

    void CreateTextureImage() {
        int tex_width, tex_height, tex_channels;
        stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);
        vk::DeviceSize image_size = tex_width * tex_height * 4;
        mip_levels = static_cast<uint32_t>(std::floor(std::log2(std::max(tex_width, tex_height)))) + 1;

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        vk::raii::Buffer staging_buffer({});
        vk::raii::DeviceMemory staging_buffer_memory({});
        CreateBuffer(image_size, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, staging_buffer, staging_buffer_memory);

        void* data = staging_buffer_memory.mapMemory(0, image_size);
        memcpy(data, pixels, image_size);
        staging_buffer_memory.unmapMemory();

        stbi_image_free(pixels);

        CreateImage(tex_width, tex_height, mip_levels, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, texture_image, texture_image_memory);

        TransitionImageLayout(texture_image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mip_levels);
        CopyBufferToImage(staging_buffer, texture_image, static_cast<uint32_t>(tex_width), static_cast<uint32_t>(tex_height));
        
        GenerateMipmaps(texture_image, vk::Format::eR8G8B8A8Srgb, tex_width, tex_height, mip_levels);
    }

    void GenerateMipmaps(vk::raii::Image& image, vk::Format image_format, int32_t tex_width, int32_t tex_height, uint32_t mip_levels) {
        // Check if image format supports linear blit-ing
        vk::FormatProperties format_properties = physical_device.getFormatProperties(image_format);

        if (!(format_properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
            throw std::runtime_error("texture image format does not support linear blitting!");
        }

        std::unique_ptr<vk::raii::CommandBuffer> command_buffer = BeginSingleTimeCommands();

        vk::ImageMemoryBarrier barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = image,
        };
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;

        int32_t mip_width = tex_width;
        int32_t mip_height = tex_height;

        for (uint32_t i = 1; i < mip_levels; i++) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

            command_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);

            vk::ArrayWrapper1D<vk::Offset3D, 2> offsets, dstOffsets;
            offsets[0] = vk::Offset3D(0, 0, 0);
            offsets[1] = vk::Offset3D(mip_width, mip_height, 1);
            dstOffsets[0] = vk::Offset3D(0, 0, 0);
            dstOffsets[1] = vk::Offset3D(mip_width > 1 ? mip_width / 2 : 1, mip_height > 1 ? mip_height / 2 : 1, 1);
            vk::ImageBlit blit = {
                .srcSubresource = {},
                .srcOffsets = offsets,
                .dstSubresource = {},
                .dstOffsets = dstOffsets
            };
            blit.srcSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i - 1, 0, 1);
            blit.dstSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i, 0, 1);

            command_buffer->blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, { blit }, vk::Filter::eLinear);

            barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            command_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

            if (mip_width > 1) mip_width /= 2;
            if (mip_height > 1) mip_height /= 2;
        }

        barrier.subresourceRange.baseMipLevel = mip_levels - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        command_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

        EndSingleTimeCommands(*command_buffer);
    }

    void CreateTextureImageView() {
        texture_image_view = CreateImageView(texture_image, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mip_levels);
    }

    void CreateTextureSampler() {
        vk::PhysicalDeviceProperties properties = physical_device.getProperties();
        vk::SamplerCreateInfo sampler_info{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .mipLodBias = 0.0f,
            .anisotropyEnable = vk::True,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = vk::False,
            .compareOp = vk::CompareOp::eAlways };
        texture_sampler = vk::raii::Sampler(device, sampler_info);
    }

    [[nodiscard]] vk::raii::ImageView CreateImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspect_flags, uint32_t mip_levels) const {
        vk::ImageViewCreateInfo view_info{
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {aspect_flags, 0, mip_levels, 0, 1}
        };
        return vk::raii::ImageView(device, view_info);
    }

    void CreateImage(uint32_t width, uint32_t height, uint32_t mip_levels, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& image_memory) {
        vk::ImageCreateInfo image_info{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {width, height, 1},
            .mipLevels = mip_levels,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined
        };

        image = vk::raii::Image(device, image_info);

        vk::MemoryRequirements mem_requirements = image.getMemoryRequirements();
        vk::MemoryAllocateInfo alloc_info{ .allocationSize = mem_requirements.size,
                                           .memoryTypeIndex = FindMemoryType(mem_requirements.memoryTypeBits, properties) };
        image_memory = vk::raii::DeviceMemory(device, alloc_info);
        image.bindMemory(image_memory, 0);
    }

    void TransitionImageLayout(const vk::raii::Image& image, vk::ImageLayout old_layout, vk::ImageLayout new_layout, uint32_t mip_levels) {
        auto command_buffer = BeginSingleTimeCommands();

        vk::ImageMemoryBarrier barrier{ .oldLayout = old_layout, .newLayout = new_layout,
                                        .image = image,
                                        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, mip_levels, 0, 1 } };

        vk::PipelineStageFlags source_stage;
        vk::PipelineStageFlags destination_stage;

        if (old_layout == vk::ImageLayout::eUndefined && new_layout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            source_stage = vk::PipelineStageFlagBits::eTopOfPipe;
            destination_stage = vk::PipelineStageFlagBits::eTransfer;
        } else if (old_layout == vk::ImageLayout::eTransferDstOptimal && new_layout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            source_stage = vk::PipelineStageFlagBits::eTransfer;
            destination_stage = vk::PipelineStageFlagBits::eFragmentShader;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }
        command_buffer->pipelineBarrier(source_stage, destination_stage, {}, {}, nullptr, barrier);
        EndSingleTimeCommands(*command_buffer);
    }

    void CopyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height) {
        std::unique_ptr<vk::raii::CommandBuffer> command_buffer = BeginSingleTimeCommands();
        vk::BufferImageCopy region{ .bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0,
                                    .imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
                                    .imageOffset = {0, 0, 0}, .imageExtent = {width, height, 1} };
        command_buffer->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});
        EndSingleTimeCommands(*command_buffer);
    }

    void LoadModel() {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
            throw std::runtime_error(warn + err);
        }

        std::unordered_map<Vertex, uint32_t> unique_vertices{};

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};

                vertex.pos = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.tex_coord = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };

                vertex.color = {1.0f, 1.0f, 1.0f};

                if (!unique_vertices.contains(vertex)) {
                    unique_vertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                indices.push_back(unique_vertices[vertex]);
            }
        }
    }

    void CreateVertexBuffer() {
        vk::DeviceSize buffer_size = sizeof(vertices[0]) * vertices.size();
        vk::raii::Buffer staging_buffer({});
        vk::raii::DeviceMemory staging_buffer_memory({});
        CreateBuffer(buffer_size, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, staging_buffer, staging_buffer_memory);

        void* data_staging = staging_buffer_memory.mapMemory(0, buffer_size);
        memcpy(data_staging, vertices.data(), buffer_size);
        staging_buffer_memory.unmapMemory();

        CreateBuffer(buffer_size, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertex_buffer, vertex_buffer_memory);

        CopyBuffer(staging_buffer, vertex_buffer, buffer_size);
    }

    void CreateIndexBuffer() {
        vk::DeviceSize buffer_size = sizeof(indices[0]) * indices.size();

        vk::raii::Buffer staging_buffer({});
        vk::raii::DeviceMemory staging_buffer_memory({});
        CreateBuffer(buffer_size, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, staging_buffer, staging_buffer_memory);

        void* data = staging_buffer_memory.mapMemory(0, buffer_size);
        memcpy(data, indices.data(), (size_t) buffer_size);
        staging_buffer_memory.unmapMemory();

        CreateBuffer(buffer_size, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, index_buffer, index_buffer_memory);

        CopyBuffer(staging_buffer, index_buffer, buffer_size);
    }

    void CreateUniformBuffers() {
        uniform_buffers.clear();
        uniform_buffers_memory.clear();
        uniform_buffers_mapped.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DeviceSize buffer_size = sizeof(UniformBufferObject);
            vk::raii::Buffer buffer({});
            vk::raii::DeviceMemory buffer_mem({});
            CreateBuffer(buffer_size, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, buffer_mem);
            uniform_buffers.emplace_back(std::move(buffer));
            uniform_buffers_memory.emplace_back(std::move(buffer_mem));
            uniform_buffers_mapped.emplace_back(uniform_buffers_memory[i].mapMemory(0, buffer_size));
        }
    }

    void CreateDescriptorPool() {
        std::array pool_size {
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT)
        };
        vk::DescriptorPoolCreateInfo pool_info{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(pool_size.size()),
            .pPoolSizes = pool_size.data()
        };
        descriptor_pool = vk::raii::DescriptorPool(device, pool_info);
    }

    void CreateDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptor_set_layout);
        vk::DescriptorSetAllocateInfo alloc_info{
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()
        };

        descriptor_sets.clear();
        descriptor_sets = device.allocateDescriptorSets(alloc_info);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DescriptorBufferInfo buffer_info{
                .buffer = uniform_buffers[i],
                .offset = 0,
                .range = sizeof(UniformBufferObject)
            };
            vk::DescriptorImageInfo image_info{
                .sampler = texture_sampler,
                .imageView = texture_image_view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            };
            std::array descriptor_writes {
                vk::WriteDescriptorSet{
                    .dstSet = descriptor_sets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &buffer_info
                },
                vk::WriteDescriptorSet{
                    .dstSet = descriptor_sets[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &image_info
                }
            };
            device.updateDescriptorSets(descriptor_writes, {});
        }
    }

    void CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& buffer_memory) {
        vk::BufferCreateInfo buffer_info{ .size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive };
        buffer = vk::raii::Buffer(device, buffer_info);
        vk::MemoryRequirements mem_requirements = buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo alloc_info{ .allocationSize = mem_requirements.size, .memoryTypeIndex = FindMemoryType(mem_requirements.memoryTypeBits, properties) };
        buffer_memory = vk::raii::DeviceMemory(device, alloc_info);
        buffer.bindMemory(buffer_memory, 0);
    }

    std::unique_ptr<vk::raii::CommandBuffer> BeginSingleTimeCommands() {
        vk::CommandBufferAllocateInfo alloc_info{ .commandPool = command_pool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
        std::unique_ptr<vk::raii::CommandBuffer> command_buffer = std::make_unique<vk::raii::CommandBuffer>(std::move(vk::raii::CommandBuffers(device, alloc_info).front()));

        vk::CommandBufferBeginInfo begin_info{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
        command_buffer->begin(begin_info);

        return command_buffer;
    }

    void EndSingleTimeCommands(vk::raii::CommandBuffer& command_buffer) {
        command_buffer.end();

        vk::SubmitInfo submit_info{ .commandBufferCount = 1, .pCommandBuffers = &*command_buffer };
        queue.submit(submit_info, nullptr);
        queue.waitIdle();
    }

    void CopyBuffer(vk::raii::Buffer & src_buffer, vk::raii::Buffer & dst_buffer, vk::DeviceSize size) {
        vk::CommandBufferAllocateInfo alloc_info{ .commandPool = command_pool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount =  1};
        vk::raii::CommandBuffer command_copy_buffer = std::move(device.allocateCommandBuffers(alloc_info).front());
        command_copy_buffer.begin(vk::CommandBufferBeginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        command_copy_buffer.copyBuffer(*src_buffer, *dst_buffer, vk::BufferCopy(0, 0, size));
        command_copy_buffer.end();
        queue.submit(vk::SubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*command_copy_buffer }, nullptr);
        queue.waitIdle();
    }

    uint32_t FindMemoryType(uint32_t type_filter, vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties mem_properties = physical_device.getMemoryProperties();

        for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
            if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void CreateCommandBuffers() {
        command_buffers.clear();
        vk::CommandBufferAllocateInfo alloc_info{ .commandPool = command_pool, .level = vk::CommandBufferLevel::ePrimary,
                                                  .commandBufferCount = MAX_FRAMES_IN_FLIGHT };
        command_buffers = vk::raii::CommandBuffers(device, alloc_info);
    }

    void RecordCommandBuffer(uint32_t image_index) {
        command_buffers[current_frame].begin( {} );

        // Before starting rendering, transition swapchain image to COLOR_ATTACHMENT_OPTIMAL
        transition_image_layout(
                image_index,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eColorAttachmentOptimal,
                {},                                                // srcAccessMask (no need to wait for previous operations)
                vk::AccessFlagBits2::eColorAttachmentWrite,        // dstAccessMask
                vk::PipelineStageFlagBits2::eTopOfPipe,            // srcStage
                vk::PipelineStageFlagBits2::eColorAttachmentOutput // dstStage
        );

        // Transition depth image to depth attachment optimal layout
        vk::ImageMemoryBarrier2 depth_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
            .srcAccessMask = {},
            .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
            .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = depth_image,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eDepth,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        vk::DependencyInfo depth_dependency_info = {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &depth_barrier
        };
        command_buffers[current_frame].pipelineBarrier2(depth_dependency_info);

        vk::ClearValue clear_color = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
        vk::ClearValue clear_depth = vk::ClearDepthStencilValue(1.0f, 0);

        vk::RenderingAttachmentInfo color_attachment_info = {
            .imageView = swap_chain_image_views[image_index],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clear_color
        };
        vk::RenderingAttachmentInfo depth_attachment_info = {
            .imageView = depth_image_view,
            .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .clearValue = clear_depth
        };

        vk::RenderingInfo rendering_info = {
            .renderArea = { .offset = { 0, 0 }, .extent = swap_chain_extent },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_attachment_info,
            .pDepthAttachment = &depth_attachment_info
        };

        command_buffers[current_frame].beginRendering(rendering_info);
        command_buffers[current_frame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_pipeline);
        command_buffers[current_frame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swap_chain_extent.width), static_cast<float>(swap_chain_extent.height), 0.0f, 1.0f));
        command_buffers[current_frame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swap_chain_extent));
        command_buffers[current_frame].bindVertexBuffers(0, *vertex_buffer, {0});
        command_buffers[current_frame].bindIndexBuffer(*index_buffer, 0, vk::IndexTypeValue<decltype(indices)::value_type>::value);
        command_buffers[current_frame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, *descriptor_sets[current_frame], nullptr);
        command_buffers[current_frame].drawIndexed(indices.size(), 1, 0, 0, 0);
        command_buffers[current_frame].endRendering();

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
        command_buffers[current_frame].end();
    }

    void transition_image_layout(
            uint32_t image_index,
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
            .image = swap_chain_images[image_index],
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
        command_buffers[current_frame].pipelineBarrier2(dependency_info);
    }

    void CreateSyncObjects() {
        present_complete_semaphores.clear();
        render_finished_semaphores.clear();
        in_flight_fences.clear();
        
        for (size_t i = 0; i < swap_chain_images.size(); i++) {
            present_complete_semaphores.emplace_back(device, vk::SemaphoreCreateInfo());
            render_finished_semaphores.emplace_back(device, vk::SemaphoreCreateInfo());
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            in_flight_fences.emplace_back(device, vk::FenceCreateInfo{ .flags = vk::FenceCreateFlagBits::eSignaled });
        }
    }

    void UpdateUniformBuffer(uint32_t current_image) {
        static auto start_time = std::chrono::high_resolution_clock::now();

        auto current_time = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(current_time - start_time).count();

        UniformBufferObject ubo{};
        ubo.model = rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));

        static float camX = 2.0f, camY = 2.0f, camZ = 2.0f;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camY += 0.01f;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camY -= 0.01f;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camX -= 0.01f;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camX += 0.01f;

        ubo.view = lookAt(glm::vec3(camX, camY, camZ), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(swap_chain_extent.width) / static_cast<float>(swap_chain_extent.height), 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        memcpy(uniform_buffers_mapped[current_image], &ubo, sizeof(ubo));
    }

    void DrawFrame() {
        while (vk::Result::eTimeout == device.waitForFences(*in_flight_fences[current_frame], vk::True, UINT64_MAX))
            ;

        // TODO: code duplication of error handling
        uint32_t image_index;
        vk::Result result;
        try {
            auto [res, idx] = swap_chain.acquireNextImage(UINT64_MAX, *present_complete_semaphores[semaphore_index], nullptr);
            result = res;
            image_index = idx;
        } catch (const vk::SystemError& e) {
            if (e.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR)) {
                RecreateSwapChain();
                return;
            } else {
                throw;
            }
        }

        if (result == vk::Result::eErrorOutOfDateKHR) {
            RecreateSwapChain();
            return;
        }
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        // TODO: END code duplication of error handling
        
        UpdateUniformBuffer(current_frame);

        device.resetFences(*in_flight_fences[current_frame]);
        command_buffers[current_frame].reset();
        RecordCommandBuffer(image_index);

        vk::PipelineStageFlags wait_destination_stage_mask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        const vk::SubmitInfo submit_info{ .waitSemaphoreCount = 1, .pWaitSemaphores = &*present_complete_semaphores[semaphore_index],
                                          .pWaitDstStageMask = &wait_destination_stage_mask, .commandBufferCount = 1, .pCommandBuffers = &*command_buffers[current_frame],
                                          .signalSemaphoreCount = 1, .pSignalSemaphores = &*render_finished_semaphores[image_index] };
        queue.submit(submit_info, *in_flight_fences[current_frame]);

        try {
            const vk::PresentInfoKHR present_info_KHR{ .waitSemaphoreCount = 1, .pWaitSemaphores = &*render_finished_semaphores[image_index],
                                                   .swapchainCount = 1, .pSwapchains = &*swap_chain, .pImageIndices = &image_index };
            result = queue.presentKHR(present_info_KHR);
            if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || frame_buffer_resized) {
                frame_buffer_resized = false;
                RecreateSwapChain();
            } else if (result != vk::Result::eSuccess) {
                throw std::runtime_error("failed to present swap chain image!");
            }
        }
        catch (const vk::SystemError& e) {
            if (e.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR)) {
                RecreateSwapChain();
                return;
            } else {
                throw;
            }
        }

        semaphore_index = (semaphore_index + 1) % present_complete_semaphores.size();
        current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
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
