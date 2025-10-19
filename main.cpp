#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <algorithm>

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
    vk::raii::Queue                  queue           = nullptr;
    vk::raii::SwapchainKHR           swap_chain      = nullptr;
    std::vector<vk::Image>	     swap_chain_images;
    vk::SurfaceFormatKHR	     swap_chain_surface_format;
    vk::Extent2D		     swap_chain_extent;
    std::vector<vk::raii::ImageView> swap_chain_image_views;

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
        //CreateGraphicsPipeline();
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
        uint32_t queue_index = ~0u;
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
	    auto surface_capabilities = physical_device.getSurfaceCapabilities(*surface);
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
								.compositeAlpha   = vk::CompositeAlphaFlagBitsKHR:eOpaque,
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
	    glfwGetFrameBufferSize(window, &width, &height);

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
