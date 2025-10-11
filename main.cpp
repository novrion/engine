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
    void run() {
        init_window();
        init_vulkan();
        main_loop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;

    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT debug_messenger = nullptr;

    vk::raii::PhysicalDevice physical_device = nullptr;
    vk::raii::Device device = nullptr;

    std::vector<const char*> device_extensions = {
        vk::KHRSwapchainExtensionName,
        vk::KHRSpirv14ExtensionName,
        vk::KHRSynchronization2ExtensionName,
        vk::KHRCreateRenderpass2ExtensionName
    };

    void init_window() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Engine", nullptr, nullptr);
    }

    void init_vulkan() {
        create_instance();
        setup_debug_messenger();
        pick_physical_device();
       // create_logical_device();
    }

    void main_loop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void create_instance() {
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
        auto required_extensions = get_required_extensions();

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

    void setup_debug_messenger() {
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
        debug_utils_messenger_create_info_EXT.pfnUserCallback = &debug_callback;

        debug_messenger = instance.createDebugUtilsMessengerEXT(debug_utils_messenger_create_info_EXT);
    }

    void pick_physical_device() {
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
                for (auto const & extension : device_extensions) {
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

    std::vector<const char*> get_required_extensions() {
        uint32_t glfw_extension_count = 0;
        auto glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

        std::vector extensions(glfw_extensions, glfw_extensions + glfw_extension_count);
        if (enable_validation_layers) {
            extensions.push_back(vk::EXTDebugUtilsExtensionName);
        }

        return extensions;
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debug_callback(
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
    App app;

    try {
        app.run();
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
