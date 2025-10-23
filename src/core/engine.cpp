#include "engine.h"
#include "window.h"

Engine::Engine(Window& window_ref) {
    this->window = &window_ref;

    // Make sure window notifies when window resizes
    this->window->SetFramebufferResizeCallback([this]() { frame_buffer_resized = true; });

    CreateInstance();
    SetupDebugMessenger();
}

void Engine::Cleanup() {
    //device.waitIdle();
}





void Engine::CreateInstance() {

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
                    { return strcmp(layer_property.layerName, required_layer) == 0; })) {
            throw std::runtime_error("required layer not supported: " + std::string(required_layer));
        }
    }

    // Get required extensions
    auto required_extensions = window->GetRequiredVulkanExtensions();
    if (enable_validation_layers) {
        required_extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

    // Check if required extensions are supported by Vulkan implementation
    auto extension_properties = context.enumerateInstanceExtensionProperties();
    for (auto const& required_extension : required_extensions) {
        if (std::ranges::none_of(extension_properties,
                    [required_extension](auto const& extension_property)
                    { return strcmp(extension_property.extensionName, required_extension) == 0; })) {
            throw std::runtime_error("required extension not supported: " + std::string(required_extension));
        }
    }

    constexpr vk::ApplicationInfo app_info{
        .applicationVersion = VK_MAKE_VERSION(1, 4, 0),
        .engineVersion = VK_MAKE_VERSION(1, 4, 0),
        .apiVersion = vk::ApiVersion14
    };

    vk::InstanceCreateInfo create_info{
        .pApplicationInfo = &app_info,
        .enabledLayerCount = static_cast<uint32_t>(required_layers.size()),
        .ppEnabledLayerNames = required_layers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(required_extensions.size()),
        .ppEnabledExtensionNames = required_extensions.data()
    };

    instance = vk::raii::Instance(context, create_info);
}

void Engine::SetupDebugMessenger() {
    if (!enable_validation_layers) return;

    vk::DebugUtilsMessageSeverityFlagsEXT severity_flags(
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
    vk::DebugUtilsMessageTypeFlagsEXT message_type_flags(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

    vk::DebugUtilsMessengerCreateInfoEXT debug_utils_messenger_create_info_EXT{
        .messageSeverity = severity_flags,
        .messageType = message_type_flags,
        .pfnUserCallback = &DebugCallback
    };
    debug_messenger = instance.createDebugUtilsMessengerEXT(debug_utils_messenger_create_info_EXT);
}

void Engine::CreateSurface() {
    VkSurfaceKHR _surface;
    if (window->CreateVulkanWindowSurface(*instance, nullptr, &_surface) != 0) {
        throw std::runtime_error("failed to create window surface");
    }
    surface = vk::raii::SurfaceKHR(instance, _surface);
}






//
// Runtime
//

VKAPI_ATTR vk::Bool32 VKAPI_CALL Engine::DebugCallback(
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

void Engine::Tick() {
}
