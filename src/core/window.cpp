#include "window.h"

//
// Initialisation
//

Window::Window(uint32_t width, uint32_t height, const char* title):
    width(width), height(height), title(title) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, FramebufferSizeCallback);
}

void Window::Cleanup() {
    glfwDestroyWindow(window);
    glfwTerminate();
}

void Window::FramebufferSizeCallback(GLFWwindow* window, int width, int height) {
    auto win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    win->width = width;
    win->height = height;

    // tell engine window resized
    if (win->resize_callback) win->resize_callback();
}

void Window::SetFramebufferResizeCallback(std::function<void()> callback) {
    resize_callback = callback;
}

//
// Runtime
//

bool Window::ShouldClose() {
    return glfwWindowShouldClose(window);
}

void Window::Tick() {
    glfwPollEvents();
}

//
// Vulkan
//

std::vector<const char*> Window::GetRequiredVulkanExtensions() const {
    uint32_t glfw_extension_count = 0;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
    std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);
    return extensions;
}

VkResult Window::CreateVulkanWindowSurface(VkInstance instance, const VkAllocationCallbacks* allocator, VkSurfaceKHR* surface) const {
    return glfwCreateWindowSurface(instance, window, allocator, surface);
}
