#pragma once

#include <functional>
#include <vector>
#include <cstdint>

struct GLFWwindow;

class Window {
public:
    Window() = default;
    ~Window() = default;

    Window(uint32_t width, uint32_t height, const char* title);
    void Cleanup();

    bool ShouldClose();
    void Tick();

    // Engine
    void SetFramebufferResizeCallback(std::function<void()> callback);

    // Vulkan
    std::vector<const char*> GetRequiredVulkanExtensions() const;

private:
    GLFWwindow* window = nullptr;
    uint32_t width, height;
    const char* title;

    static void FramebufferSizeCallback(GLFWwindow* window, int width, int height);

    // Engine
    std::function<void()> resize_callback = nullptr;
};
