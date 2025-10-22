#include "core/engine.h"
#include "core/window.h"

int main() {

    // Initialisation
    Window window(800, 600, "Engine");
    Engine engine(window);

    // Main loop
    while (!window.ShouldClose()) {
        window.Tick();
        engine.Tick();
    }

    // Cleanup
    engine.Cleanup();
    window.Cleanup();

    return 0;
}
