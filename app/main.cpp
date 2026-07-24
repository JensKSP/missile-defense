#include "game_window.hpp"

#include <QGuiApplication>
#include <QVulkanInstance>
#include <string_view>

int main(int argc, char** argv) {
    QGuiApplication app(argc, argv);
    QGuiApplication::setOrganizationName("MissileDefense");
    QGuiApplication::setApplicationName("MissileDefense"); // stable app-data path for highscores

    QVulkanInstance instance;
#ifndef NDEBUG
    instance.setLayers({"VK_LAYER_KHRONOS_validation"});
#endif
    if (!instance.create()) {
        qFatal("Failed to create Vulkan instance: %d", static_cast<int>(instance.errorCode()));
        return 1;
    }

    md::GameWindow window;
    window.setVulkanInstance(&instance);
    window.resize(1280, 720);
    window.setTitle("Missile Defense");
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--play") {
            window.play_now(); // boot straight into a game (skip the menu)
        }
    }
    window.show();

    return app.exec();
}
