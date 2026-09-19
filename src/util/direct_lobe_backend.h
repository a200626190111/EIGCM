#ifndef DIRECT_LOBE_BACKEND_H
#define DIRECT_LOBE_BACKEND_H

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct DirectLobeRay {
    std::array<double, 3> origin;
    std::array<double, 3> direction;
    std::size_t target_modifier;
};

struct DirectLobeSensor {
    std::array<double, 3> origin;
    std::array<double, 3> direction;
};

enum DirectLobeTraceMode {
    DLT_DIRECT_LOBE,
    DLT_STRAIGHT_SUN
};

class DirectLobeBackend {
public:
    DirectLobeBackend(int worker_count, int accumulation,
                      const std::vector<std::string> &render_options);
    ~DirectLobeBackend();

    void load_scene(const std::string &octree);
    std::vector<float> trace(const std::vector<std::string> &modifiers,
                             const std::vector<DirectLobeRay> &rays,
                             DirectLobeTraceMode mode = DLT_DIRECT_LOBE);
    std::vector<float> trace_sun_irradiance(
        const std::vector<std::string> &modifiers,
        const std::vector<DirectLobeSensor> &sensors, int samples,
        std::vector<float> *total_irradiance = NULL,
        std::vector<float> *visible_fraction = NULL);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
