#ifndef SPECULAR_RCONTRIB_BACKEND_H
#define SPECULAR_RCONTRIB_BACKEND_H

#include <array>
#include <memory>
#include <string>
#include <vector>

enum SpecularMaterialClass {
    SPECULAR_MATERIAL_NONE,
    SPECULAR_MATERIAL_IDEAL,
    SPECULAR_MATERIAL_ROUGH,
    SPECULAR_MATERIAL_EXCLUDED,
    SPECULAR_MATERIAL_UNSUPPORTED
};

struct SpecularMaterialInfo {
    std::string name;
    std::string type;
    SpecularMaterialClass material_class;
    double roughness;
    double specular_fraction;
};

struct SpecularContribRay {
    std::array<double, 3> origin;
    std::array<double, 3> direction;
    std::size_t target_modifier;
    unsigned int reflection_orders;
};

class SpecularRcontribBackend {
public:
    SpecularRcontribBackend(
        int worker_count, const std::vector<std::string> &render_options,
        bool direct_specular_only = false,
        bool integrated_path_check = false,
        const std::vector<std::string> &reflection_modifiers =
            std::vector<std::string>());
    ~SpecularRcontribBackend();

    void load_scene(const std::string &octree);
    std::vector<SpecularMaterialInfo> inspect_specular_materials() const;
    void set_reflection_modifiers(
        const std::vector<std::string> &reflection_modifiers);
    std::vector<float> trace(
        const std::vector<std::string> &modifiers,
        const std::vector<SpecularContribRay> &rays);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
