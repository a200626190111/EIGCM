#ifndef SPECULARCONTRAST_OPTIONS_H
#define SPECULARCONTRAST_OPTIONS_H

#include <array>
#include <string>
#include <vector>

namespace specularcontrast {

typedef std::array<double, 3> Vec3;

struct Options {
    bool no_header = false;
    bool no_hit_check = false;
    bool integrated_path_check = false;
    bool separate_path_prefilter = false;
    bool include_direct_sun = false;
    bool all_normal_pairs = false;
    bool direct_specular_only = false;
    bool adaptive_sun_disk = false;
    bool adaptive_rough_cells = false;
    bool cluster_rough_sources = false;
    bool auto_materials = false;
    bool ideal_only = false;
    bool origin_reuse = true;
    bool quiet = false;
    std::string views_path;
    std::string normals_path;
    std::string normal_rad_path;
    std::string suns_path;
    std::string mirror_modifiers_path;
    std::string auto_material_allowlist_path;
    std::string proposal_modifiers_path;
    std::string transparent_modifiers_path;
    std::string nonspec_octree;
    std::string reflection_octree;
    std::string output_path;
    std::string mirror_illuminance_output_path;
    std::string save_normals_path;
    std::string save_paths_path;
    std::string oconv = "oconv";
    std::string rcontrib;
    std::string rtrace = "rtrace";
    std::string octree;
    std::vector<std::string> rcontrib_options = {
        "-ab", "0", "-lw", "1e-7", "-st", "0",
        "-dj", "0", "-dt", "0", "-dc", "1"
    };
    int nproc = 1;
    int batch_size = 32;
    int view_batch_size = 32;
    int reflection_level = 5;
    int allowlist_fallback_level = -1;
    int reflection_seed = 0;
    int max_specular_bounces = 1;
    int max_transparent_hits = 4;
    int sun_disk_samples = 1;
    int secondary_sun_disk_samples = 0;
    int sun_disk_pilot_samples = 16;
    int secondary_sun_disk_pilot_samples = 4;
    int sun_disk_seed = 0;
    int rough_samples = 0;
    int rough_pilot_samples = 64;
    int rough_medium_samples = 1024;
    int rough_secondary_samples = 16;
    int rough_seed = 0;
    double threshold = 2000.0;
    double normal_tolerance = 0.1;
    double visible_fraction = 1.0;
    double roughness = 0.0;
    double rough_extent = 3.0;
    double rough_pilot_threshold = 0.0;
    double rough_pilot_guard_angle = 0.0;
    double rough_pilot_anchor_angle = 5.0;
    double rough_cell_trigger = 0.5;
    double sun_disk_pilot_guard_angle = 8.0;
    Vec3 up = {{0.0, 0.0, 1.0}};
};

Options parse_options(int argc, char *argv[]);
void validate_options(const Options &options);

} // namespace specularcontrast

#endif
