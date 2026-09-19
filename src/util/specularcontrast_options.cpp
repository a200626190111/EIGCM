#include "specularcontrast_options.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace specularcontrast {

namespace {

bool parse_double(const std::string &word, double &value)
{
    char *end = NULL;
    errno = 0;
    value = std::strtod(word.c_str(), &end);
    return !errno && end != word.c_str() && *end == '\0';
}

int parse_int(const std::string &word, const std::string &what)
{
    char *end = NULL;
    errno = 0;
    const long value = std::strtol(word.c_str(), &end, 10);
    if (errno || end == word.c_str() || *end || value < 0 ||
            value > std::numeric_limits<int>::max())
        throw std::runtime_error("invalid " + what + " '" + word + "'");
    return static_cast<int>(value);
}

std::vector<std::string> split_words(const std::string &text)
{
    std::istringstream input(text);
    std::vector<std::string> result;
    std::string word;
    while (input >> word)
        result.push_back(word);
    return result;
}

void require_file(const std::string &path, const std::string &label)
{
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input)
        throw std::runtime_error(label + " '" + path + "' cannot be opened");
}

void validate_rcontrib_options(const std::vector<std::string> &options)
{
    const char *forbidden[] = {"-f", "-h", "-V", "-m", "-M", "-n",
                               "-c", "-x", "-y", "-lr", "-ss"};
    for (std::size_t i = 0; i < options.size(); ++i)
        for (std::size_t j = 0;
                j < sizeof(forbidden)/sizeof(forbidden[0]); ++j)
            if (options[i] == forbidden[j] ||
                    options[i].compare(
                        0, std::strlen(forbidden[j]), forbidden[j]) == 0)
                throw std::runtime_error(
                    "rcontrib option '" + options[i] +
                    "' is managed by specularcontrast");
}

void usage(FILE *stream)
{
    std::fprintf(stream,
        "Usage: specularcontrast -vf views.pts [-N normals.txt | --normal-rad file.rad]\n"
        "       -S suns.rad [-M mirror.mod] [options] scene.oct\n\n"
        "Options:\n"
        "  -? | --help                 show this help\n"
        "  -h                          suppress Radiance matrix header\n"
        "  -N file                     load mirror normals (all ordered pairs for 2R)\n"
        "  --normal-rad file           extract normals (all ordered pairs for 2R)\n"
        "  --max-specular-bounces n    include 1 or 2 ideal reflections (default 1)\n"
        "  --all-normal-pairs          test all ordered pairs for 2R geometry\n"
        "  --roughness alpha           effective Radiance roughness (enables rough mode)\n"
        "  --rough-samples n           square Shirley-Chiu sample count (default 0=off)\n"
        "  --adaptive-rough-sampling   center-test rough paths, then refine only retained paths\n"
        "  --rough-pilot-samples n     rescue samples after a rejected center ray (default 64)\n"
        "  --rough-pilot-guard-angle d refine only near hit sun directions above 16 samples\n"
        "  --rough-pilot-anchor-angle d angular spacing of annual rescue anchors (default 5)\n"
        "  --adaptive-rough-cells      refine only bright 16x16 rough-cap cells to final resolution\n"
        "  --rough-medium-samples n    accepted adaptive integration level (default 1024)\n"
        "  --rough-cell-trigger f      cell-refinement trigger as glare-threshold fraction (default 0.5)\n"
        "  --rough-secondary-samples n Radiance samples at the second rough bounce (2R only; default 16)\n"
        "  --rough-extent sigma        sampled lobe radius in standard deviations (default 3)\n"
        "  --rough-pilot-threshold f   pilot luminance as a fraction of the glare threshold (default 1e-6)\n"
        "  --rough-prefilter fraction  legacy alias for --rough-pilot-threshold\n"
        "  --rough-seed seed           deterministic rough-cap rotation seed (default 0)\n"
        "  --cluster-rough-sources     cluster each sampled rough lobe before DGP evaluation\n"
        "  --auto-materials            classify ordinary Radiance reflectors and sample ideal/rough paths automatically\n"
        "  --auto-material-allowlist f restrict --auto-materials to listed material identifiers\n"
        "  --allowlist-fallback-level n re-search missing allowlisted materials at level n\n"
        "  --ideal-only                automatically retain ideal reflectors and ignore rough materials\n"
        "  --reflection-level level    path presearch level; increase to recover missed view paths (default 5)\n"
        "  --reflection-seed seed      reproducible search jitter seed (default 0)\n"
        "  --reflection-octree file    prebuilt scene containing a skyglow sky\n"
        "  --normal-samples count      legacy alias; count must equal 2*4^level\n"
        "  --save-normals file         save normals with view/modifier/material/primitive sources\n"
        "  --save-reflection-paths f   save sourced ordered 2R paths and originating views\n"
        "  -M file                     optional mirror material filter (default all)\n"
        "  --proposal-modifiers file   material filter used only for path-tree proposals\n"
        "  --transparent-modifiers f   legacy external prefilter pass list only\n"
        "  --max-transparent-hits n    legacy prefilter crossing limit (default 4)\n"
        "  --sun-disk-samples n        square equal-solid-angle disk samples (default 1)\n"
        "  --secondary-sun-disk-samples n samples for 2R-only paths (default: sun-disk value)\n"
        "  --adaptive-sun-disk         pilot-test paths before full solar-disk integration\n"
        "  --sun-disk-pilot-samples n pilot samples for direct/1R paths (default 16)\n"
        "  --secondary-sun-disk-pilot-samples n pilot samples for 2R-only paths (default 4)\n"
        "  --sun-disk-pilot-guard-angle d protect matching paths within d degrees (default 8)\n"
        "  --sun-disk-seed seed        deterministic solar-disk rotation seed (default 0)\n"
        "  --include-direct-sun        add deterministic 0R solar-disk contrast\n"
        "  --no-hit-check              ignore -M during hit checks (2R geometry still checked)\n"
        "  --integrated-path-check     validate paths during final Radiance tracing (default with -M)\n"
        "  --separate-path-prefilter   use the legacy external rtrace path check\n"
        "  --nonspec-octree file       matching scene with target specularity set to 0\n"
        "  --direct-specular-only      suppress direct diffuse terms (built-in backend)\n"
        "  --no-origin-reuse           trace duplicate co-located view rays separately\n"
        "  --mirror-illuminance-output f output 1R/2R view-plane illuminance in lux\n"
        "  -n count                    Radiance worker count (default 1)\n"
        "  -b count                    solar modifiers/batch (default 32)\n"
        "  --view-batch-size count     viewpoints held per pass (default 32)\n"
        "  -t luminance                glare threshold in cd/m^2 (default 2000)\n"
        "  --normal-tolerance degrees  normal deduplication tolerance (default 0.1)\n"
        "  --visible-fraction value    visible reflected solar-disk fraction\n"
        "  -u ux uy uz                 view up vector (default 0 0 1)\n"
        "  --oconv executable          oconv command used for reflection search\n"
        "  --rcontrib executable       external backend (default built-in CPU)\n"
        "  --rtrace executable         rtrace command\n"
        "  --rcontrib-options \"...\"  rendering options\n"
        "  -o file                     output matrix (default stdout)\n"
        "  -q                          suppress progress messages\n\n"
        "Rows are viewpoints; columns follow source records in suns.rad.\n");
}

std::string require_option_value(int &index, int argc, char *argv[],
                                 const std::string &option)
{
    if (++index >= argc)
        throw std::runtime_error("missing argument for " + option);
    return argv[index];
}


} // namespace

Options parse_options(int argc, char *argv[])
{
    Options options;
    bool adaptive_rough_sampling = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "-?" || argument == "--help") {
            usage(stdout);
            std::exit(0);
        } else if (argument == "-h" || argument == "--no-header")
            options.no_header = true;
        else if (argument == "-vf" || argument == "--views")
            options.views_path = require_option_value(index, argc, argv, argument);
        else if (argument == "-N" || argument == "--normals")
            options.normals_path = require_option_value(index, argc, argv, argument);
        else if (argument == "--normal-rad")
            options.normal_rad_path = require_option_value(index, argc, argv, argument);
        else if (argument == "--reflection-octree")
            options.reflection_octree = require_option_value(
                index, argc, argv, argument);
        else if (argument == "-S" || argument == "--suns")
            options.suns_path = require_option_value(index, argc, argv, argument);
        else if (argument == "-M" || argument == "--mirror-modifiers")
            options.mirror_modifiers_path = require_option_value(index, argc, argv, argument);
        else if (argument == "--auto-material-allowlist")
            options.auto_material_allowlist_path = require_option_value(
                index, argc, argv, argument);
        else if (argument == "--proposal-modifiers")
            options.proposal_modifiers_path = require_option_value(
                index, argc, argv, argument);
        else if (argument == "--transparent-modifiers")
            options.transparent_modifiers_path = require_option_value(
                index, argc, argv, argument);
        else if (argument == "--max-transparent-hits")
            options.max_transparent_hits = parse_int(
                require_option_value(index, argc, argv, argument),
                "maximum transparent hit count");
        else if (argument == "--sun-disk-samples")
            options.sun_disk_samples = parse_int(
                require_option_value(index, argc, argv, argument),
                "solar-disk sample count");
        else if (argument == "--secondary-sun-disk-samples")
            options.secondary_sun_disk_samples = parse_int(
                require_option_value(index, argc, argv, argument),
                "secondary solar-disk sample count");
        else if (argument == "--adaptive-sun-disk")
            options.adaptive_sun_disk = true;
        else if (argument == "--sun-disk-pilot-samples")
            options.sun_disk_pilot_samples = parse_int(
                require_option_value(index, argc, argv, argument),
                "solar-disk pilot sample count");
        else if (argument == "--secondary-sun-disk-pilot-samples")
            options.secondary_sun_disk_pilot_samples = parse_int(
                require_option_value(index, argc, argv, argument),
                "secondary solar-disk pilot sample count");
        else if (argument == "--sun-disk-pilot-guard-angle") {
            const std::string value = require_option_value(
                index, argc, argv, argument);
            if (!parse_double(value, options.sun_disk_pilot_guard_angle))
                throw std::runtime_error(
                    "invalid solar-disk pilot guard angle '" + value + "'");
        }
        else if (argument == "--sun-disk-seed")
            options.sun_disk_seed = parse_int(
                require_option_value(index, argc, argv, argument),
                "solar-disk sampling seed");
        else if (argument == "--include-direct-sun")
            options.include_direct_sun = true;
        else if (argument == "--no-hit-check")
            options.no_hit_check = true;
        else if (argument == "--integrated-path-check")
            options.integrated_path_check = true;
        else if (argument == "--separate-path-prefilter")
            options.separate_path_prefilter = true;
        else if (argument == "--nonspec-octree")
            options.nonspec_octree = require_option_value(index, argc, argv, argument);
        else if (argument == "--direct-specular-only")
            options.direct_specular_only = true;
        else if (argument == "--no-origin-reuse")
            options.origin_reuse = false;
        else if (argument == "--mirror-illuminance-output" ||
                 argument == "--illuminance-output")
            options.mirror_illuminance_output_path = require_option_value(
                index, argc, argv, argument);
        else if (argument == "-o" || argument == "--output")
            options.output_path = require_option_value(index, argc, argv, argument);
        else if (argument == "--save-normals")
            options.save_normals_path = require_option_value(index, argc, argv, argument);
        else if (argument == "--save-reflection-paths" ||
                 argument == "--save-paths")
            options.save_paths_path = require_option_value(index, argc, argv, argument);
        else if (argument == "--oconv")
            options.oconv = require_option_value(index, argc, argv, argument);
        else if (argument == "--rcontrib")
            options.rcontrib = require_option_value(index, argc, argv, argument);
        else if (argument == "--rtrace")
            options.rtrace = require_option_value(index, argc, argv, argument);
        else if (argument == "--rcontrib-options")
            options.rcontrib_options = split_words(
                require_option_value(index, argc, argv, argument));
        else if (argument == "-n" || argument == "--nproc")
            options.nproc = parse_int(require_option_value(index, argc, argv, argument),
                                      "worker count");
        else if (argument == "-b" || argument == "--batch-size")
            options.batch_size = parse_int(
                require_option_value(index, argc, argv, argument), "batch size");
        else if (argument == "--view-batch-size")
            options.view_batch_size = parse_int(
                require_option_value(index, argc, argv, argument),
                "view batch size");
        else if (argument == "--reflection-level")
            options.reflection_level = parse_int(
                require_option_value(index, argc, argv, argument),
                "reflection search level");
        else if (argument == "--allowlist-fallback-level")
            options.allowlist_fallback_level = parse_int(
                require_option_value(index, argc, argv, argument),
                "allowlist fallback search level");
        else if (argument == "--reflection-seed")
            options.reflection_seed = parse_int(
                require_option_value(index, argc, argv, argument),
                "reflection search seed");
        else if (argument == "--max-specular-bounces")
            options.max_specular_bounces = parse_int(
                require_option_value(index, argc, argv, argument),
                "maximum specular bounce count");
        else if (argument == "--all-normal-pairs")
            options.all_normal_pairs = true;
        else if (argument == "--rough-samples")
            options.rough_samples = parse_int(
                require_option_value(index, argc, argv, argument),
                "rough sample count");
        else if (argument == "--adaptive-rough-sampling")
            adaptive_rough_sampling = true;
        else if (argument == "--adaptive-rough-cells")
            options.adaptive_rough_cells = true;
        else if (argument == "--rough-pilot-samples")
            options.rough_pilot_samples = parse_int(
                require_option_value(index, argc, argv, argument),
                "rough pilot sample count");
        else if (argument == "--rough-medium-samples")
            options.rough_medium_samples = parse_int(
                require_option_value(index, argc, argv, argument),
                "rough medium sample count");
        else if (argument == "--rough-secondary-samples")
            options.rough_secondary_samples = parse_int(
                require_option_value(index, argc, argv, argument),
                "rough secondary sample count");
        else if (argument == "--rough-seed")
            options.rough_seed = parse_int(
                require_option_value(index, argc, argv, argument),
                "rough sampling seed");
        else if (argument == "--cluster-rough-sources")
            options.cluster_rough_sources = true;
        else if (argument == "--auto-materials")
            options.auto_materials = true;
        else if (argument == "--ideal-only")
            options.ideal_only = true;
        else if (argument == "--normal-samples") {
            const int samples = parse_int(
                require_option_value(index, argc, argv, argument),
                "normal sample count");
            long long side = 1;
            int level = 0;
            while (2LL*side*side < samples && level < 15) {
                side *= 2;
                ++level;
            }
            if (2LL*side*side != samples)
                throw std::runtime_error(
                    "--normal-samples must equal 2*4^level; use "
                    "--reflection-level instead");
            options.reflection_level = level;
        }
        else if (argument == "-t" || argument == "--threshold") {
            const std::string value = require_option_value(index, argc, argv, argument);
            if (!parse_double(value, options.threshold))
                throw std::runtime_error("invalid luminance threshold '" + value + "'");
        } else if (argument == "--roughness") {
            const std::string value = require_option_value(index, argc, argv, argument);
            if (!parse_double(value, options.roughness))
                throw std::runtime_error("invalid roughness '" + value + "'");
        } else if (argument == "--rough-extent") {
            const std::string value = require_option_value(index, argc, argv, argument);
            if (!parse_double(value, options.rough_extent))
                throw std::runtime_error("invalid rough extent '" + value + "'");
        } else if (argument == "--rough-prefilter" ||
                   argument == "--rough-pilot-threshold") {
            const std::string value = require_option_value(index, argc, argv, argument);
            if (!parse_double(value, options.rough_pilot_threshold))
                throw std::runtime_error("invalid rough prefilter '" + value + "'");
        } else if (argument == "--rough-pilot-guard-angle") {
            const std::string value = require_option_value(index, argc, argv, argument);
            if (!parse_double(value, options.rough_pilot_guard_angle))
                throw std::runtime_error(
                    "invalid rough pilot guard angle '" + value + "'");
        } else if (argument == "--rough-pilot-anchor-angle") {
            const std::string value = require_option_value(index, argc, argv, argument);
            if (!parse_double(value, options.rough_pilot_anchor_angle))
                throw std::runtime_error(
                    "invalid rough pilot anchor angle '" + value + "'");
        } else if (argument == "--rough-cell-trigger") {
            const std::string value = require_option_value(index, argc, argv, argument);
            if (!parse_double(value, options.rough_cell_trigger))
                throw std::runtime_error(
                    "invalid rough cell trigger '" + value + "'");
        } else if (argument == "--normal-tolerance") {
            const std::string value = require_option_value(index, argc, argv, argument);
            if (!parse_double(value, options.normal_tolerance))
                throw std::runtime_error("invalid normal tolerance '" + value + "'");
        } else if (argument == "--visible-fraction") {
            const std::string value = require_option_value(index, argc, argv, argument);
            if (!parse_double(value, options.visible_fraction))
                throw std::runtime_error("invalid visible fraction '" + value + "'");
        } else if (argument == "-u" || argument == "--up") {
            for (int component = 0; component < 3; ++component) {
                const std::string value =
                    require_option_value(index, argc, argv, argument);
                if (!parse_double(value, options.up[component]))
                    throw std::runtime_error("invalid view up component '" + value + "'");
            }
        } else if (argument == "-q" || argument == "--quiet")
            options.quiet = true;
        else if (!argument.empty() && argument[0] == '-')
            throw std::runtime_error("unknown option '" + argument + "'");
        else if (!options.octree.empty())
            throw std::runtime_error("unexpected extra argument '" + argument + "'");
        else
            options.octree = argument;
    }
    if (options.ideal_only)
        options.auto_materials = true;
    if (options.auto_materials) {
        options.integrated_path_check = true;
        options.direct_specular_only = true;
        options.cluster_rough_sources = !options.ideal_only;
        if (!options.ideal_only && options.rough_samples == 0)
            options.rough_samples = 4096;
    } else if (!options.separate_path_prefilter && options.rcontrib.empty() &&
            !options.mirror_modifiers_path.empty())
        options.integrated_path_check = true;
    if ((adaptive_rough_sampling || options.adaptive_rough_cells) &&
            options.rough_pilot_threshold <= 0.0)
        options.rough_pilot_threshold = 1.0e-6;
    return options;
}

void validate_options(const Options &options)
{
    if (options.views_path.empty() || options.suns_path.empty() ||
            options.octree.empty())
        throw std::runtime_error("-vf, -S, and the scene octree are required");
    if (!options.normals_path.empty() && !options.normal_rad_path.empty())
        throw std::runtime_error("-N and --normal-rad are mutually exclusive");
    if (options.nproc < 1 || options.batch_size < 1 ||
            options.view_batch_size < 1)
        throw std::runtime_error(
            "worker, solar-batch, and view-batch counts must be positive");
    if (options.reflection_level < 0 || options.reflection_level > 10)
        throw std::runtime_error("reflection search level must lie in [0, 10]");
    if (options.allowlist_fallback_level < -1 ||
            options.allowlist_fallback_level > 10)
        throw std::runtime_error(
            "allowlist fallback search level must lie in [0, 10]");
    if (options.allowlist_fallback_level >= 0 &&
            options.allowlist_fallback_level <= options.reflection_level)
        throw std::runtime_error(
            "allowlist fallback search level must exceed --reflection-level");
    if (options.max_specular_bounces < 1 ||
            options.max_specular_bounces > 2)
        throw std::runtime_error(
            "maximum specular bounce count must be 1 or 2");
    if (options.all_normal_pairs && options.max_specular_bounces < 2)
        throw std::runtime_error(
            "--all-normal-pairs requires --max-specular-bounces 2");
    if (options.max_transparent_hits < 1 || options.max_transparent_hits > 64)
        throw std::runtime_error(
            "maximum transparent hit count must lie in [1, 64]");
    if (options.sun_disk_samples < 1)
        throw std::runtime_error("solar-disk sample count must be positive");
    {
        const int side = static_cast<int>(std::sqrt(
            static_cast<double>(options.sun_disk_samples)));
        if (side*side != options.sun_disk_samples)
            throw std::runtime_error(
                "solar-disk sample count must be a perfect square "
                "(e.g. 4, 16, or 64)");
    }
    if (options.secondary_sun_disk_samples < 0)
        throw std::runtime_error(
            "secondary solar-disk sample count cannot be negative");
    if (options.secondary_sun_disk_samples > 0) {
        const int side = static_cast<int>(std::sqrt(
            static_cast<double>(options.secondary_sun_disk_samples)));
        if (side*side != options.secondary_sun_disk_samples)
            throw std::runtime_error(
                "secondary solar-disk sample count must be a perfect square");
    }
    if (options.sun_disk_pilot_samples < 1 ||
            options.secondary_sun_disk_pilot_samples < 1)
        throw std::runtime_error(
            "solar-disk pilot sample counts must be positive");
    {
        const int primary_side = static_cast<int>(std::sqrt(
            static_cast<double>(options.sun_disk_pilot_samples)));
        const int secondary_side = static_cast<int>(std::sqrt(
            static_cast<double>(options.secondary_sun_disk_pilot_samples)));
        if (primary_side*primary_side != options.sun_disk_pilot_samples ||
                secondary_side*secondary_side !=
                options.secondary_sun_disk_pilot_samples)
            throw std::runtime_error(
                "solar-disk pilot sample counts must be perfect squares");
    }
    if (options.adaptive_sun_disk) {
        const int secondary_samples =
            options.secondary_sun_disk_samples > 0 ?
            options.secondary_sun_disk_samples : options.sun_disk_samples;
        if (options.sun_disk_samples <= 1 ||
                options.sun_disk_pilot_samples > options.sun_disk_samples ||
                options.secondary_sun_disk_pilot_samples >
                secondary_samples)
            throw std::runtime_error(
                "adaptive solar-disk pilot counts must not exceed the full "
                "sample counts");
        if (!options.rcontrib.empty() || !options.integrated_path_check)
            throw std::runtime_error(
                "--adaptive-sun-disk requires the built-in backend and "
                "integrated path checking with -M");
    }
    if (!(options.sun_disk_pilot_guard_angle >= 0.0 &&
            options.sun_disk_pilot_guard_angle < 180.0))
        throw std::runtime_error(
            "solar-disk pilot guard angle must lie in [0, 180) degrees");
    if (options.rough_samples < 0)
        throw std::runtime_error("rough sample count cannot be negative");
    if (options.rough_pilot_samples < 1)
        throw std::runtime_error("rough pilot sample count must be positive");
    if (options.rough_secondary_samples < 1)
        throw std::runtime_error(
            "rough secondary sample count must be positive");
    if (options.rough_samples > 0) {
        const int side = static_cast<int>(std::sqrt(
            static_cast<double>(options.rough_samples)));
        if (side*side != options.rough_samples)
            throw std::runtime_error(
                "rough sample count must be a perfect square (e.g. 16 or 64)");
        const int pilot_side = static_cast<int>(std::sqrt(
            static_cast<double>(options.rough_pilot_samples)));
        if (pilot_side*pilot_side != options.rough_pilot_samples ||
                options.rough_pilot_samples > options.rough_samples)
            throw std::runtime_error(
                "rough pilot sample count must be a perfect square no "
                "larger than the full rough sample count");
        if (options.adaptive_rough_cells) {
            const int medium_side = static_cast<int>(std::sqrt(
                static_cast<double>(options.rough_medium_samples)));
            if (medium_side*medium_side != options.rough_medium_samples ||
                    options.rough_medium_samples >= options.rough_samples ||
                    side%medium_side)
                throw std::runtime_error(
                    "adaptive rough-cell sampling requires a perfect-square "
                    "--rough-medium-samples grid that evenly divides the "
                    "final grid");
        }
        if (!options.auto_materials &&
                !(options.roughness > 0.0 && options.roughness <= 1.0))
            throw std::runtime_error(
                "roughness must lie in (0, 1] when rough sampling is enabled");
    } else if (!options.auto_materials && options.roughness != 0.0) {
        throw std::runtime_error(
            "--roughness requires a positive --rough-samples value");
    }
    if (options.cluster_rough_sources && options.rough_samples <= 0)
        throw std::runtime_error(
            "--cluster-rough-sources requires rough sampling to be enabled");
    if (!(options.rough_cell_trigger > 0.0 &&
            options.rough_cell_trigger <= 1.0))
        throw std::runtime_error(
            "rough cell trigger must lie in (0, 1]");
    if (options.adaptive_rough_cells &&
            (!options.rcontrib.empty() || !options.nonspec_octree.empty()))
        throw std::runtime_error(
            "adaptive rough-cell integration requires the built-in backend "
            "without --nonspec-octree");
    if (options.rough_pilot_guard_angle < 0.0 ||
            options.rough_pilot_guard_angle >= 180.0)
        throw std::runtime_error(
            "rough pilot guard angle must lie in [0, 180) degrees");
    if (!(options.rough_pilot_anchor_angle > 0.0 &&
            options.rough_pilot_anchor_angle < 180.0))
        throw std::runtime_error(
            "rough pilot anchor angle must lie in (0, 180) degrees");
    if (!options.auto_materials && options.sun_disk_samples > 1 &&
            options.rough_samples > 0)
        throw std::runtime_error(
            "solar-disk and rough-lobe sampling cannot be enabled together");
    if (!options.auto_materials && options.include_direct_sun &&
            options.rough_samples > 0)
        throw std::runtime_error(
            "direct-sun and rough-lobe sampling cannot be enabled together");
    if (options.direct_specular_only && !options.nonspec_octree.empty())
        throw std::runtime_error(
            "--direct-specular-only and --nonspec-octree are mutually exclusive");
    if (options.direct_specular_only && !options.rcontrib.empty())
        throw std::runtime_error(
            "--direct-specular-only requires the built-in rcontrib backend");
    if (!options.output_path.empty() &&
            options.output_path == options.mirror_illuminance_output_path)
        throw std::runtime_error(
            "contrast and mirror illuminance outputs must use different files");
    if (options.integrated_path_check && options.separate_path_prefilter)
        throw std::runtime_error(
            "--integrated-path-check and --separate-path-prefilter are mutually exclusive");
    if (options.integrated_path_check &&
            ((!options.auto_materials && options.mirror_modifiers_path.empty()) ||
             !options.rcontrib.empty()))
        throw std::runtime_error(
            "--integrated-path-check requires -M (or --auto-materials) and the built-in backend");
    if (options.auto_materials && !options.mirror_modifiers_path.empty())
        throw std::runtime_error(
            "--auto-materials and -M are mutually exclusive");
    if (!options.auto_material_allowlist_path.empty() &&
            !options.auto_materials)
        throw std::runtime_error(
            "--auto-material-allowlist requires --auto-materials");
    if (options.allowlist_fallback_level >= 0 &&
            options.auto_material_allowlist_path.empty())
        throw std::runtime_error(
            "--allowlist-fallback-level requires --auto-material-allowlist");
    if (options.auto_materials && (!options.normals_path.empty() ||
            !options.normal_rad_path.empty() || options.all_normal_pairs))
        throw std::runtime_error(
            "--auto-materials requires automatic path-tree discovery");
    if (options.auto_materials && !options.proposal_modifiers_path.empty())
        throw std::runtime_error(
            "--auto-materials determines proposal modifiers from the octree");
    if (options.auto_materials && !options.nonspec_octree.empty())
        throw std::runtime_error(
            "--auto-materials uses direct specular path filtering and does not accept --nonspec-octree");
    if (options.auto_materials && options.roughness != 0.0)
        throw std::runtime_error(
            "--roughness is determined from each material in --auto-materials mode");
    if (options.ideal_only && options.rough_samples != 0)
        throw std::runtime_error(
            "--ideal-only does not accept rough-lobe sampling options");
    if (!(options.rough_extent > 0.0 && options.rough_extent <= 10.0))
        throw std::runtime_error("rough extent must lie in (0, 10]");
    if (options.rough_pilot_threshold < 0.0 ||
            options.rough_pilot_threshold > 1.0)
        throw std::runtime_error("rough prefilter must lie in [0, 1]");
    if (options.rough_pilot_threshold > 0.0 && options.rough_samples <= 0)
        throw std::runtime_error(
            "adaptive rough sampling requires rough sampling to be enabled");
    if (options.visible_fraction < 0.0 || options.visible_fraction > 1.0)
        throw std::runtime_error("visible fraction must lie between 0 and 1");
    if (options.normal_tolerance < 0.0 || options.normal_tolerance >= 90.0)
        throw std::runtime_error("normal tolerance must lie in [0, 90) degrees");
    require_file(options.views_path, "viewpoint file");
    require_file(options.suns_path, "suns file");
    require_file(options.octree, "octree");
    if (!options.normals_path.empty())
        require_file(options.normals_path, "normal file");
    if (!options.normal_rad_path.empty())
        require_file(options.normal_rad_path, "normal geometry file");
    if (!options.reflection_octree.empty())
        require_file(options.reflection_octree, "reflection-search octree");
    if (!options.mirror_modifiers_path.empty())
        require_file(options.mirror_modifiers_path, "mirror modifier file");
    if (!options.auto_material_allowlist_path.empty())
        require_file(options.auto_material_allowlist_path,
                     "automatic material allowlist file");
    if (!options.proposal_modifiers_path.empty())
        require_file(options.proposal_modifiers_path,
                     "path-tree proposal modifier file");
    if (!options.integrated_path_check &&
            !options.transparent_modifiers_path.empty())
        require_file(options.transparent_modifiers_path,
                     "transparent modifier file");
    if (!options.nonspec_octree.empty())
        require_file(options.nonspec_octree, "non-specular octree");
#ifndef SPECULARCONTRIB_BUILTIN_RCONTRIB
    if (options.rcontrib.empty())
        throw std::runtime_error(
            "this build has no built-in rcontrib; specify --rcontrib");
#endif
    validate_rcontrib_options(options.rcontrib_options);
}

} // namespace specularcontrast
