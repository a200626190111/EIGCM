#ifndef lint
static const char RCSid[] = "$Id$";
#endif

/*
 * Compute first- and second-order specular DGP contrast terms using targeted
 * rcontrib rays.  Ideal reflection is the default; optional equal-solid-angle
 * quadrature integrates a finite rough-specular lobe.  The same samples may
 * also be projected onto the view direction to produce mirror illuminance.
 * Output rows are viewpoints and columns are solar records.
 */

#include "platform.h"
#include "paths.h"
#include "rtprocess.h"
#include "rtio.h"
#include "resolu.h"
#include "standard.h"

#ifdef SPECULARCONTRIB_BUILTIN_RCONTRIB
#include "specular_rcontrib_backend.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <signal.h>
#endif

namespace {

const double kLuminousEfficacy = 179.0;
const double kBrightness[3] = {0.265074, 0.670115, 0.064811};
const double kEpsilon = 1.0e-12;

typedef std::array<double, 3> Vec3;

struct Viewpoint {
    Vec3 origin;
    Vec3 direction;
};

struct Sun {
    std::string modifier;
    Vec3 direction;
    Vec3 radiance;
    double omega;
    double radiance_scale;
    bool has_radiance;
    bool active;
};

struct Candidate {
    std::size_t time_index;
    std::size_t local_modifier;
    std::size_t view_index;
    std::size_t seed_index;
    std::size_t disk_seed_index;
    unsigned int reflection_orders;
    bool direct_sun;
    Vec3 direction;
    double position_index;
    double solid_angle;
    int rough_row = -1;
    int rough_column = -1;
    int rough_side = 0;
};

typedef std::tuple<std::size_t, std::size_t, std::size_t, unsigned int>
    RoughSourceKey;

struct RoughSourceAccumulator {
    double solid_angle = 0.0;
    double luminance_solid_angle = 0.0;
    Vec3 direction_moment = {{0.0, 0.0, 0.0}};
};

const std::size_t kAllViews = std::numeric_limits<std::size_t>::max();

struct NormalSource {
    std::size_t view_index = kAllViews;
    std::string modifier;
    std::string material;
    std::string primitive;
};

struct ReflectionPath2 {
    std::size_t view_index;
    Vec3 first_normal;
    Vec3 second_normal;
    std::string first_material;
    std::string second_material;
    std::string first_modifier;
    std::string second_modifier;
    std::string first_primitive;
    std::string second_primitive;
};

struct ReflectionData {
    std::vector<Vec3> first_order_normals;
    std::vector<std::vector<NormalSource> > first_order_sources;
    std::vector<ReflectionPath2> second_order_paths;
};

struct WorkChunk {
    std::vector<std::pair<std::size_t, Sun> > suns;
    std::vector<Candidate> candidates;
    std::size_t generated;
};

struct BuiltinBackendHolder {
#ifdef SPECULARCONTRIB_BUILTIN_RCONTRIB
    std::unique_ptr<SpecularRcontribBackend> backend;
#endif
};

struct Primitive {
    std::string modifier;
    std::string type;
    std::string identifier;
    std::vector<double> real_args;
};

struct Options {
    bool no_header = false;
    bool no_hit_check = false;
    bool integrated_path_check = false;
    bool separate_path_prefilter = false;
    bool include_direct_sun = false;
    bool all_normal_pairs = false;
    bool direct_specular_only = false;
    bool adaptive_sun_sampling = false;
    bool adaptive_sun_disk = false;
    bool adaptive_rough_sampling = false;
    bool adaptive_rough_integration = false;
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
    int rough_coarse_samples = 256;
    int rough_medium_samples = 1024;
    int rough_secondary_samples = 16;
    int rough_seed = 0;
    double threshold = 2000.0;
    double normal_tolerance = 0.1;
    double visible_fraction = 1.0;
    double roughness = 0.0;
    double rough_extent = 3.0;
    double rough_prefilter = 0.0;
    double rough_pilot_guard_angle = 0.0;
    double rough_pilot_anchor_angle = 5.0;
    double rough_convergence = 0.05;
    double rough_illuminance_tolerance = 1.0;
    double rough_cell_trigger = 0.5;
    double adaptive_sun_max_angle = 1.0;
    double adaptive_sun_min_angle = 0.1;
    double adaptive_sun_error = 0.0;
    double sun_disk_pilot_guard_angle = 8.0;
    Vec3 up = {{0.0, 0.0, 1.0}};
};

double dot(const Vec3 &a, const Vec3 &b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

Vec3 cross(const Vec3 &a, const Vec3 &b)
{
    return Vec3{{a[1]*b[2] - a[2]*b[1],
                 a[2]*b[0] - a[0]*b[2],
                 a[0]*b[1] - a[1]*b[0]}};
}

Vec3 add_scaled(const Vec3 &a, const Vec3 &b, double scale)
{
    return Vec3{{a[0] + scale*b[0], a[1] + scale*b[1],
                 a[2] + scale*b[2]}};
}

double norm(const Vec3 &v)
{
    return std::sqrt(dot(v, v));
}

Vec3 normalized(const Vec3 &v, const std::string &what)
{
    const double length = norm(v);
    if (length <= kEpsilon)
        throw std::runtime_error("zero-length " + what);
    return Vec3{{v[0]/length, v[1]/length, v[2]/length}};
}

double angle(const Vec3 &a, const Vec3 &b)
{
    return std::acos(std::max(-1.0, std::min(1.0, dot(a, b))));
}

std::string trim_comment(const std::string &line)
{
    const std::string::size_type hash = line.find('#');
    const std::string text = line.substr(0, hash);
    const std::string::size_type first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return std::string();
    const std::string::size_type last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last-first+1);
}

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

std::vector<double> numeric_tokens(const std::string &line)
{
    std::vector<double> values;
    std::istringstream input(line);
    std::string word;
    while (input >> word) {
        double value;
        if (parse_double(word, value))
            values.push_back(value);
    }
    return values;
}

void require_file(const std::string &path, const std::string &label)
{
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input)
        throw std::runtime_error(label + " '" + path + "' cannot be opened");
}

std::vector<Viewpoint> load_views(const std::string &path)
{
    std::ifstream input(path.c_str());
    if (!input)
        throw std::runtime_error("cannot open viewpoint file '" + path + "'");
    std::vector<Viewpoint> views;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = trim_comment(line);
        if (line.empty())
            continue;
        const std::vector<double> values = numeric_tokens(line);
        if (values.size() != 6) {
            std::ostringstream message;
            message << path << ':' << line_number
                    << ": expected x y z dx dy dz";
            throw std::runtime_error(message.str());
        }
        Viewpoint view;
        view.origin = Vec3{{values[0], values[1], values[2]}};
        view.direction = normalized(Vec3{{values[3], values[4], values[5]}},
                                    "view direction");
        views.push_back(view);
    }
    if (views.empty())
        throw std::runtime_error("no viewpoints found in '" + path + "'");
    return views;
}

Vec3 canonical_normal(const Vec3 &candidate)
{
    Vec3 normal = normalized(candidate, "mirror normal");
    for (int component = 0; component < 3; ++component) {
        if (std::fabs(normal[component]) <= kEpsilon)
            continue;
        if (normal[component] < 0.0) {
            normal[0] = -normal[0];
            normal[1] = -normal[1];
            normal[2] = -normal[2];
        }
        break;
    }
    return normal;
}

std::size_t append_unique_normal(std::vector<Vec3> &normals,
                                 const Vec3 &candidate,
                                 double cosine_tolerance)
{
    const Vec3 normal = canonical_normal(candidate);
    for (std::size_t i = 0; i < normals.size(); ++i)
        if (std::fabs(dot(normal, normals[i])) >= cosine_tolerance)
            return i;
    normals.push_back(normal);
    return normals.size()-1;
}

bool same_normal_source(const NormalSource &first, const NormalSource &second)
{
    return first.view_index == second.view_index &&
        first.modifier == second.modifier &&
        first.material == second.material &&
        first.primitive == second.primitive;
}

void append_sourced_normal(ReflectionData &reflections, const Vec3 &candidate,
                           const NormalSource *source,
                           double cosine_tolerance)
{
    const std::size_t index = append_unique_normal(
        reflections.first_order_normals, candidate, cosine_tolerance);
    if (reflections.first_order_sources.size() <
            reflections.first_order_normals.size())
        reflections.first_order_sources.resize(
            reflections.first_order_normals.size());
    if (!source)
        return;
    std::vector<NormalSource> &sources =
        reflections.first_order_sources[index];
    for (std::size_t i = 0; i < sources.size(); ++i)
        if (same_normal_source(sources[i], *source))
            return;
    sources.push_back(*source);
}

ReflectionData load_normals(const std::string &path,
                            double tolerance_degrees)
{
    std::ifstream input(path.c_str());
    if (!input)
        throw std::runtime_error("cannot open normal file '" + path + "'");
    ReflectionData reflections;
    const double cosine_tolerance =
        std::cos(tolerance_degrees*PI/180.0);
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::size_t comment_position = line.find('#');
        const std::string coordinates = comment_position == std::string::npos ?
            line : line.substr(0, comment_position);
        if (trim_comment(coordinates).empty())
            continue;
        const std::vector<double> values = numeric_tokens(coordinates);
        Vec3 normal;
        if (values.size() == 3)
            normal = Vec3{{values[0], values[1], values[2]}};
        else if (values.size() >= 6)
            normal = Vec3{{values[values.size()-3], values[values.size()-2],
                           values[values.size()-1]}};
        else {
            std::ostringstream message;
            message << path << ':' << line_number
                    << ": expected nx ny nz, optionally after a hit point";
            throw std::runtime_error(message.str());
        }
        NormalSource source;
        bool have_source = false;
        if (comment_position != std::string::npos) {
            std::istringstream metadata(line.substr(comment_position+1));
            std::string field;
            while (metadata >> field) {
                const std::size_t equals = field.find('=');
                if (equals == std::string::npos)
                    continue;
                const std::string key = field.substr(0, equals);
                const std::string value = field.substr(equals+1);
                if (key == "view") {
                    if (value == "*")
                        source.view_index = kAllViews;
                    else {
                        char *end = NULL;
                        errno = 0;
                        const unsigned long long parsed = std::strtoull(
                            value.c_str(), &end, 10);
                        if (end == value.c_str() || *end != '\0' ||
                                errno == ERANGE) {
                            std::ostringstream message;
                            message << path << ':' << line_number
                                    << ": invalid normal-source view index";
                            throw std::runtime_error(message.str());
                        }
                        source.view_index = static_cast<std::size_t>(parsed);
                    }
                    have_source = true;
                } else if (key == "modifier") {
                    source.modifier = value == "-" ? "" : value;
                    have_source = true;
                } else if (key == "material") {
                    source.material = value == "-" ? "" : value;
                    have_source = true;
                } else if (key == "primitive") {
                    source.primitive = value == "-" ? "" : value;
                    have_source = true;
                }
            }
        }
        append_sourced_normal(reflections, normal,
                              have_source ? &source : NULL,
                              cosine_tolerance);
    }
    if (reflections.first_order_normals.empty())
        throw std::runtime_error("no mirror normals found in '" + path + "'");
    return reflections;
}

std::set<std::string> load_modifier_names(const std::string &path,
                                          const std::string &description)
{
    std::ifstream input(path.c_str());
    if (!input)
        throw std::runtime_error("cannot open " + description + " file '" +
                                 path + "'");
    std::set<std::string> names;
    std::string line;
    while (std::getline(input, line)) {
        line = trim_comment(line);
        std::istringstream words(line);
        std::string word;
        while (words >> word)
            names.insert(word);
    }
    if (names.empty())
        throw std::runtime_error("no " + description + " names found in '" +
                                 path + "'");
    return names;
}

std::vector<std::string> radiance_tokens(const std::string &path)
{
    std::ifstream input(path.c_str());
    if (!input)
        throw std::runtime_error("cannot open Radiance file '" + path + "'");
    std::vector<std::string> tokens;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = trim_comment(line);
        if (line.empty())
            continue;
        if (line[0] == '!') {
            std::ostringstream message;
            message << path << ':' << line_number
                    << ": command expansion is not supported";
            throw std::runtime_error(message.str());
        }
        std::istringstream words(line);
        std::string word;
        while (words >> word)
            tokens.push_back(word);
    }
    return tokens;
}

std::vector<std::string> take_counted(const std::vector<std::string> &tokens,
                                      std::size_t &index,
                                      const std::string &kind)
{
    if (index >= tokens.size())
        throw std::runtime_error("unexpected end before " + kind + " argument count");
    const int count = parse_int(tokens[index++], kind + " argument count");
    if (index + static_cast<std::size_t>(count) > tokens.size())
        throw std::runtime_error("invalid " + kind + " argument list");
    std::vector<std::string> result(tokens.begin()+index,
                                    tokens.begin()+index+count);
    index += count;
    return result;
}

std::vector<Primitive> load_primitives(const std::string &path)
{
    const std::vector<std::string> tokens = radiance_tokens(path);
    std::vector<Primitive> primitives;
    std::size_t index = 0;
    while (index < tokens.size()) {
        if (index + 3 > tokens.size())
            throw std::runtime_error("incomplete Radiance primitive in '" + path + "'");
        Primitive primitive;
        primitive.modifier = tokens[index++];
        primitive.type = tokens[index++];
        primitive.identifier = tokens[index++];
        take_counted(tokens, index, "string");
        take_counted(tokens, index, "integer");
        const std::vector<std::string> real_args =
            take_counted(tokens, index, "real");
        for (std::size_t i = 0; i < real_args.size(); ++i) {
            double value;
            if (!parse_double(real_args[i], value))
                throw std::runtime_error("non-numeric real argument in '" + path + "'");
            primitive.real_args.push_back(value);
        }
        primitives.push_back(primitive);
    }
    return primitives;
}

std::vector<Sun> load_suns(const std::string &path)
{
    const std::vector<Primitive> primitives = load_primitives(path);
    std::map<std::string, Vec3> lights;
    struct SourceRecord {
        std::string modifier;
        Vec3 direction;
        double angle;
    };
    std::vector<SourceRecord> records;
    for (std::size_t i = 0; i < primitives.size(); ++i) {
        const Primitive &primitive = primitives[i];
        if (primitive.type == "light" && primitive.real_args.size() >= 3) {
            lights[primitive.identifier] = Vec3{{primitive.real_args[0],
                                                 primitive.real_args[1],
                                                 primitive.real_args[2]}};
        } else if (primitive.type == "source") {
            if (primitive.real_args.size() < 4)
                throw std::runtime_error("solar source has fewer than four real arguments");
            SourceRecord record;
            record.modifier = primitive.modifier;
            record.direction = normalized(
                Vec3{{primitive.real_args[0], primitive.real_args[1],
                      primitive.real_args[2]}}, "solar source direction");
            record.angle = primitive.real_args[3]*PI/180.0;
            if (record.angle <= 0.0)
                throw std::runtime_error("solar source has a non-positive angle");
            records.push_back(record);
        }
    }
    if (records.empty())
        throw std::runtime_error("no source primitives found in '" + path + "'");
    std::set<std::string> seen;
    std::vector<Sun> suns;
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (!seen.insert(records[i].modifier).second)
            throw std::runtime_error("solar source modifiers must be unique");
        Sun sun;
        sun.modifier = records[i].modifier;
        sun.direction = records[i].direction;
        sun.omega = 2.0*PI*(1.0-std::cos(0.5*records[i].angle));
        const std::map<std::string, Vec3>::const_iterator light =
            lights.find(sun.modifier);
        sun.has_radiance = light != lights.end();
        sun.radiance = sun.has_radiance ? light->second :
            Vec3{{1.0, 1.0, 1.0}};
        sun.radiance_scale = kBrightness[0]*sun.radiance[0] +
            kBrightness[1]*sun.radiance[1] +
            kBrightness[2]*sun.radiance[2];
        sun.active = !sun.has_radiance ||
            std::max(sun.radiance[0],
                     std::max(sun.radiance[1], sun.radiance[2])) > 0.0;
        suns.push_back(sun);
    }
    return suns;
}

void validate_adaptive_sun_radiance(const std::vector<Sun> &suns)
{
    Vec3 reference = {{0.0, 0.0, 0.0}};
    bool have_reference = false;
    for (std::size_t i = 0; i < suns.size(); ++i) {
        const Sun &sun = suns[i];
        if (!sun.active)
            continue;
        if (!sun.has_radiance)
            throw std::runtime_error(
                "adaptive solar sampling requires a light primitive for " +
                sun.modifier);
        if (!(sun.radiance_scale > kEpsilon) ||
                !std::isfinite(sun.radiance_scale))
            throw std::runtime_error(
                "adaptive solar sampling requires positive finite solar "
                "radiance for " + sun.modifier);
        Vec3 chromaticity;
        for (int component = 0; component < 3; ++component) {
            if (sun.radiance[component] < 0.0 ||
                    !std::isfinite(sun.radiance[component]))
                throw std::runtime_error(
                    "adaptive solar sampling requires non-negative finite "
                    "solar RGB values");
            chromaticity[component] =
                sun.radiance[component]/sun.radiance_scale;
        }
        if (!have_reference) {
            reference = chromaticity;
            have_reference = true;
            continue;
        }
        for (int component = 0; component < 3; ++component)
            if (std::fabs(chromaticity[component]-reference[component]) >
                    1.0e-4)
                throw std::runtime_error(
                    "adaptive solar sampling requires constant solar "
                    "chromaticity across active records");
    }
}

struct AdaptiveSunNode {
    std::vector<std::size_t> members;
    std::size_t endpoint0;
    std::size_t endpoint1;
    std::size_t center;
    double diameter;
    int child0;
    int child1;
};

struct AdaptiveSunTree {
    std::vector<AdaptiveSunNode> nodes;
    int root;
};

std::size_t farthest_sun(const std::vector<std::size_t> &members,
                         std::size_t origin,
                         const std::vector<Sun> &suns)
{
    std::size_t farthest = members.front();
    double smallest_dot = 2.0;
    for (std::size_t i = 0; i < members.size(); ++i) {
        const double cosine = dot(suns[origin].direction,
                                  suns[members[i]].direction);
        if (cosine < smallest_dot) {
            smallest_dot = cosine;
            farthest = members[i];
        }
    }
    return farthest;
}

std::size_t central_sun(const std::vector<std::size_t> &members,
                        const std::vector<Sun> &suns)
{
    Vec3 mean = {{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < members.size(); ++i)
        for (int component = 0; component < 3; ++component)
            mean[component] += suns[members[i]].direction[component];
    if (norm(mean) <= kEpsilon)
        return members.front();
    mean = normalized(mean, "adaptive solar cluster mean");
    std::size_t center = members.front();
    double largest_dot = -2.0;
    for (std::size_t i = 0; i < members.size(); ++i) {
        const double cosine = dot(mean, suns[members[i]].direction);
        if (cosine > largest_dot) {
            largest_dot = cosine;
            center = members[i];
        }
    }
    return center;
}

int build_adaptive_sun_node(const std::vector<std::size_t> &members,
                            const std::vector<Sun> &suns,
                            std::vector<AdaptiveSunNode> &nodes)
{
    AdaptiveSunNode node;
    node.members = members;
    node.center = central_sun(members, suns);
    node.endpoint0 = farthest_sun(members, node.center, suns);
    node.endpoint1 = farthest_sun(members, node.endpoint0, suns);
    double radius = 0.0;
    for (std::size_t i = 0; i < members.size(); ++i)
        radius = std::max(radius, angle(suns[node.center].direction,
                                        suns[members[i]].direction));
    node.diameter = std::min(PI, std::max(
        angle(suns[node.endpoint0].direction,
              suns[node.endpoint1].direction), 2.0*radius));
    node.child0 = node.child1 = -1;
    const int node_index = static_cast<int>(nodes.size());
    nodes.push_back(node);
    if (members.size() <= 1)
        return node_index;

    std::vector<std::size_t> group0;
    std::vector<std::size_t> group1;
    group0.reserve((members.size()+1)/2);
    group1.reserve(members.size()/2);
    for (std::size_t i = 0; i < members.size(); ++i) {
        const double cosine0 = dot(suns[members[i]].direction,
                                   suns[node.endpoint0].direction);
        const double cosine1 = dot(suns[members[i]].direction,
                                   suns[node.endpoint1].direction);
        (cosine0 >= cosine1 ? group0 : group1).push_back(members[i]);
    }
    if (group0.empty() || group1.empty()) {
        group0.assign(members.begin(),
                      members.begin()+members.size()/2);
        group1.assign(members.begin()+members.size()/2, members.end());
    }
    const int child0 = build_adaptive_sun_node(group0, suns, nodes);
    const int child1 = build_adaptive_sun_node(group1, suns, nodes);
    nodes[node_index].child0 = child0;
    nodes[node_index].child1 = child1;
    return node_index;
}

AdaptiveSunTree build_adaptive_sun_tree(
    const std::vector<std::size_t> &active_suns,
    const std::vector<Sun> &suns)
{
    if (active_suns.empty())
        throw std::runtime_error(
            "adaptive solar sampling has no active solar records");
    AdaptiveSunTree tree;
    tree.nodes.reserve(2*active_suns.size());
    tree.root = build_adaptive_sun_node(active_suns, suns, tree.nodes);
    return tree;
}

void append_initial_sun_nodes(const AdaptiveSunTree &tree, int node_index,
                              double maximum_angle,
                              std::vector<int> &frontier)
{
    const AdaptiveSunNode &node = tree.nodes[node_index];
    if (node.diameter <= maximum_angle || node.child0 < 0) {
        frontier.push_back(node_index);
        return;
    }
    append_initial_sun_nodes(tree, node.child0, maximum_angle, frontier);
    append_initial_sun_nodes(tree, node.child1, maximum_angle, frontier);
}

void append_sun_probe(std::vector<std::size_t> &probes,
                      std::size_t candidate)
{
    if (std::find(probes.begin(), probes.end(), candidate) == probes.end())
        probes.push_back(candidate);
}

std::vector<std::size_t> sun_node_probes(const AdaptiveSunNode &node)
{
    std::vector<std::size_t> probes;
    append_sun_probe(probes, node.endpoint0);
    append_sun_probe(probes, node.center);
    append_sun_probe(probes, node.endpoint1);
    return probes;
}

double normalized_sun_response(double contrast, const Sun &sun)
{
    const double scale = sun.radiance_scale*sun.radiance_scale*sun.omega;
    return scale > kEpsilon ? contrast/scale : 0.0;
}

double normalized_sun_illuminance(double illuminance, const Sun &sun)
{
    const double scale = sun.radiance_scale*sun.omega;
    return scale > kEpsilon ? illuminance/scale : 0.0;
}

bool response_range_is_smooth(double minimum, double maximum,
                              double tolerance)
{
    return maximum <= kEpsilon ||
        (tolerance > 0.0 && minimum > kEpsilon &&
         (maximum-minimum)/maximum <= tolerance);
}

bool sun_node_response_is_smooth(
    const AdaptiveSunNode &node, const std::vector<Sun> &suns,
    const std::vector<double> &contrast, std::size_t nsteps,
    std::size_t first_view, std::size_t last_view,
    const std::vector<double> *illuminance, double tolerance)
{
    const std::vector<std::size_t> probes = sun_node_probes(node);
    for (std::size_t view = first_view; view < last_view; ++view) {
        double contrast_minimum = std::numeric_limits<double>::infinity();
        double contrast_maximum = 0.0;
        double illuminance_minimum = std::numeric_limits<double>::infinity();
        double illuminance_maximum = 0.0;
        for (std::size_t i = 0; i < probes.size(); ++i) {
            const std::size_t time = probes[i];
            const double response = normalized_sun_response(
                contrast[view*nsteps+time], suns[time]);
            contrast_minimum = std::min(contrast_minimum, response);
            contrast_maximum = std::max(contrast_maximum, response);
            if (illuminance) {
                const double illuminance_response =
                    normalized_sun_illuminance(
                        (*illuminance)[view*nsteps+time], suns[time]);
                illuminance_minimum = std::min(
                    illuminance_minimum, illuminance_response);
                illuminance_maximum = std::max(
                    illuminance_maximum, illuminance_response);
            }
        }
        if (!response_range_is_smooth(
                contrast_minimum, contrast_maximum, tolerance))
            return false;
        if (illuminance && !response_range_is_smooth(
                illuminance_minimum, illuminance_maximum, tolerance))
            return false;
    }
    return true;
}

void interpolate_sun_node(
    const AdaptiveSunNode &node, const std::vector<Sun> &suns,
    const std::vector<char> &sampled, std::size_t first_view,
    std::size_t last_view, std::size_t nsteps,
    std::vector<double> &contrast, std::vector<double> *illuminance)
{
    std::vector<std::size_t> available;
    for (std::size_t i = 0; i < node.members.size(); ++i)
        if (sampled[node.members[i]])
            available.push_back(node.members[i]);
    if (available.empty())
        throw std::runtime_error(
            "adaptive solar leaf has no evaluated direction");

    for (std::size_t member = 0; member < node.members.size(); ++member) {
        const std::size_t time = node.members[member];
        if (sampled[time])
            continue;
        std::vector<std::pair<double, std::size_t> > nearest;
        nearest.reserve(available.size());
        for (std::size_t i = 0; i < available.size(); ++i)
            nearest.push_back(std::make_pair(
                angle(suns[time].direction,
                      suns[available[i]].direction), available[i]));
        const std::size_t neighbor_count = std::min<std::size_t>(3,
                                                                 nearest.size());
        std::partial_sort(nearest.begin(), nearest.begin()+neighbor_count,
                          nearest.end());
        for (std::size_t view = first_view; view < last_view; ++view) {
            double weighted_response = 0.0;
            double weight_sum = 0.0;
            for (std::size_t neighbor = 0;
                    neighbor < neighbor_count; ++neighbor) {
                const double distance = std::max(1.0e-8,
                                                  nearest[neighbor].first);
                const double weight = 1.0/(distance*distance);
                const std::size_t sample_time = nearest[neighbor].second;
                weighted_response += weight*normalized_sun_response(
                    contrast[view*nsteps+sample_time], suns[sample_time]);
                weight_sum += weight;
            }
            contrast[view*nsteps+time] = weighted_response/weight_sum*
                suns[time].radiance_scale*suns[time].radiance_scale*
                suns[time].omega;
            if (illuminance) {
                double weighted_illuminance = 0.0;
                for (std::size_t neighbor = 0;
                        neighbor < neighbor_count; ++neighbor) {
                    const double distance = std::max(
                        1.0e-8, nearest[neighbor].first);
                    const double weight = 1.0/(distance*distance);
                    const std::size_t sample_time =
                        nearest[neighbor].second;
                    weighted_illuminance += weight*
                        normalized_sun_illuminance(
                            (*illuminance)[view*nsteps+sample_time],
                            suns[sample_time]);
                }
                (*illuminance)[view*nsteps+time] =
                    weighted_illuminance/weight_sum*
                    suns[time].radiance_scale*suns[time].omega;
            }
        }
    }
}

struct AdaptiveSunStats {
    std::size_t sampled;
    std::size_t leaves;
    int passes;
};

AdaptiveSunStats evaluate_adaptive_suns(
    const AdaptiveSunTree &tree, const std::vector<Sun> &suns,
    const Options &options, std::size_t first_view,
    std::size_t last_view, std::size_t nsteps,
    std::vector<double> &contrast, std::vector<double> *illuminance,
    const std::function<void(const std::vector<std::size_t> &)> &evaluate)
{
    const double maximum_angle = options.adaptive_sun_max_angle*PI/180.0;
    const double minimum_angle = options.adaptive_sun_min_angle*PI/180.0;
    std::vector<int> frontier;
    append_initial_sun_nodes(tree, tree.root, maximum_angle, frontier);
    std::vector<int> leaves;
    std::vector<char> sampled(suns.size(), 0);
    AdaptiveSunStats stats = {0, 0, 0};

    while (!frontier.empty()) {
        std::vector<std::size_t> pending;
        for (std::size_t n = 0; n < frontier.size(); ++n) {
            const std::vector<std::size_t> probes =
                sun_node_probes(tree.nodes[frontier[n]]);
            for (std::size_t p = 0; p < probes.size(); ++p)
                if (!sampled[probes[p]]) {
                    sampled[probes[p]] = 1;
                    pending.push_back(probes[p]);
                }
        }
        if (!pending.empty()) {
            evaluate(pending);
            stats.sampled += pending.size();
        }
        ++stats.passes;

        std::vector<int> next;
        std::vector<std::size_t> terminal_pending;
        for (std::size_t n = 0; n < frontier.size(); ++n) {
            const AdaptiveSunNode &node = tree.nodes[frontier[n]];
            const bool can_refine = node.child0 >= 0 &&
                node.members.size() > 3 && node.diameter > minimum_angle;
            const bool smooth = sun_node_response_is_smooth(
                node, suns, contrast, nsteps, first_view, last_view,
                illuminance, options.adaptive_sun_error);
            if (!smooth) {
                if (can_refine) {
                    next.push_back(node.child0);
                    next.push_back(node.child1);
                    continue;
                }
                for (std::size_t member = 0;
                        member < node.members.size(); ++member) {
                    const std::size_t time = node.members[member];
                    if (!sampled[time]) {
                        sampled[time] = 1;
                        terminal_pending.push_back(time);
                    }
                }
                leaves.push_back(frontier[n]);
            } else {
                leaves.push_back(frontier[n]);
            }
        }
        if (!terminal_pending.empty()) {
            evaluate(terminal_pending);
            stats.sampled += terminal_pending.size();
        }
        frontier.swap(next);
    }

    for (std::size_t leaf = 0; leaf < leaves.size(); ++leaf)
        interpolate_sun_node(tree.nodes[leaves[leaf]], suns, sampled,
                             first_view, last_view, nsteps, contrast,
                             illuminance);
    stats.leaves = leaves.size();
    return stats;
}

ReflectionData load_rad_normals(const std::string &path,
                                const std::set<std::string> &materials,
                                double tolerance_degrees)
{
    const std::vector<Primitive> primitives = load_primitives(path);
    ReflectionData reflections;
    const double cosine_tolerance =
        std::cos(tolerance_degrees*PI/180.0);
    for (std::size_t p = 0; p < primitives.size(); ++p) {
        const Primitive &primitive = primitives[p];
        if (primitive.type != "polygon" ||
                (!materials.empty() &&
                 !materials.count(primitive.modifier)))
            continue;
        if (primitive.real_args.size() < 9 || primitive.real_args.size()%3)
            throw std::runtime_error("target mirror polygon has invalid coordinates");
        std::vector<Vec3> vertices;
        for (std::size_t i = 0; i < primitive.real_args.size(); i += 3)
            vertices.push_back(Vec3{{primitive.real_args[i],
                                     primitive.real_args[i+1],
                                     primitive.real_args[i+2]}});
        const Vec3 edge0 = Vec3{{vertices[1][0]-vertices[0][0],
                                 vertices[1][1]-vertices[0][1],
                                 vertices[1][2]-vertices[0][2]}};
        for (std::size_t i = 2; i < vertices.size(); ++i) {
            const Vec3 edge1 = Vec3{{vertices[i][0]-vertices[0][0],
                                     vertices[i][1]-vertices[0][1],
                                     vertices[i][2]-vertices[0][2]}};
            const Vec3 candidate = cross(edge0, edge1);
            if (norm(candidate) > kEpsilon) {
                NormalSource source;
                source.view_index = kAllViews;
                source.modifier = primitive.modifier;
                source.material = primitive.modifier;
                source.primitive = primitive.identifier;
                append_sourced_normal(reflections, candidate, &source,
                                      cosine_tolerance);
                break;
            }
        }
    }
    if (reflections.first_order_normals.empty())
        throw std::runtime_error("no eligible polygons found in '" + path + "'");
    return reflections;
}

Vec3 reflected_eye_direction(const Vec3 &sun_direction, const Vec3 &normal)
{
    return normalized(add_scaled(sun_direction, normal,
                                 -2.0*dot(sun_direction, normal)),
                      "reflected ray");
}

Vec3 twice_reflected_eye_direction(const Vec3 &sun_direction,
                                   const Vec3 &first_normal,
                                   const Vec3 &second_normal)
{
    const Vec3 after_second = reflected_eye_direction(sun_direction,
                                                       second_normal);
    return reflected_eye_direction(after_second, first_normal);
}

void view_basis(const Vec3 &forward, const Vec3 &up, Vec3 &right, Vec3 &local_up)
{
    local_up = add_scaled(up, forward, -dot(up, forward));
    if (norm(local_up) <= kEpsilon) {
        const Vec3 global_z = {{0.0, 0.0, 1.0}};
        local_up = add_scaled(global_z, forward, -forward[2]);
        if (norm(local_up) <= kEpsilon) {
            const Vec3 global_y = {{0.0, 1.0, 0.0}};
            local_up = add_scaled(global_y, forward, -forward[1]);
        }
    }
    local_up = normalized(local_up, "view up vector");
    right = normalized(cross(forward, local_up), "view horizontal vector");
}

double guth_position_index(const Vec3 &source_direction,
                           const Vec3 &forward, const Vec3 &up)
{
    Vec3 horizontal, local_up;
    view_basis(forward, up, horizontal, local_up);
    const Vec3 source = normalized(source_direction,
                                   "glare-source direction");
    double sigma = angle(source, forward);
    if (sigma < 1.0e-9)
        return 1.0;
    Vec3 projected = add_scaled(source, forward, -dot(source, forward));
    if (norm(projected) < kEpsilon)
        return 16.0;
    projected = normalized(projected, "projected glare-source direction");
    double tau = angle(projected, local_up);
    double position;
    if (dot(projected, local_up) >= 0.0) {
        tau *= 180.0/PI;
        sigma *= 180.0/PI;
        position = std::exp(
            (35.2 - 0.31889*tau - 1.22*std::exp(-2.0*tau/9.0))
                /1000.0*sigma +
            (21.0 + 0.26667*tau - 0.002963*tau*tau)
                /100000.0*sigma*sigma);
    } else {
        const double beta = std::atan(std::tan(sigma)*std::sqrt(
            1.0 + 0.3225*std::cos(tau)*std::cos(tau)))*180.0/PI;
        position = std::exp(6.49/1000.0*beta +
                            21.0/100000.0*beta*beta);
    }
    return std::max(1.0, std::min(position, 16.0));
}

std::vector<std::string> split_words(const std::string &text)
{
    std::istringstream input(text);
    std::vector<std::string> words;
    std::string word;
    while (input >> word)
        words.push_back(word);
    return words;
}

std::vector<char> run_process(const std::vector<std::string> &arguments,
                              const std::vector<float> &input,
                              const std::string &label)
{
    std::vector<char *> argv;
    for (std::size_t i = 0; i < arguments.size(); ++i)
        argv.push_back(const_cast<char *>(arguments[i].c_str()));
    argv.push_back(NULL);
    SUBPROC process_data = sp_inactive;
    const int opened = open_process(&process_data, argv.data());
    if (opened <= 0)
        throw std::runtime_error("cannot start " + label + " ('" +
                                 arguments[0] + "')");

    const int write_fd = process_data.w;
    std::atomic<bool> write_ok(true);
    std::thread writer([&input, write_fd, &write_ok]() {
        const char *position = reinterpret_cast<const char *>(input.data());
        std::size_t remaining = input.size()*sizeof(float);
        while (remaining) {
            const std::size_t request = std::min<std::size_t>(
                remaining, static_cast<std::size_t>(std::numeric_limits<int>::max()));
            const ssize_t count = write(write_fd, position,
                                        static_cast<unsigned int>(request));
            if (count < 0) {
                if (errno == EINTR)
                    continue;
                write_ok = false;
                break;
            }
            if (!count) {
                write_ok = false;
                break;
            }
            position += count;
            remaining -= static_cast<std::size_t>(count);
        }
        close(write_fd);
    });

    std::vector<char> output;
    char buffer[16384];
    bool read_ok = true;
    for (;;) {
        const ssize_t count = read(process_data.r, buffer, sizeof(buffer));
        if (count > 0) {
            output.insert(output.end(), buffer, buffer+count);
            continue;
        }
        if (count < 0 && errno == EINTR)
            continue;
        read_ok = count == 0;
        break;
    }
    writer.join();
    process_data.w = -1;
    close(process_data.r);
    process_data.r = -1;
    const int status = close_process(&process_data);
    if (!write_ok || !read_ok)
        throw std::runtime_error("pipe failure while running " + label);
    if (status != 0) {
        std::ostringstream message;
        message << label << " exited with status " << status;
        throw std::runtime_error(message.str());
    }
    return output;
}

std::vector<float> pack_rays(const std::vector<Candidate> &candidates,
                             const std::vector<Viewpoint> &views)
{
    std::vector<float> rays;
    rays.reserve(candidates.size()*6);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const Viewpoint &view = views[candidates[i].view_index];
        for (int component = 0; component < 3; ++component)
            rays.push_back(static_cast<float>(view.origin[component]));
        for (int component = 0; component < 3; ++component)
            rays.push_back(static_cast<float>(candidates[i].direction[component]));
    }
    return rays;
}

class TemporaryFiles {
public:
    ~TemporaryFiles()
    {
        for (std::size_t i = 0; i < paths_.size(); ++i)
            std::remove(paths_[i].c_str());
    }

    std::string add(char *name_template)
    {
        char path[4096];
        const int descriptor = temp_fd(path, sizeof(path), name_template);
        if (descriptor < 0)
            throw std::runtime_error("cannot create a temporary file");
        close(descriptor);
        paths_.push_back(path);
        return path;
    }

private:
    std::vector<std::string> paths_;
};

void write_bytes(const std::string &path, const std::vector<char> &bytes)
{
    std::ofstream output(path.c_str(), std::ios::binary);
    if (!output)
        throw std::runtime_error("cannot write temporary file '" + path + "'");
    if (!bytes.empty())
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output)
        throw std::runtime_error("error writing temporary file '" + path + "'");
}

std::string path_leaf(const std::string &path)
{
    const std::string::size_type separator = path.find_last_of("/\\");
    return separator == std::string::npos ? path : path.substr(separator+1);
}

std::vector<std::string> reflection_scene_arguments(const Options &options)
{
    std::ifstream input(options.octree.c_str(), std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot inspect octree header '" +
                                 options.octree + "'");
    std::string command_line;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line[line.size()-1] == '\r')
            line.erase(line.size()-1);
        if (line.empty())
            break;
        if (line.compare(0, 6, "oconv ") == 0)
            command_line = line;
    }
    if (command_line.empty())
        throw std::runtime_error(
            "input octree does not retain its oconv source command; provide "
            "--reflection-octree built from the sunless scene and skyglow");

    const std::vector<std::string> words = split_words(command_line);
    std::vector<std::string> arguments;
    const std::string suns_leaf = path_leaf(options.suns_path);
    for (std::size_t i = 1; i < words.size(); ++i) {
        if (words[i] == "-f")
            continue;
        if (words[i] == "-i")
            throw std::runtime_error(
                "input octree is frozen or nested, so its solar sources cannot "
                "be removed; provide --reflection-octree");
        if (words[i] == options.suns_path || path_leaf(words[i]) == suns_leaf)
            continue;
        arguments.push_back(words[i]);
    }
    if (arguments.empty())
        throw std::runtime_error(
            "no sunless scene inputs remain in the octree header; provide "
            "--reflection-octree");
    return arguments;
}

std::string make_reflection_octree(const Options &options,
                                   TemporaryFiles &temporary)
{
    if (!options.reflection_octree.empty())
        return options.reflection_octree;

    char sky_template[] = "specularcontrast_sky_XXXXXX";
    char octree_template[] = "specularcontrast_scene_XXXXXX";
    const std::string sky_path = temporary.add(sky_template);
    const std::string octree_path = temporary.add(octree_template);
    {
        std::ofstream sky(sky_path.c_str());
        if (!sky)
            throw std::runtime_error("cannot write temporary reflection sky");
        sky << "void glow skyglow\n"
               "0\n0\n4 1 1 1 0\n\n"
               "skyglow source sky\n"
               "0\n0\n4 0 0 1 180\n";
        if (!sky)
            throw std::runtime_error("error writing temporary reflection sky");
    }

    std::vector<std::string> command;
    command.push_back(options.oconv);
    command.push_back("-w");
    const std::vector<std::string> scene_arguments =
        reflection_scene_arguments(options);
    command.insert(command.end(), scene_arguments.begin(),
                   scene_arguments.end());
    command.push_back(sky_path);
    const std::vector<float> no_input;
    const std::vector<char> octree = run_process(
        command, no_input, "oconv reflection-search scene");
    if (octree.empty())
        throw std::runtime_error("oconv returned an empty reflection-search scene");
    write_bytes(octree_path, octree);
    return octree_path;
}

std::vector<Vec3> reflection_search_directions(int level, int seed)
{
    const int side = 1 << level;
    const std::size_t count = static_cast<std::size_t>(side)*side;
    std::mt19937 generator(static_cast<unsigned int>(seed));
    std::uniform_real_distribution<double> jitter(
        0.0, 0.5/static_cast<double>(side));
    std::vector<Vec3> directions;
    directions.reserve(count);

    // Raytraverse-style equal-area Shirley-Chiu sampling on the canonical
    // +Y hemisphere.  Rays are rotated into each observer's forward frame by
    // pack_reflection_search_rays().
    for (int row = 0; row < side; ++row) {
        for (int column = 0; column < side; ++column) {
            const double u = row/static_cast<double>(side) + jitter(generator);
            const double v = column/static_cast<double>(side) + jitter(generator);
            const double a = 2.0*u - 1.0;
            const double b = 2.0*v - 1.0;
            const bool a_dominant = a*a > b*b;
            const double radius = a_dominant ? a :
                (std::fabs(b) <= kEpsilon ? 0.0 : b);
            double phi = 0.0;
            if (a_dominant)
                phi = b/(2.0*a);
            else if (std::fabs(b) > kEpsilon)
                phi = 1.0-a/(2.0*b);
            phi *= PI/2.0;
            const double spherical = radius*std::sqrt(
                std::max(0.0, 2.0-radius*radius));

            // Canonical local coordinates use +Y as the hemisphere apex.
            const Vec3 direction = {{
                std::cos(phi)*spherical,
                1.0-radius*radius,
                std::sin(phi)*spherical
            }};
            directions.push_back(normalized(direction,
                                             "reflection-search direction"));
        }
    }
    return directions;
}

std::vector<float> pack_reflection_search_rays(
    const std::vector<Viewpoint> &views, std::size_t first, std::size_t last,
    const std::vector<Vec3> &directions)
{
    std::vector<float> rays;
    rays.reserve((last-first)*directions.size()*6);
    for (std::size_t view = first; view < last; ++view) {
        const Vec3 &forward = views[view].direction;
        const Vec3 reference = std::fabs(forward[2]) < 0.9 ?
            Vec3{{0.0, 0.0, 1.0}} : Vec3{{0.0, 1.0, 0.0}};
        const Vec3 local_x = normalized(
            cross(forward, reference), "reflection-search horizontal axis");
        const Vec3 local_z = cross(local_x, forward);
        for (std::size_t direction = 0; direction < directions.size(); ++direction) {
            const Vec3 world_direction = {{
                directions[direction][0]*local_x[0] +
                    directions[direction][1]*forward[0] +
                    directions[direction][2]*local_z[0],
                directions[direction][0]*local_x[1] +
                    directions[direction][1]*forward[1] +
                    directions[direction][2]*local_z[1],
                directions[direction][0]*local_x[2] +
                    directions[direction][1]*forward[2] +
                    directions[direction][2]*local_z[2]
            }};
            if (dot(world_direction, forward) <= kEpsilon)
                throw std::runtime_error(
                    "reflection-search ray escaped the observer hemisphere");
            for (int component = 0; component < 3; ++component)
                rays.push_back(static_cast<float>(views[view].origin[component]));
            for (int component = 0; component < 3; ++component)
                rays.push_back(static_cast<float>(world_direction[component]));
        }
    }
    return rays;
}

struct TraceRecord {
    Vec3 normal;
    Vec3 direction;
    std::string primitive;
    std::string modifier;
    std::string material;
};

bool parse_trace_double(const std::string &word, double &value)
{
    char *end = NULL;
    errno = 0;
    value = std::strtod(word.c_str(), &end);
    return end != word.c_str() && *end == '\0' &&
        (errno == 0 || errno == ERANGE);
}

void append_unique_path(std::vector<ReflectionPath2> &paths,
                        const ReflectionPath2 &candidate,
                        double cosine_tolerance)
{
    ReflectionPath2 path = candidate;
    path.first_normal = canonical_normal(path.first_normal);
    path.second_normal = canonical_normal(path.second_normal);
    for (std::size_t i = 0; i < paths.size(); ++i)
        if (paths[i].view_index == path.view_index &&
                std::fabs(dot(paths[i].first_normal,
                              path.first_normal)) >= cosine_tolerance &&
                std::fabs(dot(paths[i].second_normal,
                              path.second_normal)) >= cosine_tolerance &&
                paths[i].first_material == path.first_material &&
                paths[i].second_material == path.second_material &&
                paths[i].first_modifier == path.first_modifier &&
                paths[i].second_modifier == path.second_modifier)
            return;
    paths.push_back(path);
}

void collect_reflection_paths(const std::vector<char> &output,
                              std::size_t first_view,
                              std::size_t view_count,
                              std::size_t directions_per_view,
                              const std::set<std::string> &materials,
                              double cosine_tolerance,
                              int max_specular_bounces,
                              ReflectionData &result)
{
    struct LeveledRecord {
        std::size_t level;
        TraceRecord record;
    };
    const std::string text(output.begin(), output.end());
    std::istringstream lines(text);
    std::vector<LeveledRecord> records;
    std::string line;
    while (std::getline(lines, line)) {
        std::size_t level = 0;
        while (level < line.size() && line[level] == '\t')
            ++level;
        const std::vector<std::string> words = split_words(line.substr(level));
        if (words.empty())
            continue;
        if (words.size() != 9)
            throw std::runtime_error(
                "unexpected rtrace reflection-tree record width");
        TraceRecord record;
        for (int component = 0; component < 3; ++component) {
            if (!parse_trace_double(words[component],
                                    record.normal[component]) ||
                    !parse_trace_double(words[component+3],
                                        record.direction[component]))
                throw std::runtime_error(
                    "invalid numeric value in rtrace reflection tree");
        }
        record.primitive = words[6];
        record.modifier = words[7];
        record.material = words[8];
        LeveledRecord leveled;
        leveled.level = level;
        leveled.record = record;
        records.push_back(leveled);
    }

    // Trace trees are emitted in postorder, and each level-0 record closes
    // one input ray.  Keeping these boundaries preserves the originating
    // viewpoint for an ordered eye -> surface 1 -> surface 2 -> sky path.
    std::size_t segment_start = 0;
    std::size_t primary_index = 0;
    for (std::size_t segment_end = 0; segment_end < records.size();
            ++segment_end) {
        if (records[segment_end].level != 0)
            continue;
        const std::size_t view_index = first_view +
            primary_index/directions_per_view;
        if (view_index >= first_view+view_count)
            throw std::runtime_error(
                "rtrace returned too many reflection-search trees");

        for (std::size_t sky = segment_start; sky < segment_end; ++sky) {
            if (records[sky].record.material != "skyglow" ||
                    records[sky].level == 0)
                continue;

            const std::size_t parent_level = records[sky].level-1;
            std::size_t parent = records.size();
            for (std::size_t i = sky+1; i <= segment_end; ++i) {
                if (records[i].level < parent_level)
                    break;
                if (records[i].level == parent_level) {
                    parent = i;
                    break;
                }
            }
            if (parent == records.size())
                continue;
            const TraceRecord &second = records[parent].record;
            const bool second_target = materials.empty() ||
                materials.count(second.material) != 0 ||
                materials.count(second.modifier) != 0;

            // The rtrace tree already contains the actual child ray emitted by
            // Radiance.  Keep its perturbed shading normal as the proposal;
            // re-applying the ideal reflection law here would reject textured
            // normals and finite rough-specular lobes.
            if (second_target && norm(second.normal) > kEpsilon) {
                NormalSource source;
                source.view_index = view_index;
                source.modifier = second.modifier;
                source.material = second.material;
                source.primitive = second.primitive;
                append_sourced_normal(result, second.normal, &source,
                                      cosine_tolerance);
            }

            if (max_specular_bounces < 2 || records[sky].level != 2 ||
                    !second_target || norm(second.normal) <= kEpsilon)
                continue;

            const std::size_t first_level = parent_level-1;
            std::size_t first = records.size();
            for (std::size_t i = parent+1; i <= segment_end; ++i) {
                if (records[i].level < first_level)
                    break;
                if (records[i].level == first_level) {
                    first = i;
                    break;
                }
            }
            if (first == records.size() || records[first].level != 0)
                continue;
            const TraceRecord &first_record = records[first].record;
            const bool first_target = materials.empty() ||
                materials.count(first_record.material) != 0 ||
                materials.count(first_record.modifier) != 0;
            if (!first_target || norm(first_record.normal) <= kEpsilon)
                continue;

            ReflectionPath2 path;
            path.view_index = view_index;
            path.first_normal = first_record.normal;
            path.second_normal = second.normal;
            path.first_material = first_record.material;
            path.second_material = second.material;
            path.first_modifier = first_record.modifier;
            path.second_modifier = second.modifier;
            path.first_primitive = first_record.primitive;
            path.second_primitive = second.primitive;
            append_unique_path(result.second_order_paths, path,
                               cosine_tolerance);
            NormalSource first_source;
            first_source.view_index = view_index;
            first_source.modifier = first_record.modifier;
            first_source.material = first_record.material;
            first_source.primitive = first_record.primitive;
            append_sourced_normal(result, first_record.normal, &first_source,
                                  cosine_tolerance);
        }
        segment_start = segment_end+1;
        ++primary_index;
    }

    const std::size_t expected = view_count*directions_per_view;
    if (primary_index != expected || segment_start != records.size())
        throw std::runtime_error(
            "unexpected number of rtrace reflection-search trees");
}

ReflectionData discover_reflection_paths(
    const std::vector<Viewpoint> &views, double tolerance_degrees,
    const std::set<std::string> &materials, const Options &options)
{
    TemporaryFiles temporary;
    const std::string reflection_scene =
        make_reflection_octree(options, temporary);
    const std::vector<Vec3> directions = reflection_search_directions(
        options.reflection_level, options.reflection_seed);
    const double cosine_tolerance =
        std::cos(tolerance_degrees*PI/180.0);
    const std::size_t views_per_batch = std::max<std::size_t>(
        1, 131072/directions.size());
    ReflectionData result;

    for (std::size_t first = 0; first < views.size(); first += views_per_batch) {
        const std::size_t last = std::min(views.size(), first+views_per_batch);
        std::vector<std::string> command;
        command.push_back(options.rtrace);
        command.push_back("-n"); command.push_back(std::to_string(options.nproc));
        command.push_back("-h-"); command.push_back("-ffa");
        command.push_back("-ab"); command.push_back("0");
        command.push_back("-lr");
        command.push_back(std::to_string(options.max_specular_bounces));
        command.push_back("-ss"); command.push_back("0");
        command.push_back("-st"); command.push_back(".001");
        command.push_back("-otndsmM"); command.push_back("-w-");
        command.push_back(reflection_scene);
        const std::vector<char> output = run_process(
            command,
            pack_reflection_search_rays(views, first, last, directions),
            "rtrace Raytraverse reflection search");
        collect_reflection_paths(output, first, last-first, directions.size(),
                                 materials, cosine_tolerance,
                                 options.max_specular_bounces, result);
    }
    return result;
}

bool source_uses_any_material(const NormalSource &source,
                              const std::set<std::string> &materials)
{
    return materials.count(source.material) != 0 ||
        materials.count(source.modifier) != 0;
}

bool path_uses_any_material(const ReflectionPath2 &path,
                            const std::set<std::string> &materials)
{
    return materials.count(path.first_material) != 0 ||
        materials.count(path.second_material) != 0 ||
        materials.count(path.first_modifier) != 0 ||
        materials.count(path.second_modifier) != 0;
}

std::set<std::string> discovered_reflection_materials(
    const ReflectionData &reflections)
{
    std::set<std::string> materials;
    for (std::size_t normal = 0;
            normal < reflections.first_order_sources.size(); ++normal)
        for (std::size_t source = 0;
                source < reflections.first_order_sources[normal].size();
                ++source) {
            const NormalSource &record =
                reflections.first_order_sources[normal][source];
            if (!record.material.empty())
                materials.insert(record.material);
            if (!record.modifier.empty())
                materials.insert(record.modifier);
        }
    for (std::size_t path = 0;
            path < reflections.second_order_paths.size(); ++path) {
        const ReflectionPath2 &record = reflections.second_order_paths[path];
        if (!record.first_material.empty())
            materials.insert(record.first_material);
        if (!record.second_material.empty())
            materials.insert(record.second_material);
        if (!record.first_modifier.empty())
            materials.insert(record.first_modifier);
        if (!record.second_modifier.empty())
            materials.insert(record.second_modifier);
    }
    return materials;
}

void merge_missing_material_paths(ReflectionData &target,
                                  const ReflectionData &fallback,
                                  const std::set<std::string> &missing,
                                  double tolerance_degrees)
{
    const double cosine_tolerance =
        std::cos(tolerance_degrees*PI/180.0);
    for (std::size_t normal = 0;
            normal < fallback.first_order_normals.size(); ++normal) {
        if (normal >= fallback.first_order_sources.size())
            continue;
        for (std::size_t source = 0;
                source < fallback.first_order_sources[normal].size();
                ++source) {
            const NormalSource &record =
                fallback.first_order_sources[normal][source];
            if (source_uses_any_material(record, missing))
                append_sourced_normal(target,
                    fallback.first_order_normals[normal], &record,
                    cosine_tolerance);
        }
    }
    for (std::size_t index = 0;
            index < fallback.second_order_paths.size(); ++index) {
        const ReflectionPath2 &path = fallback.second_order_paths[index];
        if (!path_uses_any_material(path, missing))
            continue;
        append_unique_path(target.second_order_paths, path,
                           cosine_tolerance);
        NormalSource first;
        first.view_index = path.view_index;
        first.modifier = path.first_modifier;
        first.material = path.first_material;
        first.primitive = path.first_primitive;
        append_sourced_normal(target, path.first_normal, &first,
                              cosine_tolerance);
        NormalSource second;
        second.view_index = path.view_index;
        second.modifier = path.second_modifier;
        second.material = path.second_material;
        second.primitive = path.second_primitive;
        append_sourced_normal(target, path.second_normal, &second,
                              cosine_tolerance);
    }
}

ReflectionData reflection_data_from_normals(
    const std::vector<Vec3> &normals, std::size_t view_count,
    int max_specular_bounces)
{
    ReflectionData result;
    result.first_order_normals = normals;
    result.first_order_sources.resize(normals.size());
    if (max_specular_bounces < 2)
        return result;
    for (std::size_t view = 0; view < view_count; ++view)
        for (std::size_t first = 0; first < normals.size(); ++first)
            for (std::size_t second = 0; second < normals.size(); ++second) {
                ReflectionPath2 path;
                path.view_index = view;
                path.first_normal = normals[first];
                path.second_normal = normals[second];
                result.second_order_paths.push_back(path);
            }
    return result;
}

void append_all_normal_pairs(ReflectionData &result,
                             std::size_t view_count,
                             double tolerance_degrees)
{
    const double cosine_tolerance =
        std::cos(tolerance_degrees*PI/180.0);
    for (std::size_t view = 0; view < view_count; ++view)
        for (std::size_t first = 0;
                first < result.first_order_normals.size(); ++first)
            for (std::size_t second = 0;
                    second < result.first_order_normals.size(); ++second) {
                ReflectionPath2 path;
                path.view_index = view;
                path.first_normal = result.first_order_normals[first];
                path.second_normal = result.first_order_normals[second];
                append_unique_path(result.second_order_paths, path,
                                   cosine_tolerance);
            }
}

void save_normals(const std::string &path, const ReflectionData &reflections)
{
    std::ofstream output(path.c_str());
    if (!output)
        throw std::runtime_error("cannot write normal file '" + path + "'");
    output << "# nx ny nz; source metadata remain comments for legacy -N "
              "compatibility\n";
    output << std::setprecision(12);
    for (std::size_t i = 0; i < reflections.first_order_normals.size(); ++i) {
        const Vec3 &normal = reflections.first_order_normals[i];
        const std::vector<NormalSource> *sources =
            i < reflections.first_order_sources.size() ?
            &reflections.first_order_sources[i] : NULL;
        if (!sources || sources->empty()) {
            output << normal[0] << ' ' << normal[1] << ' ' << normal[2]
                   << '\n';
            continue;
        }
        for (std::size_t source_index = 0;
                source_index < sources->size(); ++source_index) {
            const NormalSource &source = (*sources)[source_index];
            output << normal[0] << ' ' << normal[1] << ' ' << normal[2]
                   << " # view=";
            if (source.view_index == kAllViews)
                output << '*';
            else
                output << source.view_index;
            output << " modifier="
                   << (source.modifier.empty() ? "-" : source.modifier)
                   << " material="
                   << (source.material.empty() ? "-" : source.material)
                   << " primitive="
                   << (source.primitive.empty() ? "-" : source.primitive)
                   << '\n';
        }
    }
}

void save_reflection_paths(const std::string &path,
                           const std::vector<ReflectionPath2> &paths)
{
    std::ofstream output(path.c_str());
    if (!output)
        throw std::runtime_error("cannot write reflection path file '" +
                                 path + "'");
    output << "# view n1x n1y n1z n2x n2y n2z material1 material2 "
              "modifier1 modifier2 primitive1 primitive2\n";
    output << std::setprecision(12);
    for (std::size_t i = 0; i < paths.size(); ++i)
        output << paths[i].view_index << ' '
               << paths[i].first_normal[0] << ' '
               << paths[i].first_normal[1] << ' '
               << paths[i].first_normal[2] << ' '
               << paths[i].second_normal[0] << ' '
               << paths[i].second_normal[1] << ' '
               << paths[i].second_normal[2] << ' '
               << (paths[i].first_material.empty() ? "-" :
                   paths[i].first_material) << ' '
               << (paths[i].second_material.empty() ? "-" :
                   paths[i].second_material) << ' '
               << (paths[i].first_modifier.empty() ? "-" :
                   paths[i].first_modifier) << ' '
               << (paths[i].second_modifier.empty() ? "-" :
                   paths[i].second_modifier) << ' '
               << (paths[i].first_primitive.empty() ? "-" :
                   paths[i].first_primitive) << ' '
               << (paths[i].second_primitive.empty() ? "-" :
                   paths[i].second_primitive) << '\n';
}

std::size_t normal_source_count(const ReflectionData &reflections)
{
    std::size_t count = 0;
    for (std::size_t normal = 0;
            normal < reflections.first_order_sources.size(); ++normal)
        count += reflections.first_order_sources[normal].size();
    return count;
}

void append_candidate(std::vector<Candidate> &candidates,
                      std::size_t first_for_view_time,
                      std::size_t time_index, std::size_t local_modifier,
                      std::size_t view_index, std::size_t seed_index,
                      int reflection_order, const Vec3 &direction,
                      double solid_angle,
                      const std::vector<Viewpoint> &views, const Vec3 &up)
{
    if (dot(direction, views[view_index].direction) <= kEpsilon)
        return;
    const double duplicate_cosine = 1.0-1.0e-10;
    for (std::size_t i = first_for_view_time; i < candidates.size(); ++i)
        if (dot(direction, candidates[i].direction) >= duplicate_cosine) {
            candidates[i].reflection_orders |=
                1u << static_cast<unsigned int>(reflection_order-1);
            candidates[i].solid_angle = std::max(
                candidates[i].solid_angle, solid_angle);
            return;
        }
    Candidate candidate;
    candidate.time_index = time_index;
    candidate.local_modifier = local_modifier;
    candidate.view_index = view_index;
    candidate.seed_index = seed_index;
    candidate.disk_seed_index = 0;
    candidate.reflection_orders =
        1u << static_cast<unsigned int>(reflection_order-1);
    candidate.direct_sun = false;
    candidate.direction = direction;
    candidate.position_index = guth_position_index(
        direction, views[view_index].direction, up);
    candidate.solid_angle = solid_angle;
    candidates.push_back(candidate);
}

bool normal_applies_to_view(const ReflectionData &reflections,
                            std::size_t normal_index,
                            std::size_t view_index)
{
    if (normal_index >= reflections.first_order_sources.size() ||
            reflections.first_order_sources[normal_index].empty())
        return true;
    const std::vector<NormalSource> &sources =
        reflections.first_order_sources[normal_index];
    for (std::size_t source = 0; source < sources.size(); ++source)
        if (sources[source].view_index == kAllViews ||
                sources[source].view_index == view_index)
            return true;
    return false;
}

std::vector<Candidate> build_candidates(
    const std::vector<std::pair<std::size_t, Sun> > &chunk,
    const std::vector<Viewpoint> &views, const ReflectionData &reflections,
    const Vec3 &up, std::size_t first_view, std::size_t last_view)
{
    if (first_view > last_view || last_view > views.size())
        throw std::runtime_error("invalid viewpoint batch range");
    std::vector<std::vector<std::size_t> > paths_by_view(views.size());
    for (std::size_t path = 0;
            path < reflections.second_order_paths.size(); ++path) {
        const std::size_t view =
            reflections.second_order_paths[path].view_index;
        if (view >= views.size())
            throw std::runtime_error("reflection path has an invalid view index");
        paths_by_view[view].push_back(path);
    }

    std::vector<Candidate> candidates;
    for (std::size_t local = 0; local < chunk.size(); ++local) {
        for (std::size_t view = first_view; view < last_view; ++view) {
            const std::size_t first_for_view_time = candidates.size();
            for (std::size_t normal = 0;
                    normal < reflections.first_order_normals.size(); ++normal) {
                if (!normal_applies_to_view(reflections, normal, view))
                    continue;
                const Vec3 direction = reflected_eye_direction(
                    chunk[local].second.direction,
                    reflections.first_order_normals[normal]);
                append_candidate(candidates, first_for_view_time,
                                 chunk[local].first, local, view, normal, 1,
                                 direction, chunk[local].second.omega,
                                 views, up);
            }
            for (std::size_t local_path = 0;
                    local_path < paths_by_view[view].size(); ++local_path) {
                const std::size_t path_index = paths_by_view[view][local_path];
                const ReflectionPath2 &path =
                    reflections.second_order_paths[path_index];
                const Vec3 direction = twice_reflected_eye_direction(
                    chunk[local].second.direction, path.first_normal,
                    path.second_normal);
                append_candidate(candidates, first_for_view_time,
                                 chunk[local].first, local, view, path_index,
                                 2,
                                 direction, chunk[local].second.omega,
                                 views, up);
            }
        }
    }
    return candidates;
}

void append_direct_sun_candidates(
    std::vector<Candidate> &candidates,
    const std::vector<std::pair<std::size_t, Sun> > &chunk,
    const std::vector<Viewpoint> &views, const Vec3 &up,
    std::size_t first_view, std::size_t last_view)
{
    for (std::size_t local = 0; local < chunk.size(); ++local)
        for (std::size_t view = first_view; view < last_view; ++view) {
            const Vec3 &direction = chunk[local].second.direction;
            if (dot(direction, views[view].direction) <= kEpsilon)
                continue;
            Candidate candidate;
            candidate.time_index = chunk[local].first;
            candidate.local_modifier = local;
            candidate.view_index = view;
            candidate.seed_index = 0;
            candidate.disk_seed_index = 0;
            candidate.reflection_orders = 0;
            candidate.direct_sun = true;
            candidate.direction = direction;
            candidate.position_index = guth_position_index(
                direction, views[view].direction, up);
            candidate.solid_angle = chunk[local].second.omega;
            candidates.push_back(candidate);
        }
}

struct RoughCap {
    Candidate seed;
    double radius;
    double cosine_radius;
    double omega;
};

int highest_reflection_order(unsigned int orders)
{
    return (orders & 2u) ? 2 : 1;
}

double rough_cap_radius(const Candidate &candidate, const Sun &sun,
                        const Options &options)
{
    const double solar_radius = std::acos(std::max(
        -1.0, std::min(1.0, 1.0-sun.omega/(2.0*PI))));
    const double single_bounce_sigma = 2.0*std::atan(options.roughness);
    const double effective_sigma = single_bounce_sigma*std::sqrt(
        static_cast<double>(highest_reflection_order(
            candidate.reflection_orders)));
    return std::min(PI, solar_radius +
                    options.rough_extent*effective_sigma);
}

void concentric_disk(double u, double v, double &x, double &y)
{
    const double a = 2.0*u-1.0;
    const double b = 2.0*v-1.0;
    if (std::fabs(a) <= kEpsilon && std::fabs(b) <= kEpsilon) {
        x = y = 0.0;
        return;
    }
    double radius;
    double phi;
    if (std::fabs(a) > std::fabs(b)) {
        radius = a;
        phi = PI/4.0*(b/a);
    } else {
        radius = b;
        phi = PI/2.0-PI/4.0*(a/b);
    }
    x = radius*std::cos(phi);
    y = radius*std::sin(phi);
}

std::uint64_t stable_double_bits(double value)
{
    if (value == 0.0)
        value = 0.0;
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::size_t stable_vector_key(const Vec3 &value)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (int component = 0; component < 3; ++component) {
        hash ^= stable_double_bits(value[component]);
        hash *= 1099511628211ULL;
    }
    return static_cast<std::size_t>(hash);
}

std::size_t candidate_sampling_key(const Candidate &candidate)
{
    std::uint64_t hash = static_cast<std::uint64_t>(
        stable_vector_key(candidate.direction));
    hash ^= static_cast<std::uint64_t>(candidate.reflection_orders) +
        0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    hash ^= static_cast<std::uint64_t>(candidate.direct_sun) +
        0x85ebca6bULL + (hash << 6) + (hash >> 2);
    return static_cast<std::size_t>(hash);
}

double rough_rotation(int seed, std::size_t time_index,
                      std::size_t origin_key, std::size_t proposal_key)
{
    unsigned long long value = static_cast<unsigned int>(seed);
    value ^= (time_index+0x9e3779b9ULL) + (value << 6) + (value >> 2);
    value ^= (origin_key+0x85ebca6bULL) + (value << 6) + (value >> 2);
    value ^= (proposal_key+0xc2b2ae35ULL) + (value << 6) + (value >> 2);
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    const double unit = static_cast<double>(value & 0xffffffffULL)/
                        4294967296.0;
    return 2.0*PI*unit;
}

Vec3 cap_sample_direction(const Vec3 &axis, double radius,
                          double disk_x, double disk_y, double rotation)
{
    const Vec3 reference = std::fabs(axis[2]) < 0.9 ?
        Vec3{{0.0, 0.0, 1.0}} : Vec3{{0.0, 1.0, 0.0}};
    const Vec3 tangent_x = normalized(cross(reference, axis),
                                      "rough-cap tangent");
    const Vec3 tangent_y = cross(axis, tangent_x);
    const double rotated_x = disk_x*std::cos(rotation)-
                             disk_y*std::sin(rotation);
    const double rotated_y = disk_x*std::sin(rotation)+
                             disk_y*std::cos(rotation);
    const double radius_squared = std::min(
        1.0, rotated_x*rotated_x+rotated_y*rotated_y);
    const double cosine_theta = std::max(
        -1.0, 1.0-radius_squared*(1.0-std::cos(radius)));
    const double sine_theta = std::sqrt(std::max(
        0.0, 1.0-cosine_theta*cosine_theta));
    const double disk_radius = std::sqrt(radius_squared);
    const double local_x = disk_radius > kEpsilon ?
        sine_theta*rotated_x/disk_radius : 0.0;
    const double local_y = disk_radius > kEpsilon ?
        sine_theta*rotated_y/disk_radius : 0.0;
    return normalized(Vec3{{
        cosine_theta*axis[0] + local_x*tangent_x[0] +
            local_y*tangent_y[0],
        cosine_theta*axis[1] + local_x*tangent_x[1] +
            local_y*tangent_y[1],
            cosine_theta*axis[2] + local_x*tangent_x[2] +
            local_y*tangent_y[2]}}, "rough-cap sample direction");
}

void expand_solar_disk_candidates(std::vector<WorkChunk> &work,
                                  const std::vector<Viewpoint> &views,
                                  const Vec3 &up, const Options &options)
{
    const int secondary_samples = options.secondary_sun_disk_samples > 0 ?
        options.secondary_sun_disk_samples : options.sun_disk_samples;
    if (options.sun_disk_samples <= 1 && secondary_samples <= 1)
        return;

    for (std::size_t batch = 0; batch < work.size(); ++batch) {
        std::vector<Candidate> expanded;
        expanded.reserve(work[batch].candidates.size()*std::max(
                         options.sun_disk_samples, secondary_samples));
        for (std::size_t candidate_index = 0;
                candidate_index < work[batch].candidates.size();
                ++candidate_index) {
            const Candidate &seed = work[batch].candidates[candidate_index];
            const int sample_count =
                !seed.direct_sun && !(seed.reflection_orders & 1u) ?
                secondary_samples : options.sun_disk_samples;
            const int side = static_cast<int>(
                std::sqrt(static_cast<double>(sample_count)));
            if (seed.view_index >= views.size())
                throw std::runtime_error(
                    "solar-disk candidate has an invalid view index");
            const double radius = std::acos(std::max(
                -1.0, std::min(1.0,
                    1.0-seed.solid_angle/(2.0*PI))));
            const double rotation = rough_rotation(
                options.sun_disk_seed, seed.time_index,
                stable_vector_key(views[seed.view_index].origin),
                candidate_sampling_key(seed));
            for (int row = 0; row < side; ++row)
                for (int column = 0; column < side; ++column) {
                    double disk_x, disk_y;
                    concentric_disk((column+0.5)/side,
                                    (row+0.5)/side,
                                    disk_x, disk_y);
                    Candidate sample = seed;
                    sample.disk_seed_index = candidate_index;
                    sample.direction = cap_sample_direction(
                        seed.direction, radius, disk_x, disk_y, rotation);
                    if (dot(sample.direction,
                            views[seed.view_index].direction) <= kEpsilon)
                        continue;
                    sample.position_index = guth_position_index(
                        sample.direction,
                        views[seed.view_index].direction, up);
                    sample.solid_angle = seed.solid_angle/sample_count;
                    expanded.push_back(sample);
                }
        }
        work[batch].candidates.swap(expanded);
        work[batch].generated = work[batch].candidates.size();
    }
}

void expand_rough_candidates(std::vector<WorkChunk> &work,
                             const std::vector<Sun> &suns,
                             const std::vector<Viewpoint> &views,
                             const Vec3 &up, const Options &options)
{
    if (options.rough_samples <= 0)
        return;
    const int side = static_cast<int>(
        std::sqrt(static_cast<double>(options.rough_samples)));

    for (std::size_t batch = 0; batch < work.size(); ++batch) {
        typedef std::pair<std::size_t, std::size_t> GroupKey;
        std::map<GroupKey, std::vector<Candidate> > groups;
        for (std::size_t i = 0; i < work[batch].candidates.size(); ++i) {
            const Candidate &candidate = work[batch].candidates[i];
            groups[GroupKey(candidate.time_index,
                            candidate.view_index)].push_back(candidate);
        }

        std::vector<Candidate> expanded;
        for (std::map<GroupKey, std::vector<Candidate> >::const_iterator group =
                groups.begin(); group != groups.end(); ++group) {
            if (group->first.first >= suns.size() ||
                    group->first.second >= views.size())
                throw std::runtime_error("invalid rough-sampling group index");
            const Sun &sun = suns[group->first.first];
            const Viewpoint &view = views[group->first.second];
            std::vector<RoughCap> caps;
            caps.reserve(group->second.size());
            for (std::size_t i = 0; i < group->second.size(); ++i) {
                RoughCap cap;
                cap.seed = group->second[i];
                cap.radius = rough_cap_radius(cap.seed, sun, options);
                cap.cosine_radius = std::cos(cap.radius);
                cap.omega = 2.0*PI*(1.0-cap.cosine_radius);
                caps.push_back(cap);
            }

            for (std::size_t cap_index = 0; cap_index < caps.size();
                    ++cap_index) {
                const RoughCap &cap = caps[cap_index];
                const double rotation = rough_rotation(
                    options.rough_seed, cap.seed.time_index,
                    stable_vector_key(view.origin),
                    candidate_sampling_key(cap.seed));
                for (int row = 0; row < side; ++row)
                    for (int column = 0; column < side; ++column) {
                        double disk_x, disk_y;
                        concentric_disk((column+0.5)/side,
                                        (row+0.5)/side,
                                        disk_x, disk_y);
                        const Vec3 direction = cap_sample_direction(
                            cap.seed.direction, cap.radius, disk_x, disk_y,
                            rotation);
                        if (dot(direction, view.direction) <= kEpsilon)
                            continue;

                        double mixture_density = 0.0;
                        for (std::size_t other = 0; other < caps.size(); ++other)
                            if (dot(direction, caps[other].seed.direction) >=
                                    caps[other].cosine_radius-kEpsilon)
                                mixture_density += 1.0/caps[other].omega;
                        if (mixture_density <= 0.0)
                            throw std::runtime_error(
                                "invalid rough-cap mixture density");

                        Candidate sample = cap.seed;
                        sample.direction = direction;
                        sample.rough_row = row;
                        sample.rough_column = column;
                        sample.rough_side = side;
                        sample.position_index = guth_position_index(
                            direction, view.direction, up);
                        sample.solid_angle = 1.0/(
                            options.rough_samples*mixture_density);
                        expanded.push_back(sample);
                    }
            }
        }
        work[batch].candidates.swap(expanded);
        work[batch].generated = work[batch].candidates.size();
    }
}

struct SurfaceHit {
    Vec3 point;
    Vec3 normal;
    std::string material;
};

std::vector<SurfaceHit> parse_surface_hits(const std::vector<char> &output,
                                           std::size_t expected,
                                           const std::string &label)
{
    const std::string text(output.begin(), output.end());
    std::istringstream lines(text);
    std::vector<SurfaceHit> hits;
    std::string line;
    while (std::getline(lines, line)) {
        const std::vector<std::string> words = split_words(line);
        if (words.empty())
            continue;
        if (words.size() != 7)
            throw std::runtime_error("unexpected " + label +
                                     " record width");
        SurfaceHit hit;
        for (int component = 0; component < 3; ++component) {
            if (!parse_trace_double(words[component], hit.point[component]) ||
                    !parse_trace_double(words[component+3],
                                        hit.normal[component]))
                throw std::runtime_error("invalid numeric value in " + label);
        }
        hit.material = words[6];
        hits.push_back(hit);
    }
    if (hits.size() != expected)
        throw std::runtime_error("unexpected " + label + " output size");
    return hits;
}

std::vector<SurfaceHit> trace_surface_hits(
    const std::vector<float> &rays, std::size_t ray_count,
    const Options &options, const std::string &label)
{
    if (!ray_count)
        return std::vector<SurfaceHit>();
    std::vector<std::string> command;
    command.push_back(options.rtrace);
    command.push_back("-n"); command.push_back(std::to_string(options.nproc));
    command.push_back("-h-"); command.push_back("-ffa");
    command.push_back("-opnM"); command.push_back("-w-");
    command.push_back(options.octree);
    return parse_surface_hits(run_process(command, rays, label), ray_count,
                              label);
}

std::vector<SurfaceHit> trace_surface_hits_through(
    const std::vector<float> &rays, std::size_t ray_count,
    const std::set<std::string> &transparent_materials,
    const Options &options, const std::string &label)
{
    if (transparent_materials.empty())
        return trace_surface_hits(rays, ray_count, options, label);
    if (rays.size() != 6*ray_count)
        throw std::runtime_error("invalid ray buffer for " + label);

    std::vector<SurfaceHit> resolved(ray_count);
    std::vector<int> transparent_hits(ray_count, 0);
    std::vector<std::size_t> pending_indices;
    pending_indices.reserve(ray_count);
    for (std::size_t index = 0; index < ray_count; ++index)
        pending_indices.push_back(index);
    std::vector<float> pending_rays = rays;

    while (!pending_indices.empty()) {
        const std::vector<SurfaceHit> hits = trace_surface_hits(
            pending_rays, pending_indices.size(), options, label);
        std::vector<std::size_t> next_indices;
        std::vector<float> next_rays;
        next_indices.reserve(pending_indices.size());
        next_rays.reserve(6*pending_indices.size());

        for (std::size_t pending = 0; pending < pending_indices.size();
                ++pending) {
            const std::size_t original = pending_indices[pending];
            const SurfaceHit &hit = hits[pending];
            const bool may_continue =
                transparent_materials.count(hit.material) &&
                transparent_hits[original] < options.max_transparent_hits &&
                norm(hit.normal) > kEpsilon;
            if (!may_continue) {
                resolved[original] = hit;
                continue;
            }

            ++transparent_hits[original];
            Vec3 direction;
            for (int component = 0; component < 3; ++component)
                direction[component] = pending_rays[6*pending+3+component];
            const Vec3 origin = add_scaled(hit.point, direction, 1.0e-5);
            next_indices.push_back(original);
            for (int component = 0; component < 3; ++component)
                next_rays.push_back(static_cast<float>(origin[component]));
            for (int component = 0; component < 3; ++component)
                next_rays.push_back(static_cast<float>(direction[component]));
        }
        pending_indices.swap(next_indices);
        pending_rays.swap(next_rays);
    }
    return resolved;
}

bool is_reflecting_hit(const SurfaceHit &hit,
                       const std::set<std::string> &materials,
                       const std::set<std::string> &solar_modifiers)
{
    return hit.material != "void" &&
        !solar_modifiers.count(hit.material) &&
        (materials.empty() || materials.count(hit.material)) &&
        norm(hit.normal) > kEpsilon;
}

std::vector<bool> valid_reflection_paths(
    const std::vector<Candidate> &candidates,
    const std::vector<Sun> &suns,
    const std::vector<Viewpoint> &views,
    const std::set<std::string> &materials,
    const std::set<std::string> &transparent_materials,
    const Options &options)
{
    if (candidates.empty())
        return std::vector<bool>();

    std::set<std::string> solar_modifiers;
    for (std::size_t i = 0; i < suns.size(); ++i)
        solar_modifiers.insert(suns[i].modifier);
    const std::vector<SurfaceHit> first_hits = trace_surface_hits_through(
        pack_rays(candidates, views), candidates.size(),
        transparent_materials, options,
        "rtrace first reflection hit check");

    struct SecondProbe {
        std::size_t candidate_index;
        Vec3 incoming;
    };
    std::vector<SecondProbe> second_probes;
    std::vector<float> second_rays;
    std::vector<bool> retain(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const Candidate &candidate = candidates[i];
        if (candidate.direct_sun) {
            retain[i] = true;
            continue;
        }
        const SurfaceHit &first = first_hits[i];
        const bool first_is_target = is_reflecting_hit(
            first, materials, solar_modifiers);
        if (candidate.reflection_orders & 1u) {
            if (materials.empty() || first_is_target)
                retain[i] = true;
        }
        if (!first_is_target)
            continue;
        const Vec3 after_first = reflected_eye_direction(
            candidate.direction, first.normal);
        if (!(candidate.reflection_orders & 2u))
            continue;

        SecondProbe probe;
        probe.candidate_index = i;
        probe.incoming = after_first;
        second_probes.push_back(probe);
        const Vec3 origin = add_scaled(first.point, after_first, 1.0e-5);
        for (int component = 0; component < 3; ++component)
            second_rays.push_back(static_cast<float>(origin[component]));
        for (int component = 0; component < 3; ++component)
            second_rays.push_back(static_cast<float>(after_first[component]));
    }

    const std::vector<SurfaceHit> second_hits = trace_surface_hits_through(
        second_rays, second_probes.size(), transparent_materials, options,
        "rtrace second reflection hit check");
    for (std::size_t probe_index = 0;
            probe_index < second_probes.size(); ++probe_index) {
        const std::size_t candidate_index =
            second_probes[probe_index].candidate_index;
        const SurfaceHit &second = second_hits[probe_index];
        if (!is_reflecting_hit(second, materials, solar_modifiers))
            continue;
        // rcontrib's source sampling is the final directional test.  Keeping
        // every geometrically valid second hit also preserves solar-disk edge
        // samples that a separate center-ray visibility test can miss.
        retain[candidate_index] = true;
    }

    return retain;
}

void filter_work_paths(std::vector<WorkChunk> &work,
                       const std::vector<Sun> &suns,
                       const std::vector<Viewpoint> &views,
                       const std::set<std::string> &materials,
                       const std::set<std::string> &transparent_materials,
                       const Options &options)
{
    std::vector<Candidate> flattened;
    std::vector<std::size_t> source_batch;
    for (std::size_t batch = 0; batch < work.size(); ++batch)
        for (std::size_t i = 0; i < work[batch].candidates.size(); ++i) {
            flattened.push_back(work[batch].candidates[i]);
            source_batch.push_back(batch);
        }
    const std::vector<bool> retain = valid_reflection_paths(
        flattened, suns, views, materials, transparent_materials, options);
    if (retain.size() != flattened.size())
        throw std::runtime_error("invalid reflection-path filter result");

    std::vector<std::vector<Candidate> > retained(work.size());
    for (std::size_t i = 0; i < flattened.size(); ++i)
        if (retain[i])
            retained[source_batch[i]].push_back(flattened[i]);
    for (std::size_t batch = 0; batch < work.size(); ++batch)
        work[batch].candidates.swap(retained[batch]);
}

void filter_rough_work_hits(std::vector<WorkChunk> &work,
                            const std::vector<Sun> &suns,
                            const std::vector<Viewpoint> &views,
                            const std::set<std::string> &materials,
                            const std::set<std::string> &transparent_materials,
                            const Options &options)
{
    std::vector<Candidate> flattened;
    std::vector<std::size_t> source_batch;
    for (std::size_t batch = 0; batch < work.size(); ++batch)
        for (std::size_t i = 0; i < work[batch].candidates.size(); ++i) {
            flattened.push_back(work[batch].candidates[i]);
            source_batch.push_back(batch);
        }
    if (flattened.empty())
        return;

    std::set<std::string> solar_modifiers;
    for (std::size_t i = 0; i < suns.size(); ++i)
        solar_modifiers.insert(suns[i].modifier);
    const std::vector<SurfaceHit> hits = trace_surface_hits_through(
        pack_rays(flattened, views), flattened.size(),
        transparent_materials, options,
        "rtrace rough-sample hit check");
    std::vector<std::vector<Candidate> > retained(work.size());
    for (std::size_t i = 0; i < flattened.size(); ++i)
        if (is_reflecting_hit(hits[i], materials, solar_modifiers))
            retained[source_batch[i]].push_back(flattened[i]);
    for (std::size_t batch = 0; batch < work.size(); ++batch)
        work[batch].candidates.swap(retained[batch]);
}

void validate_rcontrib_options(const std::vector<std::string> &options)
{
    const char *forbidden[] = {"-f", "-h", "-V", "-m", "-M", "-n",
                               "-c", "-x", "-y", "-lr", "-ss"};
    for (std::size_t i = 0; i < options.size(); ++i)
        for (std::size_t j = 0; j < sizeof(forbidden)/sizeof(forbidden[0]); ++j)
            if (options[i] == forbidden[j] ||
                    options[i].compare(0, std::strlen(forbidden[j]), forbidden[j]) == 0)
                throw std::runtime_error("rcontrib option '" + options[i] +
                    "' is managed by specularcontrast");
}

std::vector<std::string> controlled_rcontrib_options(const Options &options)
{
    std::vector<std::string> result = options.rcontrib_options;
    result.push_back("-lr");
    result.push_back(std::to_string(options.max_specular_bounces));
    result.push_back("-ss");
    result.push_back(options.rough_samples > 0 &&
                     options.max_specular_bounces > 1 ?
                     std::to_string(options.rough_secondary_samples) : "0");
    return result;
}

std::vector<float> trace_external_values(
    const std::string &scene,
    const std::vector<std::pair<std::size_t, Sun> > &chunk,
    const std::vector<Candidate> &candidates,
    const std::vector<Viewpoint> &views, const Options &options,
    const std::string &label)
{
    std::vector<std::string> command;
    command.push_back(options.rcontrib);
    command.push_back("-n"); command.push_back(std::to_string(options.nproc));
    command.push_back("-h-"); command.push_back("-fff");
    command.push_back("-V+");
    const std::vector<std::string> render_options =
        controlled_rcontrib_options(options);
    command.insert(command.end(), render_options.begin(),
                   render_options.end());
    for (std::size_t i = 0; i < chunk.size(); ++i) {
        command.push_back("-m");
        command.push_back(chunk[i].second.modifier);
    }
    command.push_back(scene);
    const std::vector<char> output = run_process(
        command, pack_rays(candidates, views), label);
    if (output.size()%sizeof(float))
        throw std::runtime_error(label + " returned a truncated float record");
    std::vector<float> values(output.size()/sizeof(float));
    if (!output.empty())
        std::memcpy(values.data(), output.data(), output.size());
    return values;
}

bool collect_mirror_illuminance(const Options &options)
{
    return !options.mirror_illuminance_output_path.empty();
}

void accumulate_candidate_outputs(
    const Candidate &candidate, double luminance,
    const std::vector<Viewpoint> &views, const Options &options,
    std::vector<double> &contrast, std::vector<double> &illuminance,
    std::size_t nsteps,
    std::map<RoughSourceKey, RoughSourceAccumulator> *rough_sources)
{
    const std::size_t output_index =
        candidate.view_index*nsteps + candidate.time_index;
    if (collect_mirror_illuminance(options) && !candidate.direct_sun &&
            luminance > 0.0) {
        const double projected_cosine = std::max(
            0.0, dot(candidate.direction,
                     views[candidate.view_index].direction));
        illuminance[output_index] += luminance*candidate.solid_angle*
            projected_cosine*options.visible_fraction;
    }
    if (luminance <= options.threshold)
        return;
    if (rough_sources) {
        const RoughSourceKey key(candidate.view_index, candidate.time_index,
                                 candidate.seed_index,
                                 candidate.reflection_orders);
        RoughSourceAccumulator &source = (*rough_sources)[key];
        source.solid_angle += candidate.solid_angle;
        const double weighted_luminance =
            luminance*candidate.solid_angle;
        source.luminance_solid_angle += weighted_luminance;
        for (int component = 0; component < 3; ++component)
            source.direction_moment[component] +=
                weighted_luminance*candidate.direction[component];
    } else {
        contrast[output_index] +=
            luminance*luminance*candidate.solid_angle*
            options.visible_fraction/
            (candidate.position_index*candidate.position_index);
    }
}

void flush_rough_sources(
    const std::map<RoughSourceKey, RoughSourceAccumulator> &rough_sources,
    const std::vector<Viewpoint> &views, const Options &options,
    std::vector<double> &contrast, std::size_t nsteps)
{
    for (std::map<RoughSourceKey, RoughSourceAccumulator>::const_iterator it =
            rough_sources.begin(); it != rough_sources.end(); ++it) {
        const std::size_t view_index = std::get<0>(it->first);
        const std::size_t time_index = std::get<1>(it->first);
        const RoughSourceAccumulator &source = it->second;
        if (source.solid_angle <= kEpsilon ||
                source.luminance_solid_angle <= 0.0)
            continue;
        const double mean_luminance =
            source.luminance_solid_angle/source.solid_angle;
        const Vec3 centroid = normalized(source.direction_moment,
                                          "rough-source centroid");
        const double position_index = guth_position_index(
            centroid, views[view_index].direction, options.up);
        contrast[view_index*nsteps+time_index] +=
            mean_luminance*mean_luminance*source.solid_angle*
            options.visible_fraction/(position_index*position_index);
    }
}

void accumulate_chunk(
    const std::vector<std::pair<std::size_t, Sun> > &chunk,
    const std::vector<Candidate> &candidates,
    const std::vector<float> &values,
    const std::vector<float> *baseline,
    const std::vector<std::size_t> *baseline_rows,
    const Options &options,
    const std::vector<Viewpoint> &views,
    std::vector<double> &contrast, std::vector<double> &illuminance,
    std::size_t nsteps)
{
    if (candidates.empty())
        return;
    const std::size_t values_per_ray = 3*chunk.size();
    const std::size_t expected = values_per_ray*candidates.size();
    if (values.size() != expected)
        throw std::runtime_error("unexpected full-material rcontrib output size");
    if (baseline && !baseline_rows && baseline->size() != expected)
        throw std::runtime_error("unexpected non-specular rcontrib output size");
    if (baseline_rows && baseline_rows->size() != candidates.size())
        throw std::runtime_error("unexpected selective-baseline row map size");

    std::map<RoughSourceKey, RoughSourceAccumulator> rough_sources;
    std::map<RoughSourceKey, RoughSourceAccumulator> *rough_source_output =
        options.cluster_rough_sources ? &rough_sources : NULL;
    for (std::size_t ray = 0; ray < candidates.size(); ++ray) {
        const Candidate &candidate = candidates[ray];
        const std::size_t offset = ray*values_per_ray +
                                   3*candidate.local_modifier;
        double rgb[3] = {values[offset], values[offset+1], values[offset+2]};
        if (baseline && !candidate.direct_sun) {
            std::size_t baseline_offset = offset;
            if (baseline_rows) {
                const std::size_t baseline_row = (*baseline_rows)[ray];
                if (baseline_row == std::numeric_limits<std::size_t>::max())
                    continue;
                baseline_offset = baseline_row*values_per_ray +
                                  3*candidate.local_modifier;
            }
            if (baseline_offset+2 >= baseline->size())
                throw std::runtime_error(
                    "invalid selective non-specular baseline layout");
            for (int component = 0; component < 3; ++component)
                rgb[component] -= (*baseline)[baseline_offset+component];
        }
        const double luminance = kLuminousEfficacy*(
            kBrightness[0]*rgb[0] + kBrightness[1]*rgb[1] +
            kBrightness[2]*rgb[2]);
        if (!std::isfinite(luminance))
            throw std::runtime_error("rcontrib produced a non-finite value");
        accumulate_candidate_outputs(candidate, luminance, views, options,
                                     contrast, illuminance, nsteps,
                                     rough_source_output);
    }
    if (rough_source_output)
        flush_rough_sources(rough_sources, views, options, contrast, nsteps);
}

void accumulate_sparse_chunk(
    const std::vector<Candidate> &candidates,
    const std::vector<float> &values,
    const std::vector<float> *baseline,
    const std::vector<std::size_t> *baseline_rows,
    const Options &options,
    const std::vector<Viewpoint> &views,
    std::vector<double> &contrast, std::vector<double> &illuminance,
    std::size_t nsteps)
{
    if (candidates.empty())
        return;
    if (values.size() != 3*candidates.size())
        throw std::runtime_error(
            "unexpected sparse full-material trace output size");
    if (baseline && !baseline_rows && baseline->size() != values.size())
        throw std::runtime_error(
            "unexpected sparse non-specular trace output size");
    if (baseline_rows && baseline_rows->size() != candidates.size())
        throw std::runtime_error("unexpected selective-baseline row map size");

    std::map<RoughSourceKey, RoughSourceAccumulator> rough_sources;
    std::map<RoughSourceKey, RoughSourceAccumulator> *rough_source_output =
        options.cluster_rough_sources ? &rough_sources : NULL;
    for (std::size_t ray = 0; ray < candidates.size(); ++ray) {
        const Candidate &candidate = candidates[ray];
        const std::size_t offset = 3*ray;
        double rgb[3] = {values[offset], values[offset+1], values[offset+2]};
        if (baseline && !candidate.direct_sun) {
            std::size_t baseline_row = ray;
            if (baseline_rows) {
                baseline_row = (*baseline_rows)[ray];
                if (baseline_row == std::numeric_limits<std::size_t>::max())
                    continue;
            }
            const std::size_t baseline_offset = 3*baseline_row;
            if (baseline_offset+2 >= baseline->size())
                throw std::runtime_error(
                    "invalid sparse selective non-specular baseline layout");
            for (int component = 0; component < 3; ++component)
                rgb[component] -= (*baseline)[baseline_offset+component];
        }
        const double luminance = kLuminousEfficacy*(
            kBrightness[0]*rgb[0] + kBrightness[1]*rgb[1] +
            kBrightness[2]*rgb[2]);
        if (!std::isfinite(luminance))
            throw std::runtime_error(
                "target-source trace produced a non-finite value");
        accumulate_candidate_outputs(candidate, luminance, views, options,
                                     contrast, illuminance, nsteps,
                                     rough_source_output);
    }
    if (rough_source_output)
        flush_rough_sources(rough_sources, views, options, contrast, nsteps);
}

void evaluate_external_chunk(
    const WorkChunk &work, const std::vector<Viewpoint> &views,
    const Options &options, std::vector<double> &contrast,
    std::vector<double> &illuminance, std::size_t nsteps)
{
    if (work.candidates.empty())
        return;
    const std::vector<float> values = trace_external_values(
        options.octree, work.suns, work.candidates, views, options,
        "external rcontrib full-material trace");
    std::vector<float> baseline_values;
    const std::vector<float> *baseline = NULL;
    if (!options.nonspec_octree.empty()) {
        baseline_values = trace_external_values(
            options.nonspec_octree, work.suns, work.candidates, views,
            options, "external rcontrib non-specular baseline");
        baseline = &baseline_values;
    }
    accumulate_chunk(work.suns, work.candidates, values, baseline,
                     NULL, options, views, contrast, illuminance, nsteps);
}

#ifdef SPECULARCONTRIB_BUILTIN_RCONTRIB
struct BuiltinRayKey {
    std::array<std::uint64_t, 6> coordinates;
    std::size_t target_modifier;
    unsigned int reflection_orders;

    bool operator==(const BuiltinRayKey &other) const
    {
        return coordinates == other.coordinates &&
            target_modifier == other.target_modifier &&
            reflection_orders == other.reflection_orders;
    }
};

struct BuiltinRayKeyHash {
    std::size_t operator()(const BuiltinRayKey &key) const
    {
        std::uint64_t hash = 1469598103934665603ULL;
        for (std::size_t i = 0; i < key.coordinates.size(); ++i) {
            hash ^= key.coordinates[i];
            hash *= 1099511628211ULL;
        }
        hash ^= static_cast<std::uint64_t>(key.target_modifier) +
            0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        hash ^= static_cast<std::uint64_t>(key.reflection_orders) +
            0x85ebca6bULL + (hash << 6) + (hash >> 2);
        return static_cast<std::size_t>(hash);
    }
};

struct DeduplicatedBuiltinRays {
    std::vector<SpecularContribRay> rays;
    std::vector<std::size_t> unique_row_for_candidate;
};

BuiltinRayKey builtin_ray_key(const SpecularContribRay &ray)
{
    BuiltinRayKey key;
    for (int component = 0; component < 3; ++component) {
        key.coordinates[component] =
            stable_double_bits(ray.origin[component]);
        key.coordinates[component+3] =
            stable_double_bits(ray.direction[component]);
    }
    key.target_modifier = ray.target_modifier;
    key.reflection_orders = ray.reflection_orders;
    return key;
}

DeduplicatedBuiltinRays deduplicate_builtin_rays(
    const std::vector<Candidate> &candidates,
    const std::vector<Viewpoint> &views, bool origin_reuse)
{
    DeduplicatedBuiltinRays result;
    result.rays.reserve(candidates.size());
    result.unique_row_for_candidate.reserve(candidates.size());
    std::unordered_map<BuiltinRayKey, std::size_t, BuiltinRayKeyHash> rows;
    rows.reserve(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const Viewpoint &view = views[candidates[i].view_index];
        SpecularContribRay ray;
        ray.origin = view.origin;
        ray.direction = candidates[i].direction;
        ray.target_modifier = candidates[i].local_modifier;
        ray.reflection_orders = candidates[i].reflection_orders;
        if (!origin_reuse) {
            result.unique_row_for_candidate.push_back(result.rays.size());
            result.rays.push_back(ray);
            continue;
        }
        const BuiltinRayKey key = builtin_ray_key(ray);
        const std::unordered_map<BuiltinRayKey, std::size_t,
            BuiltinRayKeyHash>::const_iterator found = rows.find(key);
        if (found != rows.end()) {
            result.unique_row_for_candidate.push_back(found->second);
            continue;
        }
        const std::size_t row = result.rays.size();
        rows.insert(std::make_pair(key, row));
        result.rays.push_back(ray);
        result.unique_row_for_candidate.push_back(row);
    }
    return result;
}

std::vector<float> expand_builtin_values(
    const std::vector<float> &unique_values,
    const std::vector<std::size_t> &unique_row_for_candidate)
{
    if (unique_values.size()%3)
        throw std::runtime_error(
            "built-in rcontrib returned a partial RGB row");
    const std::size_t unique_rows = unique_values.size()/3;
    std::vector<float> values(3*unique_row_for_candidate.size());
    for (std::size_t candidate = 0;
            candidate < unique_row_for_candidate.size(); ++candidate) {
        const std::size_t row = unique_row_for_candidate[candidate];
        if (row >= unique_rows)
            throw std::runtime_error(
                "invalid shared built-in rcontrib row index");
        std::memcpy(values.data()+3*candidate,
                    unique_values.data()+3*row, 3*sizeof(float));
    }
    return values;
}

std::vector<std::vector<float> > trace_builtin_batches(
    SpecularRcontribBackend &backend,
    const std::vector<WorkChunk> &work,
    const std::vector<Viewpoint> &views, bool origin_reuse)
{
    std::vector<std::vector<float> > results(work.size());
    bool have_rays = false;
    for (std::size_t i = 0; i < work.size(); ++i)
        have_rays = have_rays || !work[i].candidates.empty();
    if (!have_rays)
        return results;
    for (std::size_t batch = 0; batch < work.size(); ++batch) {
        if (work[batch].candidates.empty())
            continue;
        std::vector<std::string> modifiers;
        modifiers.reserve(work[batch].suns.size());
        for (std::size_t i = 0; i < work[batch].suns.size(); ++i)
            modifiers.push_back(work[batch].suns[i].second.modifier);
        const DeduplicatedBuiltinRays shared = deduplicate_builtin_rays(
            work[batch].candidates, views, origin_reuse);
        const std::vector<float> unique_values = backend.trace(
            modifiers, shared.rays);
        results[batch] = expand_builtin_values(
            unique_values, shared.unique_row_for_candidate);
    }
    return results;
}

std::unique_ptr<SpecularRcontribBackend> create_builtin_backend(
    const std::string &scene, const Options &options,
    const std::set<std::string> &reflection_set)
{
    std::vector<std::string> reflection_modifiers;
    if (options.integrated_path_check) {
        reflection_modifiers.assign(reflection_set.begin(),
                                    reflection_set.end());
    }
    std::unique_ptr<SpecularRcontribBackend> backend(
        new SpecularRcontribBackend(
            options.nproc, controlled_rcontrib_options(options),
            options.direct_specular_only, options.integrated_path_check,
            reflection_modifiers));
    backend->load_scene(scene);
    return backend;
}

std::vector<std::vector<float> > trace_builtin_batches(
    const std::string &scene, const std::vector<WorkChunk> &work,
    const std::vector<Viewpoint> &views, const Options &options)
{
    std::set<std::string> reflection_set;
    if (options.integrated_path_check)
        reflection_set = load_modifier_names(
            options.mirror_modifiers_path, "mirror modifier");
    std::unique_ptr<SpecularRcontribBackend> backend =
        create_builtin_backend(scene, options, reflection_set);
    return trace_builtin_batches(*backend, work, views,
                                 options.origin_reuse);
}
#endif

double candidate_luminance(const WorkChunk &work,
                           const Candidate &candidate, std::size_t ray,
                           const std::vector<float> &values,
                           const std::vector<float> *baseline,
                           bool sparse)
{
    const std::size_t values_per_ray = sparse ? 3 : 3*work.suns.size();
    const std::size_t offset = ray*values_per_ray+
        (sparse ? 0 : 3*candidate.local_modifier);
    if (offset+2 >= values.size() ||
            (baseline && offset+2 >= baseline->size()))
        throw std::runtime_error("invalid rough-prefilter matrix layout");
    double rgb[3] = {values[offset], values[offset+1], values[offset+2]};
    if (baseline)
        for (int component = 0; component < 3; ++component)
            rgb[component] -= (*baseline)[offset+component];
    return kLuminousEfficacy*(kBrightness[0]*rgb[0] +
        kBrightness[1]*rgb[1] + kBrightness[2]*rgb[2]);
}

struct BaselineSelection {
    std::vector<WorkChunk> work;
    std::vector<std::vector<std::size_t> > rows;
    std::size_t selected;
};

BaselineSelection select_baseline_candidates(
    const std::vector<WorkChunk> &work,
    const std::vector<std::vector<float> > &values,
    const Options &options)
{
    if (values.size() != work.size())
        throw std::runtime_error("invalid full-material batch count");
    BaselineSelection result;
    result.work.resize(work.size());
    result.rows.resize(work.size());
    result.selected = 0;
    const std::size_t no_row = std::numeric_limits<std::size_t>::max();

    for (std::size_t batch = 0; batch < work.size(); ++batch) {
        result.work[batch].suns = work[batch].suns;
        result.work[batch].generated = 0;
        result.rows[batch].assign(work[batch].candidates.size(), no_row);
        for (std::size_t ray = 0;
                ray < work[batch].candidates.size(); ++ray) {
            const Candidate &candidate = work[batch].candidates[ray];
            if (candidate.direct_sun)
                continue;
            const double luminance = candidate_luminance(
                work[batch], candidate, ray, values[batch], NULL, true);
            if (!std::isfinite(luminance))
                throw std::runtime_error(
                    "full-material trace produced a non-finite value");
            const double baseline_trigger =
                collect_mirror_illuminance(options) ? 0.0 :
                options.threshold;
            if (luminance <= baseline_trigger)
                continue;
            result.rows[batch][ray] =
                result.work[batch].candidates.size();
            result.work[batch].candidates.push_back(candidate);
            ++result.selected;
        }
        result.work[batch].generated =
            result.work[batch].candidates.size();
    }
    return result;
}

typedef std::tuple<std::size_t, std::size_t, std::size_t, std::size_t,
                   unsigned int> RoughPilotKey;
typedef std::tuple<std::size_t, std::size_t, unsigned int, bool>
    RoughPilotPathKey;

RoughPilotKey rough_pilot_key(const Candidate &candidate)
{
    return RoughPilotKey(candidate.time_index, candidate.local_modifier,
                         candidate.view_index, candidate.seed_index,
                         candidate.reflection_orders);
}

RoughPilotPathKey rough_pilot_path_key(const Candidate &candidate)
{
    return RoughPilotPathKey(candidate.view_index, candidate.seed_index,
                             candidate.reflection_orders,
                             candidate.direct_sun);
}

std::set<std::size_t> rough_pilot_anchor_times(
    const std::vector<WorkChunk> &work, const std::vector<Sun> &suns,
    double angle_degrees)
{
    std::set<std::size_t> candidate_times;
    for (std::size_t batch = 0; batch < work.size(); ++batch)
        for (std::size_t seed = 0;
                seed < work[batch].candidates.size(); ++seed)
            candidate_times.insert(
                work[batch].candidates[seed].time_index);
    std::set<std::size_t> anchors;
    std::vector<Vec3> directions;
    const double cosine_limit = std::cos(angle_degrees*PI/180.0);
    for (std::set<std::size_t>::const_iterator time = candidate_times.begin();
            time != candidate_times.end(); ++time) {
        if (*time >= suns.size())
            throw std::runtime_error(
                "rough pilot candidate has an invalid time index");
        bool covered = false;
        for (std::size_t direction = 0; direction < directions.size();
                ++direction)
            if (dot(suns[*time].direction, directions[direction]) >=
                    cosine_limit) {
                covered = true;
                break;
            }
        if (!covered) {
            anchors.insert(*time);
            directions.push_back(suns[*time].direction);
        }
    }
    return anchors;
}

std::map<RoughPilotPathKey, std::vector<Vec3> > rough_pilot_hit_directions(
    const std::vector<WorkChunk> &work, const std::vector<Sun> &suns,
    const std::set<RoughPilotKey> &retained_keys)
{
    std::map<RoughPilotPathKey, std::vector<Vec3> > result;
    for (std::size_t batch = 0; batch < work.size(); ++batch)
        for (std::size_t seed = 0;
                seed < work[batch].candidates.size(); ++seed) {
            const Candidate &candidate = work[batch].candidates[seed];
            if (!retained_keys.count(rough_pilot_key(candidate)))
                continue;
            result[rough_pilot_path_key(candidate)].push_back(
                suns[candidate.time_index].direction);
        }
    return result;
}

bool rough_pilot_near_hit(
    const Candidate &candidate, const std::vector<Sun> &suns,
    const std::map<RoughPilotPathKey, std::vector<Vec3> > &hit_directions,
    double cosine_limit)
{
    const std::map<RoughPilotPathKey, std::vector<Vec3> >::const_iterator hit =
        hit_directions.find(rough_pilot_path_key(candidate));
    if (hit == hit_directions.end())
        return false;
    const Vec3 &sun_direction = suns[candidate.time_index].direction;
    for (std::size_t direction = 0; direction < hit->second.size();
            ++direction)
        if (dot(sun_direction, hit->second[direction]) >= cosine_limit)
            return true;
    return false;
}

struct RoughPilotStats {
    std::size_t seeds = 0;
    std::size_t center_retained = 0;
    std::size_t probe_retained = 0;
    std::size_t retained = 0;
    std::size_t rays = 0;
    std::set<RoughPilotKey> glare_keys;
    std::map<RoughPilotKey, std::vector<Vec3> > glare_directions;
};

RoughPilotStats prefilter_rough_seeds(
    std::vector<WorkChunk> &work, const std::vector<Sun> &suns,
    const std::vector<Viewpoint> &views, const Vec3 &up,
    const Options &options, BuiltinBackendHolder *builtin_backend)
{
    RoughPilotStats stats;
    if (options.rough_prefilter <= 0.0)
        return stats;

    const bool have_baseline = !options.nonspec_octree.empty();
    const bool sparse = options.rcontrib.empty();
    const double trigger = collect_mirror_illuminance(options) ? 0.0 :
        options.threshold*options.rough_prefilter;
    std::set<RoughPilotKey> retained_keys;

    const std::function<void(
        const std::vector<WorkChunk> &,
        std::vector<std::vector<float> > &,
        std::vector<std::vector<float> > &)>
        trace_pilot = [&](const std::vector<WorkChunk> &pilot_work,
                          std::vector<std::vector<float> > &values,
                          std::vector<std::vector<float> > &baselines) {
        values.resize(pilot_work.size());
        if (have_baseline)
            baselines.resize(pilot_work.size());
        if (!options.rcontrib.empty()) {
            for (std::size_t batch = 0; batch < pilot_work.size(); ++batch) {
                if (pilot_work[batch].candidates.empty())
                    continue;
                values[batch] = trace_external_values(
                    options.octree, pilot_work[batch].suns,
                    pilot_work[batch].candidates, views, options,
                    "external rcontrib rough pilot");
                if (have_baseline)
                    baselines[batch] = trace_external_values(
                        options.nonspec_octree, pilot_work[batch].suns,
                        pilot_work[batch].candidates, views, options,
                        "external rcontrib rough-pilot baseline");
            }
            return;
        }
#ifdef SPECULARCONTRIB_BUILTIN_RCONTRIB
        values = builtin_backend && builtin_backend->backend ?
            trace_builtin_batches(*builtin_backend->backend, pilot_work,
                                  views, options.origin_reuse) :
            trace_builtin_batches(options.octree, pilot_work, views, options);
        if (have_baseline)
            baselines = trace_builtin_batches(
                options.nonspec_octree, pilot_work, views, options);
#endif
    };

    const std::function<std::size_t(
        const std::vector<WorkChunk> &,
        const std::vector<std::vector<float> > &,
        const std::vector<std::vector<float> > &)>
        retain_hits = [&](const std::vector<WorkChunk> &pilot_work,
                          const std::vector<std::vector<float> > &values,
                          const std::vector<std::vector<float> > &baselines) {
        std::size_t added = 0;
        for (std::size_t batch = 0; batch < pilot_work.size(); ++batch)
            for (std::size_t ray = 0;
                    ray < pilot_work[batch].candidates.size(); ++ray) {
                const std::vector<float> *baseline = have_baseline ?
                    &baselines[batch] : NULL;
                const double luminance = candidate_luminance(
                    pilot_work[batch], pilot_work[batch].candidates[ray], ray,
                    values[batch], baseline, sparse);
                if (!std::isfinite(luminance))
                    throw std::runtime_error(
                        "rough pilot produced a non-finite value");
                const RoughPilotKey key = rough_pilot_key(
                    pilot_work[batch].candidates[ray]);
                if (luminance > options.threshold)
                    stats.glare_keys.insert(key);
                if (luminance > options.threshold ||
                        (collect_mirror_illuminance(options) &&
                         luminance > 0.0))
                    stats.glare_directions[key].push_back(
                        pilot_work[batch].candidates[ray].direction);
                if (luminance > trigger && retained_keys.insert(
                        rough_pilot_key(
                            pilot_work[batch].candidates[ray])).second)
                    ++added;
            }
        return added;
    };

    for (std::size_t batch = 0; batch < work.size(); ++batch) {
        stats.seeds += work[batch].candidates.size();
        stats.rays += work[batch].candidates.size();
    }
    std::vector<std::vector<float> > center_values;
    std::vector<std::vector<float> > center_baselines;
    trace_pilot(work, center_values, center_baselines);
    stats.center_retained = retain_hits(
        work, center_values, center_baselines);

    if (options.rough_pilot_samples > 1 &&
            retained_keys.size() < stats.seeds) {
        std::vector<int> stages;
        for (int samples = 16; samples < options.rough_pilot_samples;
                samples *= 4)
            stages.push_back(samples);
        if (stages.empty() || stages.back() != options.rough_pilot_samples)
            stages.push_back(options.rough_pilot_samples);

        const bool directional_guard =
            options.rough_pilot_guard_angle > 0.0;
        const double guard_cosine = directional_guard ? std::cos(
            options.rough_pilot_guard_angle*PI/180.0) : -1.0;
        const std::set<std::size_t> anchor_times = directional_guard ?
            rough_pilot_anchor_times(
                work, suns, options.rough_pilot_anchor_angle) :
            std::set<std::size_t>();

        for (std::size_t stage = 0; stage < stages.size() &&
                retained_keys.size() < stats.seeds; ++stage) {
            const bool guarded_stage = directional_guard &&
                stages[stage] > 16;
            const std::map<RoughPilotPathKey, std::vector<Vec3> >
                hit_directions = guarded_stage ?
                rough_pilot_hit_directions(work, suns, retained_keys) :
                std::map<RoughPilotPathKey, std::vector<Vec3> >();
            std::set<RoughPilotKey> stage_tested;
            std::vector<WorkChunk> probe_work(work.size());
            for (std::size_t batch = 0; batch < work.size(); ++batch) {
                probe_work[batch].suns = work[batch].suns;
                for (std::size_t seed = 0;
                        seed < work[batch].candidates.size(); ++seed) {
                    const Candidate &candidate =
                        work[batch].candidates[seed];
                    const RoughPilotKey key = rough_pilot_key(candidate);
                    if (retained_keys.count(key))
                        continue;
                    if (guarded_stage &&
                            !anchor_times.count(candidate.time_index) &&
                            !rough_pilot_near_hit(
                                candidate, suns, hit_directions,
                                guard_cosine))
                        continue;
                    probe_work[batch].candidates.push_back(candidate);
                    stage_tested.insert(key);
                }
                probe_work[batch].generated =
                    probe_work[batch].candidates.size();
            }
            Options probe_options = options;
            probe_options.rough_samples = stages[stage];
            expand_rough_candidates(
                probe_work, suns, views, up, probe_options);
            for (std::size_t batch = 0; batch < probe_work.size(); ++batch)
                stats.rays += probe_work[batch].candidates.size();
            std::vector<std::vector<float> > probe_values;
            std::vector<std::vector<float> > probe_baselines;
            trace_pilot(probe_work, probe_values, probe_baselines);
            stats.probe_retained += retain_hits(
                probe_work, probe_values, probe_baselines);

            if (guarded_stage) {
                const std::map<RoughPilotPathKey, std::vector<Vec3> >
                    expanded_hits = rough_pilot_hit_directions(
                        work, suns, retained_keys);
                std::vector<WorkChunk> neighbor_work(work.size());
                for (std::size_t batch = 0; batch < work.size(); ++batch) {
                    neighbor_work[batch].suns = work[batch].suns;
                    for (std::size_t seed = 0;
                            seed < work[batch].candidates.size(); ++seed) {
                        const Candidate &candidate =
                            work[batch].candidates[seed];
                        const RoughPilotKey key = rough_pilot_key(candidate);
                        if (retained_keys.count(key) ||
                                stage_tested.count(key) ||
                                !rough_pilot_near_hit(
                                    candidate, suns, expanded_hits,
                                    guard_cosine))
                            continue;
                        neighbor_work[batch].candidates.push_back(candidate);
                    }
                    neighbor_work[batch].generated =
                        neighbor_work[batch].candidates.size();
                }
                expand_rough_candidates(
                    neighbor_work, suns, views, up, probe_options);
                for (std::size_t batch = 0;
                        batch < neighbor_work.size(); ++batch)
                    stats.rays += neighbor_work[batch].candidates.size();
                std::vector<std::vector<float> > neighbor_values;
                std::vector<std::vector<float> > neighbor_baselines;
                trace_pilot(
                    neighbor_work, neighbor_values, neighbor_baselines);
                stats.probe_retained += retain_hits(
                    neighbor_work, neighbor_values, neighbor_baselines);
            }
        }
    }

    for (std::size_t batch = 0; batch < work.size(); ++batch) {
        std::vector<Candidate> retained;
        for (std::size_t ray = 0; ray < work[batch].candidates.size(); ++ray) {
            if (retained_keys.count(
                    rough_pilot_key(work[batch].candidates[ray]))) {
                retained.push_back(work[batch].candidates[ray]);
            }
        }
        work[batch].candidates.swap(retained);
    }
    stats.retained = retained_keys.size();
    return stats;
}

struct RoughStageResult {
    std::vector<double> contrast;
    std::vector<double> illuminance;
    std::size_t rays = 0;
};

RoughStageResult evaluate_rough_stage(
    const std::vector<WorkChunk> &seed_work, int samples,
    const std::vector<Sun> &suns, const std::vector<Viewpoint> &views,
    const Vec3 &up, const std::set<std::string> &mirror_materials,
    const std::set<std::string> &transparent_materials,
    const Options &options, BuiltinBackendHolder *builtin_backend)
{
    RoughStageResult result;
    result.contrast.assign(views.size()*suns.size(), 0.0);
    result.illuminance.assign(views.size()*suns.size(), 0.0);

    Options stage_options = options;
    stage_options.adaptive_rough_integration = false;
    stage_options.rough_samples = samples;
    std::vector<WorkChunk> work = seed_work;
    expand_rough_candidates(work, suns, views, up, stage_options);
    if (!stage_options.integrated_path_check) {
        const std::set<std::string> no_material_filter;
        const std::set<std::string> no_transparent_filter;
        filter_rough_work_hits(
            work, suns, views,
            stage_options.no_hit_check ? no_material_filter :
                mirror_materials,
            stage_options.no_hit_check ? no_transparent_filter :
                transparent_materials,
            stage_options);
    }
    for (std::size_t batch = 0; batch < work.size(); ++batch)
        result.rays += work[batch].candidates.size();

    if (!stage_options.rcontrib.empty()) {
        for (std::size_t batch = 0; batch < work.size(); ++batch)
            evaluate_external_chunk(
                work[batch], views, stage_options, result.contrast,
                result.illuminance, suns.size());
        return result;
    }
#ifdef SPECULARCONTRIB_BUILTIN_RCONTRIB
    const std::vector<std::vector<float> > values =
        builtin_backend && builtin_backend->backend ?
        trace_builtin_batches(*builtin_backend->backend, work, views,
                              stage_options.origin_reuse) :
        trace_builtin_batches(
            stage_options.octree, work, views, stage_options);
    std::vector<std::vector<float> > baseline_values;
    BaselineSelection baseline_selection;
    if (!stage_options.nonspec_octree.empty()) {
        baseline_selection = select_baseline_candidates(
            work, values, stage_options);
        baseline_values = trace_builtin_batches(
            stage_options.nonspec_octree, baseline_selection.work,
            views, stage_options);
    }
    for (std::size_t batch = 0; batch < work.size(); ++batch) {
        const std::vector<float> *baseline =
            baseline_values.empty() ? NULL : &baseline_values[batch];
        const std::vector<std::size_t> *baseline_rows =
            baseline_values.empty() ? NULL :
            &baseline_selection.rows[batch];
        accumulate_sparse_chunk(
            work[batch].candidates, values[batch], baseline, baseline_rows,
            stage_options, views, result.contrast, result.illuminance,
            suns.size());
    }
#endif
    return result;
}

bool rough_stage_converged(double coarse_contrast, double medium_contrast,
                           double coarse_illuminance,
                           double medium_illuminance,
                           bool pilot_found_glare, const Options &options)
{
    const double contrast_scale = std::max(
        std::fabs(coarse_contrast), std::fabs(medium_contrast));
    if (contrast_scale <= kEpsilon) {
        if (pilot_found_glare)
            return false;
    } else if (std::min(coarse_contrast, medium_contrast) <= 0.0 ||
               std::fabs(medium_contrast-coarse_contrast) >
                   options.rough_convergence*contrast_scale) {
        return false;
    }

    if (!collect_mirror_illuminance(options))
        return true;
    const double illuminance_difference = std::fabs(
        medium_illuminance-coarse_illuminance);
    const double illuminance_scale = std::max(
        std::fabs(coarse_illuminance), std::fabs(medium_illuminance));
    return illuminance_difference <=
        options.rough_illuminance_tolerance ||
        (illuminance_scale > kEpsilon && illuminance_difference <=
         options.rough_convergence*illuminance_scale);
}

void evaluate_adaptive_rough_integration(
    const std::vector<WorkChunk> &seed_work,
    const RoughPilotStats &pilot, const std::vector<Sun> &suns,
    const std::vector<Viewpoint> &views, const Vec3 &up,
    const std::set<std::string> &mirror_materials,
    const std::set<std::string> &transparent_materials,
    const Options &options, BuiltinBackendHolder *builtin_backend,
    std::vector<double> &contrast, std::vector<double> &illuminance)
{
    const RoughStageResult coarse = evaluate_rough_stage(
        seed_work, options.rough_coarse_samples, suns, views, up,
        mirror_materials, transparent_materials, options, builtin_backend);
    const RoughStageResult medium = evaluate_rough_stage(
        seed_work, options.rough_medium_samples, suns, views, up,
        mirror_materials, transparent_materials, options, builtin_backend);

    std::vector<char> pilot_glare(contrast.size(), 0);
    for (std::size_t batch = 0; batch < seed_work.size(); ++batch)
        for (std::size_t seed = 0;
                seed < seed_work[batch].candidates.size(); ++seed) {
            const Candidate &candidate = seed_work[batch].candidates[seed];
            if (pilot.glare_keys.count(rough_pilot_key(candidate)))
                pilot_glare[candidate.view_index*suns.size()+
                            candidate.time_index] = 1;
        }

    std::vector<char> converged(contrast.size(), 0);
    std::vector<char> visited(contrast.size(), 0);
    std::size_t converged_outputs = 0;
    std::size_t active_outputs = 0;
    for (std::size_t batch = 0; batch < seed_work.size(); ++batch)
        for (std::size_t seed = 0;
                seed < seed_work[batch].candidates.size(); ++seed) {
            const Candidate &candidate = seed_work[batch].candidates[seed];
            const std::size_t output = candidate.view_index*suns.size()+
                                       candidate.time_index;
            if (visited[output])
                continue;
            visited[output] = 1;
            ++active_outputs;
            if (rough_stage_converged(
                    coarse.contrast[output], medium.contrast[output],
                    coarse.illuminance[output], medium.illuminance[output],
                    pilot_glare[output] != 0, options)) {
                converged[output] = 1;
                ++converged_outputs;
                contrast[output] += medium.contrast[output];
                illuminance[output] += medium.illuminance[output];
            }
        }

    std::vector<WorkChunk> unresolved(seed_work.size());
    for (std::size_t batch = 0; batch < seed_work.size(); ++batch) {
        unresolved[batch].suns = seed_work[batch].suns;
        for (std::size_t seed = 0;
                seed < seed_work[batch].candidates.size(); ++seed) {
            const Candidate &candidate = seed_work[batch].candidates[seed];
            const std::size_t output = candidate.view_index*suns.size()+
                                       candidate.time_index;
            if (!converged[output])
                unresolved[batch].candidates.push_back(candidate);
        }
        unresolved[batch].generated = unresolved[batch].candidates.size();
    }

    RoughStageResult fine;
    if (converged_outputs < active_outputs) {
        fine = evaluate_rough_stage(
            unresolved, options.rough_samples, suns, views, up,
            mirror_materials, transparent_materials, options,
            builtin_backend);
        for (std::size_t output = 0; output < contrast.size(); ++output)
            if (!converged[output]) {
                contrast[output] += fine.contrast[output];
                illuminance[output] += fine.illuminance[output];
            }
    }
    if (!options.quiet)
        std::fprintf(stderr,
            "%s: adaptive rough integration accepted %lu/%lu outputs at "
            "%d samples; traced %lu + %lu + %lu rays\n",
            progname, static_cast<unsigned long>(converged_outputs),
            static_cast<unsigned long>(active_outputs),
            options.rough_medium_samples,
            static_cast<unsigned long>(coarse.rays),
            static_cast<unsigned long>(medium.rays),
            static_cast<unsigned long>(fine.rays));
}

struct RoughRawStage {
    std::vector<WorkChunk> work;
    std::vector<std::vector<float> > values;
    std::size_t rays = 0;
};

struct RoughPilotCellMatch {
    double cosine = -2.0;
    std::size_t output = 0;
    int cell = -1;
};

RoughRawStage trace_rough_stage_raw(
    const std::vector<WorkChunk> &seed_work, int samples,
    const std::vector<Sun> &suns, const std::vector<Viewpoint> &views,
    const Vec3 &up, const std::set<std::string> &mirror_materials,
    const std::set<std::string> &transparent_materials,
    const Options &options, BuiltinBackendHolder *builtin_backend)
{
    if (!options.rcontrib.empty() || !options.nonspec_octree.empty())
        throw std::runtime_error(
            "adaptive rough-cell integration requires the built-in backend "
            "without --nonspec-octree");
    RoughRawStage result;
    result.work = seed_work;
    Options stage_options = options;
    stage_options.adaptive_rough_cells = false;
    stage_options.rough_samples = samples;
    expand_rough_candidates(result.work, suns, views, up, stage_options);
    if (!stage_options.integrated_path_check) {
        const std::set<std::string> no_material_filter;
        const std::set<std::string> no_transparent_filter;
        filter_rough_work_hits(
            result.work, suns, views,
            stage_options.no_hit_check ? no_material_filter :
                mirror_materials,
            stage_options.no_hit_check ? no_transparent_filter :
                transparent_materials,
            stage_options);
    }
    for (std::size_t batch = 0; batch < result.work.size(); ++batch)
        result.rays += result.work[batch].candidates.size();
#ifdef SPECULARCONTRIB_BUILTIN_RCONTRIB
    result.values = builtin_backend && builtin_backend->backend ?
        trace_builtin_batches(
            *builtin_backend->backend, result.work, views,
            stage_options.origin_reuse) :
        trace_builtin_batches(
            stage_options.octree, result.work, views, stage_options);
#endif
    return result;
}

void evaluate_adaptive_rough_cells(
    const std::vector<WorkChunk> &seed_work,
    const RoughPilotStats &pilot, const std::vector<Sun> &suns,
    const std::vector<Viewpoint> &views, const Vec3 &up,
    const std::set<std::string> &mirror_materials,
    const std::set<std::string> &transparent_materials,
    const Options &options, BuiltinBackendHolder *builtin_backend,
    std::vector<double> &contrast, std::vector<double> &illuminance)
{
    const int medium_samples = options.rough_medium_samples;
    const int medium_side = static_cast<int>(std::sqrt(
        static_cast<double>(medium_samples)));
    const int fine_side = static_cast<int>(std::sqrt(
        static_cast<double>(options.rough_samples)));
    const RoughRawStage medium = trace_rough_stage_raw(
        seed_work, medium_samples, suns, views, up, mirror_materials,
        transparent_materials, options, builtin_backend);

    const std::size_t output_count = contrast.size();
    std::vector<int> seed_count(output_count, 0);
    std::vector<char> pilot_glare(output_count, 0);
    std::vector<char> fallback(output_count, 0);
    for (std::size_t batch = 0; batch < seed_work.size(); ++batch)
        for (std::size_t seed = 0;
                seed < seed_work[batch].candidates.size(); ++seed) {
            const Candidate &candidate = seed_work[batch].candidates[seed];
            const std::size_t output = candidate.view_index*suns.size()+
                                       candidate.time_index;
            ++seed_count[output];
            if (pilot.glare_keys.count(rough_pilot_key(candidate)))
                pilot_glare[output] = 1;
            const double center_angle = std::acos(std::max(
                -1.0, std::min(1.0, dot(
                    candidate.direction,
                    views[candidate.view_index].direction))));
            if (center_angle+rough_cap_radius(
                    candidate, suns[candidate.time_index], options) >=
                    PI/2.0-kEpsilon)
                fallback[output] = 1;
        }
    for (std::size_t output = 0; output < output_count; ++output)
        if (seed_count[output] > 1)
            fallback[output] = 1;

    std::vector<std::set<int> > active_cells(output_count);
    std::map<RoughPilotKey, std::vector<RoughPilotCellMatch> > pilot_matches;
    for (std::map<RoughPilotKey, std::vector<Vec3> >::const_iterator hit =
            pilot.glare_directions.begin();
            hit != pilot.glare_directions.end(); ++hit)
        pilot_matches[hit->first].resize(hit->second.size());
    const double cell_trigger = options.threshold*options.rough_cell_trigger;
    for (std::size_t batch = 0; batch < medium.work.size(); ++batch)
        for (std::size_t ray = 0;
                ray < medium.work[batch].candidates.size(); ++ray) {
            const Candidate &candidate =
                medium.work[batch].candidates[ray];
            const RoughPilotKey pilot_key = rough_pilot_key(candidate);
            const std::map<RoughPilotKey, std::vector<Vec3> >::const_iterator
                pilot_hit = pilot.glare_directions.find(pilot_key);
            std::map<RoughPilotKey,
                     std::vector<RoughPilotCellMatch> >::iterator matches =
                pilot_matches.find(pilot_key);
            if (pilot_hit != pilot.glare_directions.end() &&
                    matches != pilot_matches.end())
                for (std::size_t hit = 0;
                        hit < pilot_hit->second.size(); ++hit) {
                    const double cosine = dot(
                        candidate.direction, pilot_hit->second[hit]);
                    if (cosine > matches->second[hit].cosine) {
                        matches->second[hit].cosine = cosine;
                        matches->second[hit].output =
                            candidate.view_index*suns.size()+
                            candidate.time_index;
                        matches->second[hit].cell =
                            candidate.rough_row*medium_side+
                            candidate.rough_column;
                    }
                }
            const double luminance = candidate_luminance(
                medium.work[batch], candidate, ray, medium.values[batch],
                NULL, true);
            const std::size_t output = candidate.view_index*suns.size()+
                                       candidate.time_index;
            if (luminance <= cell_trigger)
                continue;
            for (int row = std::max(0, candidate.rough_row-1);
                    row <= std::min(
                        medium_side-1, candidate.rough_row+1); ++row)
                for (int column = std::max(
                         0, candidate.rough_column-1);
                        column <= std::min(
                            medium_side-1,
                            candidate.rough_column+1); ++column)
                    active_cells[output].insert(row*medium_side+column);
        }
    for (std::map<RoughPilotKey,
                  std::vector<RoughPilotCellMatch> >::const_iterator path =
            pilot_matches.begin(); path != pilot_matches.end(); ++path)
        for (std::size_t hit = 0; hit < path->second.size(); ++hit) {
            const RoughPilotCellMatch &match = path->second[hit];
            if (match.cell < 0)
                continue;
            const int center_row = match.cell/medium_side;
            const int center_column = match.cell%medium_side;
            for (int row = std::max(0, center_row-1);
                    row <= std::min(medium_side-1, center_row+1); ++row)
                for (int column = std::max(0, center_column-1);
                        column <= std::min(
                            medium_side-1, center_column+1); ++column)
                    active_cells[match.output].insert(
                        row*medium_side+column);
        }
    std::size_t fallback_outputs = 0;
    std::size_t refined_cells = 0;
    for (std::size_t output = 0; output < output_count; ++output) {
        if (!seed_count[output])
            continue;
        if ((pilot_glare[output] && active_cells[output].size() <= 27) ||
                active_cells[output].size() >
                    static_cast<std::size_t>(medium_samples/2))
            fallback[output] = 1;
        if (fallback[output])
            ++fallback_outputs;
        else
            refined_cells += active_cells[output].size();
    }

    std::map<RoughSourceKey, RoughSourceAccumulator> rough_sources;
    for (std::size_t batch = 0; batch < medium.work.size(); ++batch)
        for (std::size_t ray = 0;
                ray < medium.work[batch].candidates.size(); ++ray) {
            const Candidate &candidate =
                medium.work[batch].candidates[ray];
            const std::size_t output = candidate.view_index*suns.size()+
                                       candidate.time_index;
            const int cell = candidate.rough_row*medium_side+
                             candidate.rough_column;
            if (fallback[output] || active_cells[output].count(cell))
                continue;
            const double luminance = candidate_luminance(
                medium.work[batch], candidate, ray, medium.values[batch],
                NULL, true);
            accumulate_candidate_outputs(
                candidate, luminance, views, options, contrast, illuminance,
                suns.size(), &rough_sources);
        }

    Options fine_options = options;
    fine_options.adaptive_rough_cells = false;
    std::vector<WorkChunk> fine_work = seed_work;
    expand_rough_candidates(
        fine_work, suns, views, up, fine_options);
    for (std::size_t batch = 0; batch < fine_work.size(); ++batch) {
        std::vector<Candidate> selected;
        for (std::size_t ray = 0;
                ray < fine_work[batch].candidates.size(); ++ray) {
            const Candidate &candidate = fine_work[batch].candidates[ray];
            const std::size_t output = candidate.view_index*suns.size()+
                                       candidate.time_index;
            const int medium_row = candidate.rough_row*medium_side/fine_side;
            const int medium_column =
                candidate.rough_column*medium_side/fine_side;
            if (fallback[output] || active_cells[output].count(
                    medium_row*medium_side+medium_column))
                selected.push_back(candidate);
        }
        fine_work[batch].candidates.swap(selected);
        fine_work[batch].generated = fine_work[batch].candidates.size();
    }
    if (!fine_options.integrated_path_check) {
        const std::set<std::string> no_material_filter;
        const std::set<std::string> no_transparent_filter;
        filter_rough_work_hits(
            fine_work, suns, views,
            fine_options.no_hit_check ? no_material_filter :
                mirror_materials,
            fine_options.no_hit_check ? no_transparent_filter :
                transparent_materials,
            fine_options);
    }
    std::size_t fine_rays = 0;
    for (std::size_t batch = 0; batch < fine_work.size(); ++batch)
        fine_rays += fine_work[batch].candidates.size();
#ifdef SPECULARCONTRIB_BUILTIN_RCONTRIB
    const std::vector<std::vector<float> > fine_values =
        builtin_backend && builtin_backend->backend ?
        trace_builtin_batches(
            *builtin_backend->backend, fine_work, views,
            fine_options.origin_reuse) :
        trace_builtin_batches(
            fine_options.octree, fine_work, views, fine_options);
    for (std::size_t batch = 0; batch < fine_work.size(); ++batch)
        for (std::size_t ray = 0;
                ray < fine_work[batch].candidates.size(); ++ray) {
            const Candidate &candidate = fine_work[batch].candidates[ray];
            const double luminance = candidate_luminance(
                fine_work[batch], candidate, ray, fine_values[batch],
                NULL, true);
            accumulate_candidate_outputs(
                candidate, luminance, views, options, contrast, illuminance,
                suns.size(), &rough_sources);
        }
#endif
    flush_rough_sources(
        rough_sources, views, options, contrast, suns.size());
    if (!options.quiet)
        std::fprintf(stderr,
            "%s: adaptive rough cells used %lu medium rays and %lu fine "
            "rays; refined %lu cells and fully resolved %lu outputs\n",
            progname, static_cast<unsigned long>(medium.rays),
            static_cast<unsigned long>(fine_rays),
            static_cast<unsigned long>(refined_cells),
            static_cast<unsigned long>(fallback_outputs));
}

struct SolarDiskPilotStats {
    std::size_t seeds;
    std::size_t pilot_retained;
    std::size_t retained;
    std::size_t rays;
};

struct SolarDiskPathKey {
    std::size_t view_index;
    std::size_t seed_index;
    unsigned int reflection_orders;
    bool direct_sun;

    bool operator<(const SolarDiskPathKey &other) const
    {
        if (view_index != other.view_index)
            return view_index < other.view_index;
        if (seed_index != other.seed_index)
            return seed_index < other.seed_index;
        if (reflection_orders != other.reflection_orders)
            return reflection_orders < other.reflection_orders;
        return direct_sun < other.direct_sun;
    }
};

struct SolarDiskSeedRef {
    std::size_t batch;
    std::size_t seed;
};

SolarDiskPilotStats prefilter_solar_disk_seeds(
    std::vector<WorkChunk> &work, const std::vector<Viewpoint> &views,
    const Vec3 &up, const Options &options,
    BuiltinBackendHolder *builtin_backend)
{
    Options pilot_options = options;
    pilot_options.adaptive_sun_disk = false;
    pilot_options.sun_disk_samples = options.sun_disk_pilot_samples;
    pilot_options.secondary_sun_disk_samples =
        options.secondary_sun_disk_pilot_samples;
    std::vector<WorkChunk> pilot_work = work;
    expand_solar_disk_candidates(pilot_work, views, up, pilot_options);

    SolarDiskPilotStats stats = {0, 0, 0, 0};
    for (std::size_t batch = 0; batch < work.size(); ++batch) {
        stats.seeds += work[batch].candidates.size();
        stats.rays += pilot_work[batch].candidates.size();
    }

#ifdef SPECULARCONTRIB_BUILTIN_RCONTRIB
    const std::vector<std::vector<float> > values =
        builtin_backend && builtin_backend->backend ?
        trace_builtin_batches(*builtin_backend->backend,
                              pilot_work, views, options.origin_reuse) :
        trace_builtin_batches(options.octree, pilot_work, views, options);
    std::vector<std::vector<float> > baselines;
    if (!options.nonspec_octree.empty())
        baselines = trace_builtin_batches(
            options.nonspec_octree, pilot_work, views, options);

    std::vector<std::vector<char> > retained(work.size());
    for (std::size_t batch = 0; batch < work.size(); ++batch) {
        retained[batch].assign(work[batch].candidates.size(), 0);
        for (std::size_t ray = 0;
                ray < pilot_work[batch].candidates.size(); ++ray) {
            const Candidate &candidate =
                pilot_work[batch].candidates[ray];
            if (candidate.disk_seed_index >= retained[batch].size())
                throw std::runtime_error(
                    "solar-disk pilot has an invalid seed index");
            const std::vector<float> *baseline =
                baselines.empty() || candidate.direct_sun ? NULL :
                &baselines[batch];
            const double luminance = candidate_luminance(
                pilot_work[batch], candidate, ray, values[batch],
                baseline, true);
            if (!std::isfinite(luminance))
                throw std::runtime_error(
                    "solar-disk pilot produced a non-finite value");
            if (luminance > options.threshold ||
                    (collect_mirror_illuminance(options) &&
                     !candidate.direct_sun && luminance > 0.0))
                retained[batch][candidate.disk_seed_index] = 1;
        }
        stats.pilot_retained += static_cast<std::size_t>(std::count(
            retained[batch].begin(), retained[batch].end(), 1));
    }

    if (options.sun_disk_pilot_guard_angle > 0.0) {
        typedef std::map<SolarDiskPathKey,
                         std::vector<SolarDiskSeedRef> > PathGroups;
        PathGroups groups;
        for (std::size_t batch = 0; batch < work.size(); ++batch)
            for (std::size_t seed = 0;
                    seed < work[batch].candidates.size(); ++seed) {
                const Candidate &candidate = work[batch].candidates[seed];
                const SolarDiskPathKey key = {
                    candidate.view_index, candidate.seed_index,
                    candidate.reflection_orders, candidate.direct_sun
                };
                groups[key].push_back(SolarDiskSeedRef{batch, seed});
            }

        const double guard = options.sun_disk_pilot_guard_angle*PI/180.0;
        // Protect neighboring sun directions only along the same reflection path.
        for (PathGroups::const_iterator group = groups.begin();
                group != groups.end(); ++group) {
            std::vector<Vec3> hit_directions;
            for (std::size_t index = 0;
                    index < group->second.size(); ++index) {
                const SolarDiskSeedRef &ref = group->second[index];
                if (retained[ref.batch][ref.seed])
                    hit_directions.push_back(
                        work[ref.batch].candidates[ref.seed].direction);
            }
            if (hit_directions.empty())
                continue;
            for (std::size_t index = 0;
                    index < group->second.size(); ++index) {
                const SolarDiskSeedRef &ref = group->second[index];
                if (retained[ref.batch][ref.seed])
                    continue;
                const Vec3 &direction =
                    work[ref.batch].candidates[ref.seed].direction;
                for (std::size_t hit = 0;
                        hit < hit_directions.size(); ++hit)
                    if (angle(direction, hit_directions[hit]) <= guard) {
                        retained[ref.batch][ref.seed] = 1;
                        break;
                    }
            }
        }
    }

    for (std::size_t batch = 0; batch < work.size(); ++batch) {
        std::vector<Candidate> selected;
        for (std::size_t seed = 0; seed < retained[batch].size(); ++seed)
            if (retained[batch][seed]) {
                selected.push_back(work[batch].candidates[seed]);
                ++stats.retained;
            }
        work[batch].candidates.swap(selected);
        work[batch].generated = work[batch].candidates.size();
    }
#else
    (void)views;
    (void)options;
    (void)builtin_backend;
#endif
    return stats;
}

void evaluate_sun_subset(
    const std::vector<std::size_t> &time_indices,
    const std::vector<Sun> &suns, const std::vector<Viewpoint> &views,
    const ReflectionData &reflections, const Vec3 &up,
    const std::set<std::string> &mirror_materials,
    const std::set<std::string> &transparent_materials,
    const Options &options, std::size_t first_view,
    std::size_t last_view, BuiltinBackendHolder *builtin_backend,
    std::vector<double> &contrast, std::vector<double> &illuminance)
{
    if (time_indices.empty())
        return;
    std::vector<std::pair<std::size_t, Sun> > selected_suns;
    selected_suns.reserve(time_indices.size());
    for (std::size_t i = 0; i < time_indices.size(); ++i) {
        if (time_indices[i] >= suns.size() || !suns[time_indices[i]].active)
            throw std::runtime_error(
                "invalid active solar record selected for evaluation");
        selected_suns.push_back(std::make_pair(
            time_indices[i], suns[time_indices[i]]));
    }

    std::vector<WorkChunk> work;
    for (std::size_t first = 0; first < selected_suns.size();
            first += options.batch_size) {
        const std::size_t last = std::min(
            selected_suns.size(), first+
            static_cast<std::size_t>(options.batch_size));
        WorkChunk batch;
        batch.suns = std::vector<std::pair<std::size_t, Sun> >(
            selected_suns.begin()+first, selected_suns.begin()+last);
        batch.candidates = build_candidates(
            batch.suns, views, reflections, up, first_view, last_view);
        if (options.include_direct_sun)
            append_direct_sun_candidates(
                batch.candidates, batch.suns, views, up,
                first_view, last_view);
        batch.generated = batch.candidates.size();
        work.push_back(batch);
    }

    const bool filter_specular_paths = !options.integrated_path_check &&
        (options.max_specular_bounces > 1 ||
         (!options.no_hit_check && !mirror_materials.empty()));
    if (options.adaptive_sun_disk) {
        const SolarDiskPilotStats pilot = prefilter_solar_disk_seeds(
            work, views, up, options, builtin_backend);
        if (!options.quiet)
            std::fprintf(stderr,
                "%s: solar-disk pilot retained %lu paths and protected "
                "%lu/%lu paths from %lu rays\n",
                progname, static_cast<unsigned long>(pilot.pilot_retained),
                static_cast<unsigned long>(pilot.retained),
                static_cast<unsigned long>(pilot.seeds),
                static_cast<unsigned long>(pilot.rays));
    }
    expand_solar_disk_candidates(work, views, up, options);
    if (filter_specular_paths) {
        const std::set<std::string> no_material_filter;
        const std::set<std::string> no_transparent_filter;
        filter_work_paths(work, suns, views,
                          options.no_hit_check ? no_material_filter :
                          mirror_materials,
                          options.no_hit_check ? no_transparent_filter :
                          transparent_materials, options);
    }
    RoughPilotStats rough_pilot;
    if (options.rough_samples > 0) {
        if (options.rough_prefilter > 0.0) {
            rough_pilot = prefilter_rough_seeds(
                work, suns, views, up, options, builtin_backend);
            if (!options.quiet)
                std::fprintf(stderr,
                    "%s: rough pilot retained %lu/%lu paths "
                    "(%lu center, %lu probe) above %.3g of the glare "
                    "threshold using %lu rays\n", progname,
                    static_cast<unsigned long>(rough_pilot.retained),
                    static_cast<unsigned long>(rough_pilot.seeds),
                    static_cast<unsigned long>(rough_pilot.center_retained),
                    static_cast<unsigned long>(rough_pilot.probe_retained),
                    options.rough_prefilter,
                    static_cast<unsigned long>(rough_pilot.rays));
        }
        if (options.adaptive_rough_cells) {
            evaluate_adaptive_rough_cells(
                work, rough_pilot, suns, views, up, mirror_materials,
                transparent_materials, options, builtin_backend,
                contrast, illuminance);
            return;
        }
        if (options.adaptive_rough_integration) {
            evaluate_adaptive_rough_integration(
                work, rough_pilot, suns, views, up, mirror_materials,
                transparent_materials, options, builtin_backend,
                contrast, illuminance);
            return;
        }
        expand_rough_candidates(work, suns, views, up, options);
        if (!options.integrated_path_check) {
            const std::set<std::string> no_material_filter;
            const std::set<std::string> no_transparent_filter;
            filter_rough_work_hits(work, suns, views,
                                   options.no_hit_check ? no_material_filter :
                                   mirror_materials,
                                   options.no_hit_check ? no_transparent_filter :
                                   transparent_materials, options);
        }
    }

    if (!options.rcontrib.empty()) {
        for (std::size_t batch = 0; batch < work.size(); ++batch)
            evaluate_external_chunk(work[batch], views, options,
                                    contrast, illuminance, suns.size());
        return;
    }
#ifdef SPECULARCONTRIB_BUILTIN_RCONTRIB
    const std::vector<std::vector<float> > values =
        builtin_backend && builtin_backend->backend ?
        trace_builtin_batches(*builtin_backend->backend, work, views,
                              options.origin_reuse) :
        trace_builtin_batches(options.octree, work, views, options);
    std::vector<std::vector<float> > baseline_values;
    BaselineSelection baseline_selection;
    if (!options.nonspec_octree.empty()) {
        baseline_selection = select_baseline_candidates(
            work, values, options);
        baseline_values = trace_builtin_batches(
            options.nonspec_octree, baseline_selection.work,
            views, options);
        if (!options.quiet)
            std::fprintf(stderr,
                "%s: non-specular baseline retained %lu relevant rays\n",
                progname,
                static_cast<unsigned long>(baseline_selection.selected));
    }
    for (std::size_t batch = 0; batch < work.size(); ++batch) {
        const std::vector<float> *baseline =
            baseline_values.empty() ? NULL : &baseline_values[batch];
        const std::vector<std::size_t> *baseline_rows =
            baseline_values.empty() ? NULL :
            &baseline_selection.rows[batch];
        accumulate_sparse_chunk(
            work[batch].candidates, values[batch], baseline,
            baseline_rows, options, views, contrast, illuminance,
            suns.size());
    }
#endif
}

#ifdef SPECULARCONTRIB_BUILTIN_RCONTRIB
struct AutoMaterialPlan {
    std::set<std::string> ideal_materials;
    std::map<long long, std::set<std::string> > rough_materials;
    std::map<std::string, long long> material_group;
    std::set<std::string> proposal_materials;
    std::vector<SpecularMaterialInfo> ignored_rough_materials;
    std::vector<SpecularMaterialInfo> excluded_materials;
    std::vector<SpecularMaterialInfo> unsupported_materials;
};

struct AutoReflectionPhase {
    std::string label;
    std::set<std::string> materials;
    ReflectionData reflections;
    double roughness = 0.0;
    bool rough = false;
    bool direct_sun = false;
};

long long roughness_group_key(double roughness)
{
    const double bounded = std::max(1.0e-6, std::min(1.0, roughness));
    return static_cast<long long>(std::floor(bounded*1.0e6+0.5));
}

double roughness_from_group_key(long long key)
{
    return static_cast<double>(key)/1.0e6;
}

AutoMaterialPlan make_auto_material_plan(
    const std::vector<SpecularMaterialInfo> &materials, bool ideal_only)
{
    AutoMaterialPlan plan;
    for (std::size_t i = 0; i < materials.size(); ++i) {
        const SpecularMaterialInfo &material = materials[i];
        if (material.material_class == SPECULAR_MATERIAL_IDEAL) {
            plan.ideal_materials.insert(material.name);
            plan.material_group[material.name] = 0;
            plan.proposal_materials.insert(material.name);
        } else if (material.material_class == SPECULAR_MATERIAL_ROUGH) {
            if (ideal_only) {
                plan.ignored_rough_materials.push_back(material);
                continue;
            }
            const long long key = roughness_group_key(material.roughness);
            plan.rough_materials[key].insert(material.name);
            plan.material_group[material.name] = key;
            plan.proposal_materials.insert(material.name);
        } else if (material.material_class == SPECULAR_MATERIAL_EXCLUDED) {
            plan.excluded_materials.push_back(material);
        } else if (material.material_class ==
                SPECULAR_MATERIAL_UNSUPPORTED) {
            plan.unsupported_materials.push_back(material);
        }
    }
    return plan;
}

long long material_group_for_path(const AutoMaterialPlan &plan,
                                  const std::string &material)
{
    const std::map<std::string, long long>::const_iterator found =
        plan.material_group.find(material);
    return found == plan.material_group.end() ? -1 : found->second;
}

long long material_group_for_path(const AutoMaterialPlan &plan,
                                  const std::string &material,
                                  const std::string &modifier)
{
    const long long material_group = material_group_for_path(plan, material);
    return material_group >= 0 ? material_group :
        material_group_for_path(plan, modifier);
}

bool normal_source_matches_materials(
    const NormalSource &source, const std::set<std::string> &materials)
{
    return materials.empty() || materials.count(source.material) != 0 ||
        materials.count(source.modifier) != 0;
}

ReflectionData first_order_reflections_for_materials(
    const ReflectionData &input, const std::set<std::string> &materials)
{
    ReflectionData output;
    std::vector<std::size_t> output_indices(
        input.first_order_normals.size(), kAllViews);

    const auto add_normal_source = [&](std::size_t normal_index,
                                       const NormalSource *source) {
        std::size_t &output_index = output_indices[normal_index];
        if (output_index == kAllViews) {
            output_index = output.first_order_normals.size();
            output.first_order_normals.push_back(
                input.first_order_normals[normal_index]);
            output.first_order_sources.push_back(
                std::vector<NormalSource>());
        }
        if (!source)
            return;
        std::vector<NormalSource> &sources =
            output.first_order_sources[output_index];
        for (std::size_t i = 0; i < sources.size(); ++i)
            if (same_normal_source(sources[i], *source))
                return;
        sources.push_back(*source);
    };

    for (std::size_t normal = 0;
            normal < input.first_order_normals.size(); ++normal) {
        const std::vector<NormalSource> *sources =
            normal < input.first_order_sources.size() ?
            &input.first_order_sources[normal] : NULL;
        if (!sources || sources->empty()) {
            add_normal_source(normal, NULL);
            continue;
        }
        for (std::size_t source = 0; source < sources->size(); ++source) {
            const NormalSource &candidate = (*sources)[source];
            if (!normal_source_matches_materials(candidate, materials))
                continue;
            add_normal_source(normal, &candidate);
        }
    }

    // Candidate indices participate in deterministic sampling seeds.  Keep
    // the discovery order after filtering material-specific sources.
    ReflectionData ordered;
    for (std::size_t normal = 0; normal < output_indices.size(); ++normal) {
        const std::size_t output_index = output_indices[normal];
        if (output_index == kAllViews)
            continue;
        ordered.first_order_normals.push_back(
            input.first_order_normals[normal]);
        ordered.first_order_sources.push_back(
            output.first_order_sources[output_index]);
    }
    return ordered;
}

std::vector<AutoReflectionPhase> make_auto_reflection_phases(
    const AutoMaterialPlan &plan, const ReflectionData &reflections,
    const Options &options)
{
    std::vector<AutoReflectionPhase> phases;
    if (options.include_direct_sun) {
        AutoReflectionPhase direct;
        direct.label = "direct sun";
        direct.direct_sun = true;
        phases.push_back(direct);
    }

    if (!plan.ideal_materials.empty()) {
        AutoReflectionPhase ideal;
        ideal.label = "ideal reflection";
        ideal.materials = plan.ideal_materials;
        ideal.reflections = first_order_reflections_for_materials(
            reflections, ideal.materials);
        for (std::size_t i = 0;
                i < reflections.second_order_paths.size(); ++i) {
            const ReflectionPath2 &path = reflections.second_order_paths[i];
            if (material_group_for_path(
                    plan, path.first_material, path.first_modifier) == 0 &&
                    material_group_for_path(
                    plan, path.second_material, path.second_modifier) == 0)
                ideal.reflections.second_order_paths.push_back(path);
        }
        phases.push_back(ideal);
    }

    for (std::map<long long, std::set<std::string> >::const_iterator group =
            plan.rough_materials.begin();
            group != plan.rough_materials.end(); ++group) {
        AutoReflectionPhase rough;
        std::ostringstream label;
        label << "rough reflection alpha=" <<
            roughness_from_group_key(group->first);
        rough.label = label.str();
        rough.materials = group->second;
        rough.reflections = first_order_reflections_for_materials(
            reflections, rough.materials);
        rough.roughness = roughness_from_group_key(group->first);
        rough.rough = true;
        for (std::size_t i = 0;
                i < reflections.second_order_paths.size(); ++i) {
            const ReflectionPath2 &path = reflections.second_order_paths[i];
            if (material_group_for_path(
                    plan, path.first_material, path.first_modifier) ==
                    group->first &&
                    material_group_for_path(
                    plan, path.second_material, path.second_modifier) ==
                    group->first)
                rough.reflections.second_order_paths.push_back(path);
        }
        phases.push_back(rough);
    }

    std::map<long long, AutoReflectionPhase> mixed_phases;
    for (std::size_t i = 0; i < reflections.second_order_paths.size(); ++i) {
        const ReflectionPath2 &path = reflections.second_order_paths[i];
        const long long first_group = material_group_for_path(
            plan, path.first_material, path.first_modifier);
        const long long second_group = material_group_for_path(
            plan, path.second_material, path.second_modifier);
        if (first_group < 0 || second_group < 0 ||
                first_group == second_group)
            continue;
        const double first_alpha = first_group > 0 ?
            roughness_from_group_key(first_group) : 0.0;
        const double second_alpha = second_group > 0 ?
            roughness_from_group_key(second_group) : 0.0;
        const long long mixed_key = roughness_group_key(std::sqrt(
            first_alpha*first_alpha+second_alpha*second_alpha));
        AutoReflectionPhase &mixed = mixed_phases[mixed_key];
        mixed.rough = true;
        mixed.roughness = roughness_from_group_key(mixed_key);
        mixed.materials.insert(path.first_material);
        mixed.materials.insert(path.second_material);
        mixed.reflections.second_order_paths.push_back(path);
    }
    for (std::map<long long, AutoReflectionPhase>::iterator mixed =
            mixed_phases.begin(); mixed != mixed_phases.end(); ++mixed) {
        std::ostringstream label;
        label << "mixed 2R reflection alpha=" << mixed->second.roughness;
        mixed->second.label = label.str();
        phases.push_back(mixed->second);
    }
    return phases;
}

Options auto_phase_options(const Options &base,
                           const AutoReflectionPhase &phase)
{
    Options options = base;
    options.include_direct_sun = phase.direct_sun;
    if (phase.rough) {
        options.roughness = phase.roughness;
        options.cluster_rough_sources = true;
        options.sun_disk_samples = 1;
        options.secondary_sun_disk_samples = 0;
        options.adaptive_sun_disk = false;
    } else {
        options.rough_samples = 0;
        options.roughness = 0.0;
        options.cluster_rough_sources = false;
    }
    return options;
}

void add_matrix(std::vector<double> &target,
                const std::vector<double> &source)
{
    if (target.size() != source.size())
        throw std::runtime_error("cannot add matrices with different sizes");
    for (std::size_t i = 0; i < target.size(); ++i)
        target[i] += source[i];
}

void evaluate_auto_phase(
    const AutoReflectionPhase &phase, const Options &phase_options,
    const std::vector<Sun> &suns, const std::vector<std::size_t> &active_suns,
    const AdaptiveSunTree &adaptive_sun_tree,
    const std::vector<Viewpoint> &views, const Vec3 &up,
    BuiltinBackendHolder &builtin_backend,
    std::vector<double> &contrast, std::vector<double> &illuminance)
{
    if (!builtin_backend.backend)
        throw std::runtime_error(
            "automatic material mode has no built-in Radiance backend");
    const bool have_candidates = phase.direct_sun ||
        !phase.reflections.first_order_normals.empty() ||
        !phase.reflections.second_order_paths.empty();
    if (!have_candidates)
        return;

    const std::vector<std::string> modifiers(
        phase.materials.begin(), phase.materials.end());
    builtin_backend.backend->set_reflection_modifiers(modifiers);
    std::vector<double> phase_contrast(contrast.size(), 0.0);
    std::vector<double> phase_illuminance(illuminance.size(), 0.0);
    const std::set<std::string> no_transparent_materials;

    if (!phase_options.quiet)
        std::fprintf(stderr,
            "%s: auto phase '%s': %lu modifiers, %lu 1R proposals, "
            "%lu 2R paths%s", progname, phase.label.c_str(),
            static_cast<unsigned long>(phase.materials.size()),
            static_cast<unsigned long>(
                phase.reflections.first_order_normals.size()),
            static_cast<unsigned long>(
                phase.reflections.second_order_paths.size()),
            phase.rough ? " (rough-lobe quadrature)" : "");
        std::fputc('\n', stderr);

    for (std::size_t first_view = 0; first_view < views.size();
            first_view += static_cast<std::size_t>(
                phase_options.view_batch_size)) {
        const std::size_t last_view = std::min(
            views.size(), first_view+static_cast<std::size_t>(
                phase_options.view_batch_size));
        const std::function<void(const std::vector<std::size_t> &)>
            evaluate = [&](const std::vector<std::size_t> &indices) {
                evaluate_sun_subset(
                    indices, suns, views, phase.reflections, up,
                    phase.materials, no_transparent_materials, phase_options,
                    first_view, last_view, &builtin_backend, phase_contrast,
                    phase_illuminance);
            };
        if (phase_options.adaptive_sun_sampling && !active_suns.empty()) {
            std::vector<double> *adaptive_illuminance =
                collect_mirror_illuminance(phase_options) ?
                &phase_illuminance : NULL;
            evaluate_adaptive_suns(
                adaptive_sun_tree, suns, phase_options, first_view, last_view,
                suns.size(), phase_contrast, adaptive_illuminance, evaluate);
        } else {
            evaluate(active_suns);
        }
    }
    add_matrix(contrast, phase_contrast);
    add_matrix(illuminance, phase_illuminance);
}
#endif

void write_matrix(const Options &options, const std::vector<double> &contrast,
                  std::size_t nviews, std::size_t nsteps,
                  int argc, char *argv[])
{
    FILE *output = stdout;
    if (!options.output_path.empty()) {
        output = std::fopen(options.output_path.c_str(), "w");
        if (!output)
            throw std::runtime_error("cannot open output file '" +
                                     options.output_path + "'");
    }
    if (!options.no_header) {
        newheader("RADIANCE", output);
        printargs(argc, argv, output);
        fputnow(output);
        std::fprintf(output, "NROWS=%lu\nNCOLS=%lu\nNCOMP=1\n",
                     static_cast<unsigned long>(nviews),
                     static_cast<unsigned long>(nsteps));
        fputformat("ascii", output);
        std::fputc('\n', output);
    }
    for (std::size_t view = 0; view < nviews; ++view)
        for (std::size_t time = 0; time < nsteps; ++time)
            std::fprintf(output, "%.9e%c", contrast[view*nsteps+time],
                         time+1 == nsteps ? '\n' : '\t');
    if (std::fflush(output) == EOF ||
            (output != stdout && std::fclose(output) == EOF))
        throw std::runtime_error("error writing output matrix");
}

void write_mirror_illuminance_matrix(
    const Options &options, const std::vector<double> &illuminance,
    std::size_t nviews, std::size_t nsteps, int argc, char *argv[])
{
    if (!collect_mirror_illuminance(options))
        return;
    FILE *output = std::fopen(
        options.mirror_illuminance_output_path.c_str(), "w");
    if (!output)
        throw std::runtime_error(
            "cannot open mirror illuminance output file '" +
            options.mirror_illuminance_output_path + "'");
    if (!options.no_header) {
        newheader("RADIANCE", output);
        printargs(argc, argv, output);
        fputnow(output);
        std::fprintf(output,
            "NROWS=%lu\nNCOLS=%lu\nNCOMP=1\n"
            "QUANTITY=illuminance\nPATH_FAMILY=mirror_1R_2R\n",
            static_cast<unsigned long>(nviews),
            static_cast<unsigned long>(nsteps));
        fputformat("ascii", output);
        std::fputc('\n', output);
    }
    for (std::size_t view = 0; view < nviews; ++view)
        for (std::size_t time = 0; time < nsteps; ++time)
            std::fprintf(output, "%.9e%c",
                         illuminance[view*nsteps+time],
                         time+1 == nsteps ? '\n' : '\t');
    if (std::fflush(output) == EOF || std::fclose(output) == EOF)
        throw std::runtime_error(
            "error writing mirror illuminance matrix");
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
        "  --rough-coarse-samples n    first adaptive integration level (default 256)\n"
        "  --rough-medium-samples n    accepted adaptive integration level (default 1024)\n"
        "  --rough-convergence f       relative contrast/illuminance tolerance (default 0.05)\n"
        "  --rough-illuminance-tolerance lux absolute illuminance tolerance (default 1)\n"
        "  --rough-cell-trigger f      cell-refinement trigger as glare-threshold fraction (default 0.5)\n"
        "  --rough-secondary-samples n Radiance samples at the second rough bounce (2R only; default 16)\n"
        "  --rough-extent sigma        sampled lobe radius in standard deviations (default 3)\n"
        "  --rough-pilot-threshold f   pilot luminance as a fraction of the glare threshold (default 1e-6)\n"
        "  --rough-seed seed           deterministic rough-cap rotation seed (default 0)\n"
        "  --auto-materials            classify ordinary Radiance reflectors and sample ideal/rough paths automatically\n"
        "  --auto-material-allowlist f restrict --auto-materials to listed material identifiers\n"
        "  --allowlist-fallback-level n re-search missing allowlisted materials at level n\n"
        "  --ideal-only                automatically retain ideal reflectors and ignore rough materials\n"
        "  --reflection-level level    path presearch level; increase to recover missed view paths (default 5)\n"
        "  --reflection-seed seed      reproducible search jitter seed (default 0)\n"
        "  --reflection-octree file    prebuilt scene containing a skyglow sky\n"
        "  --save-normals file         save normals with view/modifier/material/primitive sources\n"
        "  --save-reflection-paths f   save sourced ordered 2R paths and originating views\n"
        "  -M file                     optional mirror material filter (default all)\n"
        "  --proposal-modifiers file   material filter used only for path-tree proposals\n"
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

Options parse_options(int argc, char *argv[])
{
    Options options;
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
            options.adaptive_rough_sampling = true;
        else if (argument == "--adaptive-rough-cells")
            options.adaptive_rough_cells = true;
        else if (argument == "--rough-pilot-samples")
            options.rough_pilot_samples = parse_int(
                require_option_value(index, argc, argv, argument),
                "rough pilot sample count");
        else if (argument == "--rough-coarse-samples")
            options.rough_coarse_samples = parse_int(
                require_option_value(index, argc, argv, argument),
                "rough coarse sample count");
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
        else if (argument == "--auto-materials")
            options.auto_materials = true;
        else if (argument == "--ideal-only")
            options.ideal_only = true;
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
        } else if (argument == "--rough-pilot-threshold") {
            const std::string value = require_option_value(index, argc, argv, argument);
            if (!parse_double(value, options.rough_prefilter))
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
        } else if (argument == "--rough-convergence") {
            const std::string value = require_option_value(index, argc, argv, argument);
            if (!parse_double(value, options.rough_convergence))
                throw std::runtime_error("invalid rough convergence '" + value + "'");
        } else if (argument == "--rough-illuminance-tolerance") {
            const std::string value = require_option_value(index, argc, argv, argument);
            if (!parse_double(value, options.rough_illuminance_tolerance))
                throw std::runtime_error(
                    "invalid rough illuminance tolerance '" + value + "'");
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
    if (options.adaptive_rough_integration || options.adaptive_rough_cells)
        options.adaptive_rough_sampling = true;
    if (options.adaptive_rough_sampling && options.rough_prefilter <= 0.0)
        options.rough_prefilter = 1.0e-6;
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
    if (!(options.adaptive_sun_max_angle > 0.0 &&
            options.adaptive_sun_max_angle < 180.0))
        throw std::runtime_error(
            "adaptive solar maximum angle must lie in (0, 180) degrees");
    if (!(options.adaptive_sun_min_angle > 0.0 &&
            options.adaptive_sun_min_angle <=
            options.adaptive_sun_max_angle))
        throw std::runtime_error(
            "adaptive solar minimum angle must be positive and no larger "
            "than the maximum angle");
    if (!(options.adaptive_sun_error >= 0.0 &&
            options.adaptive_sun_error <= 1.0))
        throw std::runtime_error(
            "adaptive solar response error must lie in [0, 1]");
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
        if (options.adaptive_rough_integration) {
            const int coarse_side = static_cast<int>(std::sqrt(
                static_cast<double>(options.rough_coarse_samples)));
            const int medium_side = static_cast<int>(std::sqrt(
                static_cast<double>(options.rough_medium_samples)));
            if (coarse_side*coarse_side != options.rough_coarse_samples ||
                    medium_side*medium_side != options.rough_medium_samples ||
                    options.rough_coarse_samples >=
                        options.rough_medium_samples ||
                    options.rough_medium_samples > options.rough_samples)
                throw std::runtime_error(
                    "adaptive rough integration levels must be increasing "
                    "perfect squares no larger than --rough-samples");
        }
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
    if (!(options.rough_convergence > 0.0 &&
            options.rough_convergence <= 1.0))
        throw std::runtime_error(
            "rough convergence tolerance must lie in (0, 1]");
    if (options.rough_illuminance_tolerance < 0.0)
        throw std::runtime_error(
            "rough illuminance tolerance cannot be negative");
    if (!(options.rough_cell_trigger > 0.0 &&
            options.rough_cell_trigger <= 1.0))
        throw std::runtime_error(
            "rough cell trigger must lie in (0, 1]");
    if (options.adaptive_rough_cells &&
            options.adaptive_rough_integration)
        throw std::runtime_error(
            "adaptive rough-cell and staged integration modes are mutually "
            "exclusive");
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
    if (options.rough_prefilter < 0.0 || options.rough_prefilter > 1.0)
        throw std::runtime_error("rough prefilter must lie in [0, 1]");
    if (options.rough_prefilter > 0.0 && options.rough_samples <= 0)
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

} // namespace

int main(int argc, char *argv[])
{
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN);
#endif
    progname = fixargv0(argv[0]);
    try {
        const Options options = parse_options(argc, argv);
        validate_options(options);
        const std::vector<Viewpoint> views = load_views(options.views_path);
        const Vec3 up = normalized(options.up, "view up vector");
        std::set<std::string> mirror_materials;
        if (!options.mirror_modifiers_path.empty())
            mirror_materials = load_modifier_names(
                options.mirror_modifiers_path, "mirror modifier");
        std::set<std::string> proposal_materials = mirror_materials;
        if (!options.proposal_modifiers_path.empty())
            proposal_materials = load_modifier_names(
                options.proposal_modifiers_path,
                "path-tree proposal modifier");
        BuiltinBackendHolder builtin_backend;
#ifdef SPECULARCONTRIB_BUILTIN_RCONTRIB
        AutoMaterialPlan auto_material_plan;
        std::set<std::string> auto_material_allowlist;
        std::set<std::string> matched_auto_materials;
        std::size_t inspected_auto_material_count = 0;
        if (options.auto_materials) {
            const std::set<std::string> empty_reflection_set;
            builtin_backend.backend = create_builtin_backend(
                options.octree, options, empty_reflection_set);
            const std::vector<SpecularMaterialInfo> inspected_materials =
                builtin_backend.backend->inspect_specular_materials();
            inspected_auto_material_count = inspected_materials.size();
            std::vector<SpecularMaterialInfo> selected_materials;
            if (!options.auto_material_allowlist_path.empty()) {
                auto_material_allowlist = load_modifier_names(
                    options.auto_material_allowlist_path,
                    "automatic material allowlist");
                for (std::size_t i = 0; i < inspected_materials.size(); ++i)
                    if (auto_material_allowlist.count(
                            inspected_materials[i].name)) {
                        selected_materials.push_back(inspected_materials[i]);
                        matched_auto_materials.insert(
                            inspected_materials[i].name);
                    }
            } else {
                selected_materials = inspected_materials;
            }
            auto_material_plan = make_auto_material_plan(
                selected_materials, options.ideal_only);
            mirror_materials = auto_material_plan.proposal_materials;
            proposal_materials = auto_material_plan.proposal_materials;
        }
#endif
        std::set<std::string> transparent_materials;
        if (!options.integrated_path_check &&
                !options.transparent_modifiers_path.empty())
            transparent_materials = load_modifier_names(
                options.transparent_modifiers_path, "transparent modifier");
        if (!options.integrated_path_check)
            for (std::set<std::string>::const_iterator material =
                    transparent_materials.begin();
                    material != transparent_materials.end(); ++material)
                if (mirror_materials.count(*material))
                    throw std::runtime_error("modifier '" + *material +
                        "' cannot be both mirror and transparent");

        ReflectionData reflections;
        if (!options.normals_path.empty()) {
            const ReflectionData loaded = load_normals(
                options.normals_path, options.normal_tolerance);
            reflections = reflection_data_from_normals(
                loaded.first_order_normals, views.size(),
                options.max_specular_bounces);
            reflections.first_order_sources = loaded.first_order_sources;
        }
        else if (!options.normal_rad_path.empty()) {
            const ReflectionData loaded = load_rad_normals(
                options.normal_rad_path, mirror_materials,
                options.normal_tolerance);
            reflections = reflection_data_from_normals(
                loaded.first_order_normals, views.size(),
                options.max_specular_bounces);
            reflections.first_order_sources = loaded.first_order_sources;
        }
        else if (!options.auto_materials || !proposal_materials.empty())
            reflections = discover_reflection_paths(
                views, options.normal_tolerance, proposal_materials, options);
#ifdef SPECULARCONTRIB_BUILTIN_RCONTRIB
        if (options.allowlist_fallback_level >= 0 &&
                !matched_auto_materials.empty()) {
            const std::set<std::string> discovered =
                discovered_reflection_materials(reflections);
            std::set<std::string> missing;
            for (std::set<std::string>::const_iterator material =
                    matched_auto_materials.begin();
                    material != matched_auto_materials.end(); ++material)
                if (!discovered.count(*material))
                    missing.insert(*material);
            if (!missing.empty()) {
                Options fallback_options = options;
                fallback_options.reflection_level =
                    options.allowlist_fallback_level;
                const ReflectionData fallback = discover_reflection_paths(
                    views, options.normal_tolerance, proposal_materials,
                    fallback_options);
                const std::size_t normal_sources_before =
                    normal_source_count(reflections);
                const std::size_t paths_before =
                    reflections.second_order_paths.size();
                merge_missing_material_paths(
                    reflections, fallback, missing,
                    options.normal_tolerance);
                if (!options.quiet) {
                    std::fprintf(stderr,
                        "%s: allowlist fallback level %d searched %lu "
                        "missing materials and added %lu normal records "
                        "and %lu ordered 2R paths\n",
                        progname, options.allowlist_fallback_level,
                        static_cast<unsigned long>(missing.size()),
                        static_cast<unsigned long>(
                            normal_source_count(reflections)-
                            normal_sources_before),
                        static_cast<unsigned long>(
                            reflections.second_order_paths.size()-
                            paths_before));
                }
            }
        }
#endif
        if (options.all_normal_pairs)
            append_all_normal_pairs(reflections, views.size(),
                                    options.normal_tolerance);
        const bool have_mirror_paths =
            !reflections.first_order_normals.empty() ||
            !reflections.second_order_paths.empty();
        const bool have_trace_candidates =
            have_mirror_paths || options.include_direct_sun;
        if (!options.save_normals_path.empty())
            save_normals(options.save_normals_path, reflections);
        if (!options.save_paths_path.empty())
            save_reflection_paths(options.save_paths_path,
                                  reflections.second_order_paths);

        const std::vector<Sun> suns = load_suns(options.suns_path);
        std::vector<std::size_t> active_suns;
        for (std::size_t time = 0; time < suns.size(); ++time)
            if (suns[time].active)
                active_suns.push_back(time);
        AdaptiveSunTree adaptive_sun_tree;
        adaptive_sun_tree.root = -1;
        if (options.adaptive_sun_sampling && !active_suns.empty()) {
            validate_adaptive_sun_radiance(suns);
            adaptive_sun_tree = build_adaptive_sun_tree(active_suns, suns);
        }
#ifdef SPECULARCONTRIB_BUILTIN_RCONTRIB
        if (!builtin_backend.backend && options.rcontrib.empty() &&
                options.nonspec_octree.empty() &&
                options.rough_prefilter <= 0.0 && have_trace_candidates)
            builtin_backend.backend = create_builtin_backend(
                options.octree, options, mirror_materials);
#endif
        std::vector<double> contrast(views.size()*suns.size(), 0.0);
        std::vector<double> mirror_illuminance(
            views.size()*suns.size(), 0.0);

        if (!options.quiet) {
#ifdef SPECULARCONTRIB_BUILTIN_RCONTRIB
            if (options.auto_materials) {
                if (!options.auto_material_allowlist_path.empty()) {
                    std::fprintf(stderr,
                        "%s: automatic material allowlist matched %lu of "
                        "%lu names and retained %lu of %lu inspected "
                        "material definitions\n", progname,
                        static_cast<unsigned long>(
                            matched_auto_materials.size()),
                        static_cast<unsigned long>(
                            auto_material_allowlist.size()),
                        static_cast<unsigned long>(
                            matched_auto_materials.size()),
                        static_cast<unsigned long>(
                            inspected_auto_material_count));
                    for (std::set<std::string>::const_iterator material =
                            auto_material_allowlist.begin();
                            material != auto_material_allowlist.end();
                            ++material)
                        if (!matched_auto_materials.count(*material))
                            std::fprintf(stderr,
                                "%s: warning: automatic material allowlist "
                                "entry '%s' was not found in the octree\n",
                                progname, material->c_str());
                }
                std::size_t rough_count = 0;
                for (std::map<long long, std::set<std::string> >::const_iterator
                        group = auto_material_plan.rough_materials.begin();
                        group != auto_material_plan.rough_materials.end();
                        ++group)
                    rough_count += group->second.size();
                if (options.ideal_only)
                    std::fprintf(stderr,
                        "%s: auto-selected %lu ideal reflective modifiers; "
                        "ignored %lu rough and %lu BSDF/aBSDF modifiers\n",
                        progname,
                        static_cast<unsigned long>(
                            auto_material_plan.ideal_materials.size()),
                        static_cast<unsigned long>(
                            auto_material_plan.ignored_rough_materials.size()),
                        static_cast<unsigned long>(
                            auto_material_plan.excluded_materials.size()));
                else
                    std::fprintf(stderr,
                        "%s: auto-classified %lu ideal and %lu rough reflective "
                        "modifiers in %lu roughness groups; excluded %lu BSDF/aBSDF "
                        "modifiers\n", progname,
                        static_cast<unsigned long>(
                            auto_material_plan.ideal_materials.size()),
                        static_cast<unsigned long>(rough_count),
                        static_cast<unsigned long>(
                            auto_material_plan.rough_materials.size()),
                        static_cast<unsigned long>(
                            auto_material_plan.excluded_materials.size()));
                for (std::size_t i = 0;
                        i < auto_material_plan.unsupported_materials.size();
                        ++i) {
                    const SpecularMaterialInfo &material =
                        auto_material_plan.unsupported_materials[i];
                    std::fprintf(stderr,
                        "%s: warning: skipping unsupported procedural "
                        "specular material '%s' (%s); use manual -M sampling "
                        "if it must be included\n", progname,
                        material.name.c_str(), material.type.c_str());
                }
            }
#endif
            std::fprintf(stderr,
                "%s: %lu views, %lu 1R normal directions (%lu source "
                "records), %lu ordered 2R paths, %lu time steps (%lu "
                "active), max %dR\n",
                progname, static_cast<unsigned long>(views.size()),
                static_cast<unsigned long>(
                    reflections.first_order_normals.size()),
                static_cast<unsigned long>(normal_source_count(reflections)),
                static_cast<unsigned long>(
                    reflections.second_order_paths.size()),
                static_cast<unsigned long>(suns.size()),
                static_cast<unsigned long>(active_suns.size()),
                options.max_specular_bounces);
            if (!have_mirror_paths)
                std::fprintf(stderr,
                    "%s: automatic search found no requested mirror path; "
                    "the mirror contrast term remains zero%s\n",
                    progname, options.include_direct_sun ?
                    "; evaluating direct-sun candidates only" :
                    " and a zero matrix will be written");
            if (!have_trace_candidates)
                std::fprintf(stderr,
                    "%s: skipping annual source tracing because no "
                    "candidate path exists\n", progname);
            else if (options.rcontrib.empty())
                std::fprintf(stderr,
                    "%s: using built-in Radiance rcontrib module\n",
                    progname);
            else
                std::fprintf(stderr,
                    "%s: using external rcontrib executable '%s'\n",
                    progname, options.rcontrib.c_str());
            if (mirror_materials.empty() && !options.auto_materials)
                std::fprintf(stderr,
                    "%s: no -M filter; searching all reflective materials "
                    "and disabling material-specific hit checks\n", progname);
            if (!transparent_materials.empty() &&
                    !options.integrated_path_check)
                std::fprintf(stderr,
                    "%s: crossing %lu transparent modifiers during hit checks "
                    "(maximum %d surfaces)\n", progname,
                    static_cast<unsigned long>(transparent_materials.size()),
                    options.max_transparent_hits);
            if (options.sun_disk_samples > 1)
                std::fprintf(stderr,
                    "%s: resolving the reflected solar disk with %d "
                    "equal-solid-angle samples\n", progname,
                    options.sun_disk_samples);
            if (options.secondary_sun_disk_samples > 0 &&
                    options.secondary_sun_disk_samples !=
                    options.sun_disk_samples)
                std::fprintf(stderr,
                    "%s: using %d solar-disk samples for 2R-only paths\n",
                    progname, options.secondary_sun_disk_samples);
            if (options.adaptive_sun_disk)
                std::fprintf(stderr,
                    "%s: pilot-testing solar-disk paths with %d samples "
                    "for direct/1R and %d for 2R-only paths; protecting "
                    "%.6g degrees\n", progname,
                    options.sun_disk_pilot_samples,
                    options.secondary_sun_disk_pilot_samples,
                    options.sun_disk_pilot_guard_angle);
            if (options.include_direct_sun)
                std::fprintf(stderr,
                    "%s: including deterministic direct-sun contrast\n",
                    progname);
            if (options.direct_specular_only)
                std::fprintf(stderr,
                    "%s: suppressing direct diffuse terms in the built-in "
                    "Radiance backend\n", progname);
            else if (options.nonspec_octree.empty())
                std::fprintf(stderr,
                    "%s: no non-specular baseline; assuming no diffuse component\n",
                    progname);
            if (collect_mirror_illuminance(options))
                std::fprintf(stderr,
                    "%s: accumulating all positive 1R/2R mirror energy "
                    "as view-plane illuminance in '%s'\n",
                    progname,
                    options.mirror_illuminance_output_path.c_str());
            if (options.integrated_path_check)
                std::fprintf(stderr,
                    "%s: validating actual reflected child-ray branches; "
                    "Radiance handles transmission automatically\n",
                    progname);
            if (options.adaptive_sun_sampling)
                std::fprintf(stderr,
                    "%s: adaptive solar sampling, %.4g degree maximum "
                    "diameter, %.4g degree minimum diameter, %.4g response "
                    "error\n", progname,
                    options.adaptive_sun_max_angle,
                    options.adaptive_sun_min_angle,
                    options.adaptive_sun_error);
            if (options.rough_samples > 0 && !options.auto_materials) {
                std::fprintf(stderr,
                    "%s: rough mode alpha=%.6g, %d equal-solid-angle samples, "
                    "extent=%.3g sigma", progname, options.roughness,
                    options.rough_samples, options.rough_extent);
                if (options.max_specular_bounces > 1)
                    std::fprintf(stderr, ", %d secondary samples",
                                 options.rough_secondary_samples);
                std::fputc('\n', stderr);
            }
        }

        if (options.auto_materials) {
#ifdef SPECULARCONTRIB_BUILTIN_RCONTRIB
            const std::vector<AutoReflectionPhase> phases =
                make_auto_reflection_phases(
                    auto_material_plan, reflections, options);
            for (std::size_t phase = 0; phase < phases.size(); ++phase)
                evaluate_auto_phase(
                    phases[phase], auto_phase_options(options, phases[phase]),
                    suns, active_suns, adaptive_sun_tree, views, up,
                    builtin_backend, contrast, mirror_illuminance);
#endif
        } else for (std::size_t first_view = 0;
                have_trace_candidates && first_view < views.size();
                first_view += static_cast<std::size_t>(
                    options.view_batch_size)) {
            const std::size_t last_view = std::min(
                views.size(), first_view + static_cast<std::size_t>(
                    options.view_batch_size));
            if (!options.quiet && views.size() >
                    static_cast<std::size_t>(options.view_batch_size))
                std::fprintf(stderr, "%s: viewpoints %lu-%lu/%lu\n", progname,
                    static_cast<unsigned long>(first_view+1),
                    static_cast<unsigned long>(last_view),
                    static_cast<unsigned long>(views.size()));

            const std::function<void(const std::vector<std::size_t> &)>
                evaluate = [&](const std::vector<std::size_t> &indices) {
                    evaluate_sun_subset(
                        indices, suns, views, reflections, up,
                        mirror_materials, transparent_materials, options,
                        first_view, last_view, &builtin_backend, contrast,
                        mirror_illuminance);
                };
            if (options.adaptive_sun_sampling && !active_suns.empty()) {
                std::vector<double> *adaptive_illuminance =
                    collect_mirror_illuminance(options) ?
                    &mirror_illuminance : NULL;
                const AdaptiveSunStats stats = evaluate_adaptive_suns(
                    adaptive_sun_tree, suns, options, first_view, last_view,
                    suns.size(), contrast, adaptive_illuminance, evaluate);
                if (!options.quiet)
                    std::fprintf(stderr,
                        "%s: viewpoints %lu-%lu sampled %lu/%lu active "
                        "solar directions in %d passes (%lu final clusters)\n",
                        progname,
                        static_cast<unsigned long>(first_view+1),
                        static_cast<unsigned long>(last_view),
                        static_cast<unsigned long>(stats.sampled),
                        static_cast<unsigned long>(active_suns.size()),
                        stats.passes,
                        static_cast<unsigned long>(stats.leaves));
            } else {
                evaluate(active_suns);
            }
        }
        write_matrix(options, contrast, views.size(), suns.size(), argc, argv);
        write_mirror_illuminance_matrix(
            options, mirror_illuminance, views.size(), suns.size(),
            argc, argv);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s: %s\n", progname ? progname : "specularcontrast",
                     error.what());
        return 1;
    }
    return 0;
}
