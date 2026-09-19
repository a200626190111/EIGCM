#ifndef lint
static const char RCSid[] = "$Id$";
#endif

/*
 * Integrate the non-diffuse, direct-sun transmission of one or more BSDF
 * window groups into the DGP contrast term.  The output matrix has one row
 * per viewpoint and one column per source record in the solar file.
 */

#include "bsdf.h"
#include "standard.h"
#include "object.h"
#include "face.h"
#include "octree.h"
#include "otypes.h"
#include "random.h"
#include "rtprocess.h"
#include "rtio.h"
#include "rtmath.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <signal.h>
#endif

/* The common scene reader needs the Radiance type table, but this utility
 * does not link the renderer's material dispatch module. */
extern "C" int o_default()
{
    return 0;
}

extern "C" {
FUN ofun[NUMOTYPE] = INIT_OTYPE;
CUBE thescene;
}

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
    double angular_diameter;
    double omega;
    double luminance;
    bool active;
};

struct Primitive {
    std::string modifier;
    std::string type;
    std::string identifier;
    std::vector<std::string> string_args;
    std::vector<std::string> integer_args;
    std::vector<double> real_args;
};

struct Polygon {
    std::vector<Vec3> vertices;
    Vec3 normal;
    Vec3 center;
};

struct WindowGroup {
    std::string geometry_path;
    std::string bsdf_path;
    std::string absdf_path;
    std::string absdf_identifier;
    std::vector<Polygon> polygons;
    Vec3 normal;
    Vec3 up;
    Vec3 axis[3];                 // world directions of local X, Y, Z
    const SDData *bsdf = NULL;
    bool is_absdf = false;
    bool auto_discovered = false;
    OBJECT material_object = OVOID;
    std::string material_name;
};

struct ThroughComponent {
    bool present = false;
    double coefficient = 0.0;
    double surround = 0.0;
    double min_projected_solid_angle = 0.0;
};

struct PendingContribution {
    Vec3 origin;
    Vec3 direction;
    std::size_t result_index;
    double target_distance;
    double value;
    std::size_t peak_sample_index = std::numeric_limits<std::size_t>::max();
};

struct GlareDirectionSample {
    Vec3 direction;
    std::size_t result_index;
    std::size_t partition = 0;
    std::size_t source_id = 0;
    double luminance;
    double solid_angle;
    double map_u0 = 0.0;
    double map_v0 = 0.0;
    double map_size = 0.0;
    bool adaptive_cell = false;
    bool coherent_source = false;
    bool visible = false;
};

struct PeakCandidate {
    Vec3 outgoing;
    double map_u;
    double map_v;
    double luminance;
    double solid_angle;
};

struct AdaptivePeakCell {
    Vec3 outgoing;
    double luminance;
    double map_u0;
    double map_v0;
    double map_size;
    int level;
};

struct BoundaryPeakSample {
    Vec3 outgoing;
    double luminance;
    double solid_angle;
};

struct BoundaryPeakSource {
    std::vector<BoundaryPeakSample> samples;
    std::vector<std::size_t> candidate_indices;
    std::vector<std::array<double, 2> > boundary;
    std::array<double, 2> anchor = {{0.5, 0.5}};
    double solid_angle = 0.0;
    double importance_solid_angle = 0.0;
    double scale = 1.0;
    bool resolved = false;
};

struct BoundaryStatistics {
    std::size_t resolved_sources = 0;
    double scale_sum = 0.0;
    double scale_min = std::numeric_limits<double>::infinity();
    double scale_max = 0.0;
    std::array<std::size_t, 6> scale_bins = {{0, 0, 0, 0, 0, 0}};
    std::array<std::size_t, 6> solid_angle_bins = {{0, 0, 0, 0, 0, 0}};
};

enum PeakReconstructionResult {
    PeakReconstructionNone,
    PeakReconstructionAdaptive,
    PeakReconstructionFallback,
    PeakReconstructionBoundary
};

struct Options {
    bool no_header = false;
    bool quiet = false;
    bool include_absdf_through = false;
    bool cluster_peaks = true;
    bool adaptive_peak_cells = true;
    bool candidate_boundary_peaks = false;
    bool candidate_boundary_fallback = false;
    bool boundary_allow_expansion = false;
    std::string views_path;
    std::string suns_path;
    std::string output_path;
    std::string visibility_octree;
    std::string auto_bsdf_octree;
    std::string rtrace = "rtrace";
    std::vector<WindowGroup> groups;
    int samples = 256;
    int sun_disk_samples = 1;
    int seed = 0;
    int nproc = 1;
    int visibility_batch = 65536;
    double threshold = 2000.0;
    /* Importance weights are integration measures, not a directional tiling.
     * Their equivalent caps therefore need a modest overlap allowance to
     * reconnect random samples belonging to one physical BSDF peak. */
    double peak_cluster_factor = 2.0;
    double peak_max_link_angle = 10.0;
    int adaptive_base_level = 4;
    int adaptive_max_level = 7;
    double adaptive_contrast_tolerance = 0.05;
    double adaptive_gradient_tolerance = 0.25;
    int boundary_rays = 24;
    int boundary_bisections = 10;
    /* Boundary tracing targets compact, sub-grid glare peaks. Wider fields
     * remain on the established adaptive/importance integration route. */
    double boundary_max_solid_angle = 0.05;
    double visibility_tolerance = 1.0e-3;
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

std::size_t peak_root(std::vector<std::size_t> &parent,
                      std::size_t index);
void join_peaks(std::vector<std::size_t> &parent,
                std::vector<unsigned char> &rank,
                std::size_t first, std::size_t second);
double equivalent_cap_radius(double solid_angle);

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

int parse_nonnegative_int(const std::string &word, const std::string &what)
{
    char *end = NULL;
    errno = 0;
    const long value = std::strtol(word.c_str(), &end, 10);
    if (errno || end == word.c_str() || *end || value < 0 ||
            value > std::numeric_limits<int>::max())
        throw std::runtime_error("invalid " + what + " '" + word + "'");
    return static_cast<int>(value);
}

double parse_nonnegative_double(const std::string &word,
                                const std::string &what)
{
    double value;
    if (!parse_double(word, value) || value < 0.0 || !std::isfinite(value))
        throw std::runtime_error("invalid " + what + " '" + word + "'");
    return value;
}

double parse_finite_double(const std::string &word, const std::string &what)
{
    double value;
    if (!parse_double(word, value) || !std::isfinite(value))
        throw std::runtime_error("invalid " + what + " '" + word + "'");
    return value;
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
        view.direction = normalized(
            Vec3{{values[3], values[4], values[5]}}, "view direction");
        views.push_back(view);
    }
    if (views.empty())
        throw std::runtime_error("no viewpoints found in '" + path + "'");
    return views;
}

std::vector<std::string> radiance_tokens(const std::string &path)
{
    std::ifstream input(path.c_str());
    if (!input)
        throw std::runtime_error("cannot open Radiance file '" + path + "'");
    std::vector<std::string> tokens;
    std::string line;
    while (std::getline(input, line)) {
        line = trim_comment(line);
        if (line.empty())
            continue;
        if (line[0] == '!')
            throw std::runtime_error(
                "command expansion is not supported in '" + path + "'");
        std::istringstream words(line);
        std::string word;
        while (words >> word)
            tokens.push_back(word);
    }
    return tokens;
}

std::vector<std::string> take_counted(const std::vector<std::string> &tokens,
                                      std::size_t &index,
                                      const std::string &description)
{
    if (index >= tokens.size())
        throw std::runtime_error("missing " + description + " argument count");
    const int count = parse_nonnegative_int(tokens[index++], description +
                                             " argument count");
    if (index + static_cast<std::size_t>(count) > tokens.size())
        throw std::runtime_error("incomplete " + description + " arguments");
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
        if (primitive.type == "alias") {
            if (index >= tokens.size())
                throw std::runtime_error("missing alias target in '" + path + "'");
            primitive.string_args.push_back(tokens[index++]);
            primitives.push_back(primitive);
            continue;
        }
        primitive.string_args = take_counted(tokens, index, "string");
        primitive.integer_args = take_counted(tokens, index, "integer");
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

std::string parent_directory(const std::string &path)
{
    const std::string::size_type slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : path.substr(0, slash);
}

bool file_exists(const std::string &path)
{
    std::ifstream input(path.c_str(), std::ios::binary);
    return static_cast<bool>(input);
}

std::string resolve_material_path(const std::string &material_path,
                                  const std::string &referenced_path)
{
    if (file_exists(referenced_path))
        return referenced_path;
    const std::string directory = parent_directory(material_path);
    if (!directory.empty()) {
        const std::string::size_type first = referenced_path.find_first_not_of("./\\");
        const std::string relative = first == std::string::npos ?
            referenced_path : referenced_path.substr(first);
        const std::string candidate = directory + "/" + relative;
        if (file_exists(candidate))
            return candidate;
    }
    return referenced_path;
}

void load_absdf_material(WindowGroup &group)
{
    const std::vector<Primitive> primitives = load_primitives(group.absdf_path);
    const Primitive *material = NULL;
    for (std::size_t i = 0; i < primitives.size(); ++i) {
        if (primitives[i].identifier != group.absdf_identifier)
            continue;
        if (material != NULL)
            throw std::runtime_error("duplicate material '" +
                                     group.absdf_identifier + "' in '" +
                                     group.absdf_path + "'");
        material = &primitives[i];
    }
    if (material == NULL)
        throw std::runtime_error("material '" + group.absdf_identifier +
                                 "' was not found in '" + group.absdf_path + "'");
    if (material->type != "aBSDF")
        throw std::runtime_error("material '" + group.absdf_identifier +
                                 "' is type '" + material->type +
                                 "', not aBSDF");
    if (material->modifier != "void")
        throw std::runtime_error("patterned aBSDF material '" +
                                 group.absdf_identifier +
                                 "' is not supported by ttsuncontrast");
    if (material->string_args.size() < 5)
        throw std::runtime_error("aBSDF material '" + group.absdf_identifier +
                                 "' has fewer than five string arguments");

    group.bsdf_path = resolve_material_path(group.absdf_path,
                                             material->string_args[0]);
    Vec3 material_up;
    for (int c = 0; c < 3; ++c) {
        if (!parse_double(material->string_args[c+1], material_up[c]))
            throw std::runtime_error(
                "aBSDF material '" + group.absdf_identifier +
                "' uses a non-constant up-vector expression; "
                "ttsuncontrast currently requires numeric ux uy uz");
    }
    if (material->string_args.size() > 5) {
        std::vector<std::string> transform_words(
            material->string_args.begin()+5, material->string_args.end());
        std::vector<char *> transform_args;
        for (std::size_t i = 0; i < transform_words.size(); ++i)
            transform_args.push_back(const_cast<char *>(transform_words[i].c_str()));
        XF transform;
        if (xf(&transform, static_cast<int>(transform_args.size()),
               transform_args.data()) != static_cast<int>(transform_args.size()))
            throw std::runtime_error("invalid transform on aBSDF material '" +
                                     group.absdf_identifier + "'");
        FVECT source = {material_up[0], material_up[1], material_up[2]};
        FVECT transformed;
        multv3(transformed, source, transform.xfm);
        material_up = Vec3{{transformed[0], transformed[1], transformed[2]}};
    }
    group.up = normalized(material_up, "aBSDF up vector");
    group.is_absdf = true;
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
        const std::map<std::string, Vec3>::const_iterator light =
            lights.find(records[i].modifier);
        if (light == lights.end())
            throw std::runtime_error("missing light primitive for " +
                                     records[i].modifier);
        Sun sun;
        sun.modifier = records[i].modifier;
        sun.direction = records[i].direction;
        sun.radiance = light->second;
        sun.angular_diameter = records[i].angle;
        sun.omega = 2.0*PI*(1.0-std::cos(0.5*records[i].angle));
        const double brightness = kBrightness[0]*sun.radiance[0] +
            kBrightness[1]*sun.radiance[1] +
            kBrightness[2]*sun.radiance[2];
        sun.luminance = kLuminousEfficacy*brightness;
        sun.active = brightness > 0.0;
        suns.push_back(sun);
    }
    return suns;
}

Polygon make_polygon(const Primitive &primitive, const std::string &path)
{
    if (primitive.real_args.size() < 9 || primitive.real_args.size()%3)
        throw std::runtime_error("invalid polygon '" + primitive.identifier +
                                 "' in '" + path + "'");
    Polygon polygon;
    polygon.center = Vec3{{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < primitive.real_args.size(); i += 3) {
        const Vec3 vertex = {{primitive.real_args[i], primitive.real_args[i+1],
                              primitive.real_args[i+2]}};
        polygon.vertices.push_back(vertex);
        for (int c = 0; c < 3; ++c)
            polygon.center[c] += vertex[c];
    }
    for (int c = 0; c < 3; ++c)
        polygon.center[c] /= polygon.vertices.size();
    Vec3 newell = {{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < polygon.vertices.size(); ++i) {
        const Vec3 &a = polygon.vertices[i];
        const Vec3 &b = polygon.vertices[(i+1)%polygon.vertices.size()];
        newell[0] += (a[1]-b[1])*(a[2]+b[2]);
        newell[1] += (a[2]-b[2])*(a[0]+b[0]);
        newell[2] += (a[0]-b[0])*(a[1]+b[1]);
    }
    polygon.normal = normalized(newell, "window polygon normal");
    return polygon;
}

std::vector<Polygon> load_polygons(const std::string &path)
{
    const std::vector<Primitive> primitives = load_primitives(path);
    std::vector<Polygon> polygons;
    for (std::size_t i = 0; i < primitives.size(); ++i)
        if (primitives[i].type == "polygon")
            polygons.push_back(make_polygon(primitives[i], path));
    if (polygons.empty())
        throw std::runtime_error("no polygons found in window file '" + path + "'");
    return polygons;
}

OBJREC *resolve_surface_material(OBJREC *surface, bool &decorated)
{
    decorated = false;
    OBJECT object = surface->omod;
    int aliases = 0;
    while (object != OVOID) {
        OBJREC *record = objptr(object);
        if (record->otype == MOD_ALIAS) {
            if (++aliases > 64)
                throw std::runtime_error("modifier alias loop while scanning octree");
            object = record->oargs.nsargs ?
                lastmod(object, record->oargs.sarg[0]) : record->omod;
            continue;
        }
        aliases = 0;
        if (ismaterial(record->otype)) {
            if (record->omod != OVOID)
                decorated = true;
            return record;
        }
        if (ispattern(record->otype) || istexture(record->otype) ||
                ismixture(record->otype))
            decorated = true;
        object = record->omod;
    }
    return NULL;
}

Polygon make_octree_polygon(OBJREC *object)
{
    FACE *face = getface(object);
    if (face == NULL || face->nv < 3 || face->area <= kEpsilon)
        throw std::runtime_error("invalid octree polygon '" +
                                 std::string(object->oname) + "'");
    Polygon polygon;
    polygon.center = Vec3{{0.0, 0.0, 0.0}};
    for (int vertex = 0; vertex < face->nv; ++vertex) {
        const RREAL *point = VERTEX(face, vertex);
        const Vec3 copy = {{point[0], point[1], point[2]}};
        polygon.vertices.push_back(copy);
        for (int component = 0; component < 3; ++component)
            polygon.center[component] += copy[component];
    }
    for (int component = 0; component < 3; ++component)
        polygon.center[component] /= polygon.vertices.size();
    polygon.normal = normalized(
        Vec3{{face->norm[0], face->norm[1], face->norm[2]}},
        "octree polygon normal");
    return polygon;
}

Vec3 observer_facing_normal(const Polygon &polygon,
                            const std::vector<Viewpoint> &views)
{
    Vec3 normal = polygon.normal;
    double facing = 0.0;
    for (std::size_t view = 0; view < views.size(); ++view)
        facing += dot(normal,
                      add_scaled(views[view].origin, polygon.center, -1.0));
    if (facing < 0.0)
        for (int component = 0; component < 3; ++component)
            normal[component] = -normal[component];
    return normal;
}

void configure_auto_material(WindowGroup &group, OBJREC *material,
                             const std::string &octree_path)
{
    const bool is_absdf = material->otype == MAT_ABSDF;
    const int has_thickness = is_absdf ? 0 : 1;
    if (material->oargs.nsargs < has_thickness+5)
        throw std::runtime_error("BSDF material '" +
            std::string(material->oname) + "' has too few string arguments");

    group.material_object = objndx(material);
    group.material_name = material->oname;
    group.absdf_identifier = material->oname;
    group.bsdf_path = resolve_material_path(
        octree_path, material->oargs.sarg[has_thickness]);
    group.is_absdf = is_absdf;
    group.auto_discovered = true;

    Vec3 material_up;
    for (int component = 0; component < 3; ++component) {
        if (!parse_double(material->oargs.sarg[has_thickness+1+component],
                          material_up[component]))
            throw std::runtime_error("BSDF material '" +
                group.material_name +
                "' uses a non-constant up-vector expression; automatic mode "
                "requires numeric ux uy uz");
    }
    const int transform_start = has_thickness+5;
    if (material->oargs.nsargs > transform_start) {
        std::vector<char *> arguments;
        for (int index = transform_start; index < material->oargs.nsargs; ++index)
            arguments.push_back(material->oargs.sarg[index]);
        XF transform;
        if (xf(&transform, static_cast<int>(arguments.size()), arguments.data()) !=
                static_cast<int>(arguments.size()))
            throw std::runtime_error("invalid transform on BSDF material '" +
                                     group.material_name + "'");
        FVECT source = {material_up[0], material_up[1], material_up[2]};
        FVECT transformed;
        multv3(transformed, source, transform.xfm);
        material_up = Vec3{{transformed[0], transformed[1], transformed[2]}};
    }
    group.up = normalized(material_up, "BSDF up vector");
}

void discover_bsdf_groups(Options &options,
                          const std::vector<Viewpoint> &views)
{
    readoct(const_cast<char *>(options.auto_bsdf_octree.c_str()), IO_SCENE,
            &thescene, NULL);
    std::size_t instances = 0;
    std::size_t meshes = 0;
    std::size_t matched_polygons = 0;
    for (OBJECT index = 0; index < nobjects; ++index) {
        OBJREC *object = objptr(index);
        if (object->otype == OBJ_INSTANCE) {
            ++instances;
            continue;
        }
        if (object->otype == OBJ_MESH) {
            ++meshes;
            continue;
        }
        if (object->otype != OBJ_FACE)
            continue;
        bool decorated = false;
        OBJREC *material = resolve_surface_material(object, decorated);
        if (material == NULL ||
                (material->otype != MAT_BSDF && material->otype != MAT_ABSDF))
            continue;
        if (decorated)
            throw std::runtime_error("patterned or textured BSDF material '" +
                std::string(material->oname) + "' on polygon '" +
                std::string(object->oname) +
                "' is not supported in automatic mode");

        Polygon polygon = make_octree_polygon(object);
        polygon.normal = observer_facing_normal(polygon, views);
        const OBJECT material_index = objndx(material);
        WindowGroup *group = NULL;
        for (std::size_t candidate = 0; candidate < options.groups.size();
                ++candidate) {
            WindowGroup &existing = options.groups[candidate];
            if (existing.auto_discovered &&
                    existing.material_object == material_index &&
                    std::fabs(dot(existing.polygons.front().normal,
                                  polygon.normal)) >= 0.999999) {
                group = &existing;
                break;
            }
        }
        if (group == NULL) {
            WindowGroup discovered;
            configure_auto_material(discovered, material,
                                    options.auto_bsdf_octree);
            discovered.geometry_path = "auto material '" +
                discovered.material_name + "'";
            options.groups.push_back(discovered);
            group = &options.groups.back();
        }
        group->polygons.push_back(polygon);
        ++matched_polygons;
    }
    if (matched_polygons == 0) {
        std::ostringstream message;
        message << "no polygon using a BSDF or aBSDF material was found in '"
                << options.auto_bsdf_octree << "'";
        if (instances || meshes)
            message << "; rebuild it with oconv -f so instance/mesh geometry "
                       "is expanded into the octree";
        throw std::runtime_error(message.str());
    }
    if (instances || meshes)
        std::fprintf(stderr,
            "%s: warning: automatic scan skipped %lu instance and %lu mesh "
            "objects; use an oconv -f octree if they may contain BSDF windows\n",
            progname, static_cast<unsigned long>(instances),
            static_cast<unsigned long>(meshes));
}

void orient_and_frame_group(WindowGroup &group,
                            const std::vector<Viewpoint> &views,
                            const Vec3 &up)
{
    if (group.polygons.empty())
        group.polygons = load_polygons(group.geometry_path);
    Vec3 group_center = {{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < group.polygons.size(); ++i)
        for (int c = 0; c < 3; ++c)
            group_center[c] += group.polygons[i].center[c];
    for (int c = 0; c < 3; ++c)
        group_center[c] /= group.polygons.size();

    Vec3 normal = group.polygons.front().normal;
    double facing = 0.0;
    for (std::size_t i = 0; i < views.size(); ++i)
        facing += dot(normal, add_scaled(views[i].origin, group_center, -1.0));
    if (facing < 0.0)
        for (int c = 0; c < 3; ++c)
            normal[c] = -normal[c];
    for (std::size_t i = 0; i < group.polygons.size(); ++i) {
        if (std::fabs(dot(normal, group.polygons[i].normal)) < 0.999)
            throw std::runtime_error("non-coplanar window normals in '" +
                                     group.geometry_path + "'");
        group.polygons[i].normal = normal;
    }
    group.normal = normal;
    Vec3 projected_up = add_scaled(up, normal, -dot(up, normal));
    if (norm(projected_up) <= kEpsilon)
        throw std::runtime_error("window up vector is parallel to normal for '" +
                                 group.geometry_path + "'");
    group.axis[2] = normal;
    group.axis[0] = normalized(cross(projected_up, normal),
                               "window local X axis");
    group.axis[1] = normalized(cross(normal, group.axis[0]),
                               "window local Y axis");
}

Vec3 world_to_local(const WindowGroup &group, const Vec3 &direction)
{
    return Vec3{{dot(group.axis[0], direction),
                 dot(group.axis[1], direction),
                 dot(group.axis[2], direction)}};
}

Vec3 local_to_world(const WindowGroup &group, const Vec3 &direction)
{
    Vec3 result = {{0.0, 0.0, 0.0}};
    for (int axis = 0; axis < 3; ++axis)
        for (int c = 0; c < 3; ++c)
            result[c] += group.axis[axis][c]*direction[axis];
    return normalized(result, "sampled BSDF direction");
}

bool ray_intersects_polygon(const Vec3 &origin, const Vec3 &direction,
                            const Polygon &polygon, double &distance)
{
    const double denominator = dot(direction, polygon.normal);
    if (std::fabs(denominator) <= kEpsilon)
        return false;
    distance = dot(add_scaled(polygon.vertices[0], origin, -1.0),
                   polygon.normal)/denominator;
    if (distance <= kEpsilon)
        return false;
    const Vec3 point = add_scaled(origin, direction, distance);
    double sign = 0.0;
    for (std::size_t i = 0; i < polygon.vertices.size(); ++i) {
        const Vec3 &a = polygon.vertices[i];
        const Vec3 &b = polygon.vertices[(i+1)%polygon.vertices.size()];
        const double edge_side = dot(cross(add_scaled(b, a, -1.0),
                                           add_scaled(point, a, -1.0)),
                                     polygon.normal);
        if (std::fabs(edge_side) <= 1.0e-9)
            continue;
        if (sign == 0.0)
            sign = edge_side;
        else if (sign*edge_side < 0.0)
            return false;
    }
    return true;
}

bool ray_intersects_group(const Viewpoint &view, const Vec3 &direction,
                          const WindowGroup &group, double &distance)
{
    bool hit = false;
    distance = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < group.polygons.size(); ++i) {
        double candidate_distance;
        if (ray_intersects_polygon(view.origin, direction, group.polygons[i],
                                   candidate_distance) &&
                candidate_distance < distance) {
            hit = true;
            distance = candidate_distance;
        }
    }
    return hit;
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
    if (open_process(&process_data, argv.data()) <= 0)
        throw std::runtime_error("cannot start " + label + " ('" +
                                 arguments[0] + "')");

    const int write_fd = process_data.w;
    std::atomic<bool> write_ok(true);
    std::thread writer([&input, write_fd, &write_ok]() {
        const char *position = reinterpret_cast<const char *>(input.data());
        std::size_t remaining = input.size()*sizeof(float);
        while (remaining) {
            const std::size_t request = std::min<std::size_t>(
                remaining,
                static_cast<std::size_t>(std::numeric_limits<int>::max()));
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

void accept_pending(const PendingContribution &pending,
                    std::vector<double> &contrast,
                    std::vector<GlareDirectionSample> *peak_samples)
{
    if (pending.peak_sample_index ==
            std::numeric_limits<std::size_t>::max()) {
        contrast[pending.result_index] += pending.value;
        return;
    }
    if (peak_samples == NULL ||
            pending.peak_sample_index >= peak_samples->size())
        throw std::runtime_error("invalid pending peak-sample index");
    (*peak_samples)[pending.peak_sample_index].visible = true;
}

void flush_visibility(const Options &options,
                      std::vector<PendingContribution> &pending,
                      std::vector<double> &contrast,
                      std::vector<GlareDirectionSample> *peak_samples = NULL)
{
    if (pending.empty())
        return;
    if (options.visibility_octree.empty()) {
        for (std::size_t i = 0; i < pending.size(); ++i)
            accept_pending(pending[i], contrast, peak_samples);
        pending.clear();
        return;
    }
    std::vector<float> rays;
    rays.reserve(pending.size()*6);
    for (std::size_t i = 0; i < pending.size(); ++i) {
        for (int c = 0; c < 3; ++c)
            rays.push_back(static_cast<float>(pending[i].origin[c]));
        for (int c = 0; c < 3; ++c)
            rays.push_back(static_cast<float>(pending[i].direction[c]));
    }
    std::vector<std::string> command;
    command.push_back(options.rtrace);
    command.push_back("-h");
    command.push_back("-fff");
    command.push_back("-ab");
    command.push_back("0");
    command.push_back("-oL");
    command.push_back("-n");
    command.push_back(std::to_string(options.nproc));
    command.push_back(options.visibility_octree);
    const std::vector<char> output = run_process(
        command, rays, "rtrace visibility query");
    if (output.size() != pending.size()*sizeof(float))
        throw std::runtime_error("unexpected rtrace visibility output size");
    for (std::size_t i = 0; i < pending.size(); ++i) {
        float first_hit;
        std::memcpy(&first_hit, &output[i*sizeof(float)], sizeof(float));
        if (!std::isfinite(first_hit) || first_hit < 0.0f)
            throw std::runtime_error("invalid rtrace visibility distance");
        if (first_hit <= 0.0f ||
                first_hit+options.visibility_tolerance >=
                    pending[i].target_distance)
            accept_pending(pending[i], contrast, peak_samples);
    }
    pending.clear();
}

void view_basis(const Vec3 &forward, const Vec3 &up, Vec3 &right, Vec3 &local_up)
{
    local_up = add_scaled(up, forward, -dot(up, forward));
    if (norm(local_up) <= kEpsilon) {
        const Vec3 fallback = {{0.0, 1.0, 0.0}};
        local_up = add_scaled(fallback, forward, -dot(fallback, forward));
    }
    local_up = normalized(local_up, "view up vector");
    right = normalized(cross(forward, local_up), "view horizontal vector");
}

double guth_position_index(const Vec3 &source_direction,
                           const Vec3 &forward, const Vec3 &up)
{
    Vec3 horizontal, local_up;
    view_basis(forward, up, horizontal, local_up);
    const Vec3 temp = normalized(cross(forward, horizontal),
                                 "view vertical vector");
    const double phi = angle(source_direction, temp) - PI/2.0;
    double position;
    if (phi >= 0.0) {
        const double sigma = angle(source_direction, forward)*180.0/PI;
        const double forward_dot = std::max(dot(source_direction, forward),
                                             kEpsilon);
        const Vec3 projected = normalized(
            add_scaled(forward, source_direction, 1.0/forward_dot),
            "Guth projected direction");
        const double tau = angle(projected, local_up)*180.0/PI;
        position = std::exp(
            (35.2 - 0.31889*tau - 1.22*std::exp(-2.0*tau/9.0))
                /1000.0*sigma +
            (21.0 + 0.26667*tau - 0.002963*tau*tau)
                /100000.0*sigma*sigma);
    } else {
        double theta = PI/2.0 - angle(source_direction, horizontal);
        if (std::fabs(theta) <= kEpsilon)
            theta = kEpsilon;
        double tangent_phi = std::tan(phi);
        if (std::fabs(tangent_phi) <= kEpsilon)
            tangent_phi = tangent_phi < 0.0 ? -kEpsilon : kEpsilon;
        const double d = 1.0/tangent_phi;
        const double s = std::tan(theta)/tangent_phi;
        const double radius = std::sqrt((1.0+s*s)/(d*d));
        position = 1.0 + (radius > 0.6 ? 1.2 : 0.8)*std::min(radius, 3.0);
    }
    return std::min(position, 16.0);
}

void concentric_disk(double u, double v, double &x, double &y)
{
    const double a = 2.0*u-1.0;
    const double b = 2.0*v-1.0;
    if (std::fabs(a) <= kEpsilon && std::fabs(b) <= kEpsilon) {
        x = y = 0.0;
        return;
    }
    double radius, phi;
    if (a*a > b*b) {
        radius = a;
        phi = PI/4.0*(b/a);
    } else {
        radius = b;
        phi = PI/2.0-PI/4.0*(a/b);
    }
    x = radius*std::cos(phi);
    y = radius*std::sin(phi);
}

Vec3 equal_solid_angle_hemisphere(double u, double v)
{
    double disk_x, disk_y;
    concentric_disk(u, v, disk_x, disk_y);
    const double radius2 = std::min(1.0, disk_x*disk_x+disk_y*disk_y);
    const double z = 1.0-radius2;
    const double radial = std::sqrt(radius2);
    if (radial <= kEpsilon)
        return Vec3{{0.0, 0.0, 1.0}};
    const double sine = std::sqrt(std::max(0.0, 1.0-z*z));
    return Vec3{{sine*disk_x/radial, sine*disk_y/radial, z}};
}

void inverse_concentric_disk(double x, double y, double &u, double &v)
{
    const double radius = std::min(1.0, std::sqrt(x*x+y*y));
    if (radius <= kEpsilon) {
        u = v = 0.5;
        return;
    }
    double phi = std::atan2(y, x);
    double a, b;
    if (phi >= -PI/4.0 && phi < PI/4.0) {
        a = radius;
        b = phi*a/(PI/4.0);
    } else if (phi >= PI/4.0 && phi < 3.0*PI/4.0) {
        b = radius;
        a = -(phi-PI/2.0)*b/(PI/4.0);
    } else if (phi >= 3.0*PI/4.0 || phi < -3.0*PI/4.0) {
        a = -radius;
        const double wrapped = phi >= 0.0 ? phi-PI : phi+PI;
        b = wrapped*a/(PI/4.0);
    } else {
        b = -radius;
        a = -(phi+PI/2.0)*b/(PI/4.0);
    }
    u = std::max(0.0, std::min(1.0, 0.5*(a+1.0)));
    v = std::max(0.0, std::min(1.0, 0.5*(b+1.0)));
}

void hemisphere_to_equal_solid_angle(const Vec3 &direction,
                                     double &u, double &v)
{
    const double z = std::max(0.0, std::min(1.0, direction[2]));
    const double disk_radius = std::sqrt(std::max(0.0, 1.0-z));
    const double xy_radius = std::sqrt(direction[0]*direction[0] +
                                       direction[1]*direction[1]);
    if (xy_radius <= kEpsilon) {
        u = v = 0.5;
        return;
    }
    inverse_concentric_disk(disk_radius*direction[0]/xy_radius,
                            disk_radius*direction[1]/xy_radius, u, v);
}

std::vector<Vec3> solar_disk_directions(const Sun &sun, int count)
{
    if (count == 1)
        return std::vector<Vec3>(1, sun.direction);
    const int side = static_cast<int>(std::sqrt(static_cast<double>(count)));
    if (side*side != count)
        throw std::runtime_error("sun-disk sample count must be a perfect square");
    Vec3 reference = std::fabs(sun.direction[2]) < 0.9 ?
        Vec3{{0.0, 0.0, 1.0}} : Vec3{{0.0, 1.0, 0.0}};
    const Vec3 axis_x = normalized(cross(reference, sun.direction),
                                   "solar disk X axis");
    const Vec3 axis_y = normalized(cross(sun.direction, axis_x),
                                   "solar disk Y axis");
    const double cosine_radius = std::cos(0.5*sun.angular_diameter);
    std::vector<Vec3> directions;
    directions.reserve(count);
    for (int row = 0; row < side; ++row)
        for (int column = 0; column < side; ++column) {
            double x, y;
            concentric_disk((column+0.5)/side, (row+0.5)/side, x, y);
            const double radius2 = x*x+y*y;
            const double cosine = 1.0-radius2*(1.0-cosine_radius);
            const double sine = std::sqrt(std::max(0.0, 1.0-cosine*cosine));
            const double radial = std::sqrt(radius2);
            Vec3 direction = {{sun.direction[0]*cosine,
                               sun.direction[1]*cosine,
                               sun.direction[2]*cosine}};
            if (radial > kEpsilon) {
                for (int c = 0; c < 3; ++c)
                    direction[c] += sine*(axis_x[c]*x+axis_y[c]*y)/radial;
            }
            directions.push_back(normalized(direction, "solar disk direction"));
        }
    return directions;
}

void copy_to_fvect(FVECT target, const Vec3 &source)
{
    for (int c = 0; c < 3; ++c)
        target[c] = source[c];
}

Vec3 copy_from_fvect(const FVECT source)
{
    return Vec3{{source[0], source[1], source[2]}};
}

double non_diffuse_bsdf(const WindowGroup &group, const Vec3 &incident,
                        const Vec3 &outgoing)
{
    FVECT in_vector, out_vector;
    copy_to_fvect(in_vector, incident);
    copy_to_fvect(out_vector, outgoing);
    SDValue value;
    const SDError error = SDevalBSDF(&value, in_vector, out_vector, group.bsdf);
    if (error != SDEnone)
        throw std::runtime_error("BSDF evaluation failed for '" +
                                 group.bsdf_path + "'");
    const double diffuse = incident[2] > 0.0 ?
        group.bsdf->tLambFront.cieY/PI : group.bsdf->tLambBack.cieY/PI;
    return std::max(0.0, value.cieY-diffuse);
}

const SDSpectralDF *transmission_distribution(const WindowGroup &group,
                                               const Vec3 &outgoing)
{
    if (outgoing[2] > 0.0)
        return group.bsdf->tb != NULL ? group.bsdf->tb : group.bsdf->tf;
    return group.bsdf->tf != NULL ? group.bsdf->tf : group.bsdf->tb;
}

ThroughComponent compute_absdf_through(const WindowGroup &group,
                                       const Vec3 &outgoing)
{
    static const double offsets[29][2] = {
        {0, 0}, {-0.6, 0}, {0, 0.6}, {0, -0.6}, {0.6, 0},
        {-0.6, 0.6}, {-0.6, -0.6}, {0.6, 0.6}, {0.6, -0.6},
        {-1.2, 0}, {0, 1.2}, {0, -1.2}, {1.2, 0},
        {-1.2, 1.2}, {-1.2, -1.2}, {1.2, 1.2}, {1.2, -1.2},
        {-1.8, 0}, {0, 1.8}, {0, -1.8}, {1.8, 0},
        {-1.8, 1.8}, {-1.8, -1.8}, {1.8, 1.8}, {1.8, -1.8},
        {-2.4, 0}, {0, 2.4}, {0, -2.4}, {2.4, 0}
    };
    struct PeakSample {
        Vec3 direction;
        double value;
    };

    ThroughComponent result;
    const SDSpectralDF *distribution = transmission_distribution(group,
                                                                  outgoing);
    if (distribution == NULL || distribution->minProjSA <= 0.0)
        return result;
    result.min_projected_solid_angle = distribution->minProjSA;
    const double search_radius = std::sqrt(distribution->minProjSA);
    std::vector<PeakSample> samples;
    samples.reserve(29);
    FVECT outgoing_vector;
    copy_to_fvect(outgoing_vector, outgoing);
    for (int index = 0; index < 29; ++index) {
        PeakSample sample;
        sample.direction = normalized(
            Vec3{{-outgoing[0] + offsets[index][0]*search_radius,
                  -outgoing[1] + offsets[index][1]*search_radius,
                  -outgoing[2]}}, "aBSDF through probe direction");
        FVECT probe_vector;
        copy_to_fvect(probe_vector, sample.direction);
        SDValue value;
        const SDError error = SDevalBSDF(&value, outgoing_vector,
                                         probe_vector, group.bsdf);
        if (error != SDEnone)
            throw std::runtime_error("aBSDF through-peak evaluation failed for '" +
                                     group.bsdf_path + "'");
        sample.value = value.cieY;
        samples.push_back(sample);
    }
    std::sort(samples.begin(), samples.end(),
              [](const PeakSample &left, const PeakSample &right) {
                  return left.value > right.value;
              });
    if (samples.front().value <= kEpsilon)
        return result;

    double peak_integral = 0.0;
    double surround_integral = 0.0;
    double peak_solid_angle = 0.0;
    double surround_solid_angle = 0.0;
    double peak_value_sum = 0.0;
    int peak_count = 0;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index && samples[index].value == samples[index-1].value)
            continue;
        FVECT probe_vector;
        copy_to_fvect(probe_vector, samples[index].direction);
        double projected_solid_angle;
        const SDError error = SDsizeBSDF(
            &projected_solid_angle, outgoing_vector, probe_vector,
            SDqueryMin, group.bsdf);
        if (error != SDEnone)
            throw std::runtime_error("aBSDF through-peak size query failed for '" +
                                     group.bsdf_path + "'");
        const double integrated_value =
            samples[index].value*projected_solid_angle;
        if (projected_solid_angle > 1.5*distribution->minProjSA ||
                peak_value_sum > 8.0*samples[index].value*peak_count) {
            if (!index)
                return result;
            surround_integral += integrated_value;
            surround_solid_angle += projected_solid_angle;
            continue;
        }
        peak_integral += integrated_value;
        peak_solid_angle += projected_solid_angle;
        peak_value_sum += samples[index].value;
        ++peak_count;
    }
    if (surround_solid_angle < 0.2*peak_solid_angle)
        return result;

    const double diffuse = outgoing[2] > 0.0 ?
        group.bsdf->tLambFront.cieY/PI : group.bsdf->tLambBack.cieY/PI;
    result.coefficient = std::max(0.0,
                                  peak_integral-peak_solid_angle*diffuse);
    result.surround = std::max(0.0,
        surround_integral/surround_solid_angle-diffuse);
    result.present = result.coefficient >= 0.0005;
    return result;
}

bool absdf_through_overlap(const Sun &sun, const Vec3 &incident,
                           const Vec3 &outgoing,
                           double min_projected_solid_angle)
{
    if (incident[2]*outgoing[2] >= 0.0 ||
            min_projected_solid_angle <= 0.0)
        return false;
    const double dx = incident[0] + outgoing[0];
    const double dy = incident[1] + outgoing[1];
    const double source_projected_solid_angle =
        sun.omega*std::fabs(incident[2]);
    const double overlap_limit = (2.5*4.0/PI)*
        (source_projected_solid_angle + min_projected_solid_angle +
         2.0*std::sqrt(source_projected_solid_angle*
                       min_projected_solid_angle));
    return dx*dx + dy*dy <= overlap_limit;
}

double redirected_absdf(const Sun &sun, const Vec3 &incident,
                        const Vec3 &outgoing,
                        const ThroughComponent &through, double full_value)
{
    if (!through.present)
        return full_value;
    return absdf_through_overlap(sun, incident, outgoing,
                                 through.min_projected_solid_angle) ?
        through.surround : full_value;
}

double source_luminance_for_outgoing(
        const Options &options, const WindowGroup &group, const Sun &sun,
        const std::vector<Vec3> &incident,
        const std::vector<double> &incident_cosine,
        const std::vector<double> &tau, double inverse_disk,
        const Vec3 &outgoing, double *proposal = NULL)
{
    if (outgoing[2] <= kEpsilon)
        return 0.0;
    ThroughComponent through;
    if (group.is_absdf && !options.include_absdf_through) {
        const SDSpectralDF *distribution = transmission_distribution(
            group, outgoing);
        const double min_projected_solid_angle = distribution == NULL ?
            0.0 : distribution->minProjSA;
        for (std::size_t k = 0; k < incident.size(); ++k)
            if (tau[k] > kEpsilon && absdf_through_overlap(
                    sun, incident[k], outgoing,
                    min_projected_solid_angle)) {
                through = compute_absdf_through(group, outgoing);
                break;
            }
    }

    double radiance_factor = 0.0;
    double proposal_sum = 0.0;
    for (std::size_t k = 0; k < incident.size(); ++k) {
        if (tau[k] <= kEpsilon)
            continue;
        const double full_value = non_diffuse_bsdf(
            group, incident[k], outgoing);
        const double target_value = through.present ?
            redirected_absdf(sun, incident[k], outgoing, through,
                              full_value) : full_value;
        radiance_factor += incident_cosine[k]*target_value;
        proposal_sum += full_value*outgoing[2]/tau[k];
    }
    radiance_factor *= inverse_disk;
    proposal_sum *= inverse_disk;
    if (proposal != NULL)
        *proposal = proposal_sum;
    return radiance_factor <= kEpsilon ? 0.0 :
        sun.luminance*sun.omega*radiance_factor;
}

AdaptivePeakCell evaluate_adaptive_cell(
        const Options &options, const WindowGroup &group, const Sun &sun,
        const std::vector<Vec3> &incident,
        const std::vector<double> &incident_cosine,
        const std::vector<double> &tau, double inverse_disk,
        double map_u0, double map_v0, double map_size, int level)
{
    AdaptivePeakCell cell;
    cell.map_u0 = map_u0;
    cell.map_v0 = map_v0;
    cell.map_size = map_size;
    cell.level = level;
    cell.outgoing = equal_solid_angle_hemisphere(
        map_u0+0.5*map_size, map_v0+0.5*map_size);
    cell.luminance = source_luminance_for_outgoing(
        options, group, sun, incident, incident_cosine, tau, inverse_disk,
        cell.outgoing);
    return cell;
}

bool candidate_in_cell(const PeakCandidate &candidate,
                       const AdaptivePeakCell &cell)
{
    const double upper_u = cell.map_u0+cell.map_size;
    const double upper_v = cell.map_v0+cell.map_size;
    return candidate.map_u >= cell.map_u0-kEpsilon &&
           candidate.map_u <= upper_u+kEpsilon &&
           candidate.map_v >= cell.map_v0-kEpsilon &&
           candidate.map_v <= upper_v+kEpsilon;
}

void refine_adaptive_peak_cell(
        const Options &options, const WindowGroup &group, const Sun &sun,
        const std::vector<Vec3> &incident,
        const std::vector<double> &incident_cosine,
        const std::vector<double> &tau, double inverse_disk,
        const std::vector<PeakCandidate> &candidates,
        const std::vector<std::size_t> &candidate_indices,
        const AdaptivePeakCell &cell,
        std::vector<AdaptivePeakCell> &leaves, bool &unresolved)
{
    if (unresolved)
        return;
    double candidate_max = 0.0;
    std::size_t brightest_candidate = std::numeric_limits<std::size_t>::max();
    for (std::size_t i = 0; i < candidate_indices.size(); ++i) {
        const std::size_t index = candidate_indices[i];
        if (candidates[index].luminance > candidate_max) {
            candidate_max = candidates[index].luminance;
            brightest_candidate = index;
        }
    }
    if (cell.level >= options.adaptive_max_level) {
        if (cell.luminance > options.threshold) {
            leaves.push_back(cell);
        } else if (candidate_max > options.threshold &&
                   brightest_candidate !=
                       std::numeric_limits<std::size_t>::max())
            unresolved = true;
        return;
    }

    const double child_size = 0.5*cell.map_size;
    std::array<AdaptivePeakCell, 4> child;
    std::array<std::vector<std::size_t>, 4> child_candidates;
    for (int row = 0; row < 2; ++row)
        for (int column = 0; column < 2; ++column) {
            const int index = 2*row+column;
            child[index] = evaluate_adaptive_cell(
                options, group, sun, incident, incident_cosine, tau,
                inverse_disk, cell.map_u0+column*child_size,
                cell.map_v0+row*child_size, child_size, cell.level+1);
        }
    for (std::size_t i = 0; i < candidate_indices.size(); ++i) {
        const std::size_t candidate_index = candidate_indices[i];
        for (int child_index = 0; child_index < 4; ++child_index)
            if (candidate_in_cell(candidates[candidate_index],
                                  child[child_index])) {
                child_candidates[child_index].push_back(candidate_index);
                break;
            }
    }

    double child_min = std::numeric_limits<double>::infinity();
    double child_max = 0.0;
    double child_measure = 0.0;
    int active_children = 0;
    const double child_omega = 2.0*PI*child_size*child_size;
    for (int index = 0; index < 4; ++index) {
        child_min = std::min(child_min, child[index].luminance);
        child_max = std::max(child_max, child[index].luminance);
        if (child[index].luminance > options.threshold) {
            child_measure += child[index].luminance*child[index].luminance*
                             child_omega;
            ++active_children;
        }
    }
    const double cell_omega = 2.0*PI*cell.map_size*cell.map_size;
    const double parent_measure = cell.luminance > options.threshold ?
        cell.luminance*cell.luminance*cell_omega : 0.0;
    const double measure_scale = std::max(
        std::max(parent_measure, child_measure),
        options.threshold*options.threshold*cell_omega*1.0e-3);
    const double contrast_uncertainty =
        std::fabs(child_measure-parent_measure)/measure_scale;
    const double gradient = child_max > kEpsilon ?
        (child_max-child_min)/child_max : 0.0;
    const bool parent_active = cell.luminance > options.threshold;
    const bool threshold_crossing =
        (active_children > 0 && active_children < 4) ||
        (parent_active != (active_children >= 2));
    const bool unresolved_candidate =
        candidate_max > options.threshold && child_max <= options.threshold;
    if (unresolved_candidate) {
        /* The peak is narrower than the current quadrature support.  Do not
         * assign it an arbitrary leaf-cell solid angle; the caller falls back
         * to the original importance-sample integration weights instead. */
        unresolved = true;
        return;
    }
    const bool refine = threshold_crossing ||
        (child_max > options.threshold &&
         (contrast_uncertainty > options.adaptive_contrast_tolerance ||
          gradient > options.adaptive_gradient_tolerance));

    if (refine) {
        for (int index = 0; index < 4; ++index)
            refine_adaptive_peak_cell(
                options, group, sun, incident, incident_cosine, tau,
                inverse_disk, candidates, child_candidates[index],
                child[index], leaves, unresolved);
        return;
    }

    for (int index = 0; index < 4; ++index)
        if (child[index].luminance > options.threshold)
            leaves.push_back(child[index]);
}

std::vector<AdaptivePeakCell> reconstruct_adaptive_peak_cells(
        const Options &options, const WindowGroup &group, const Sun &sun,
        const std::vector<Vec3> &incident,
        const std::vector<double> &incident_cosine,
        const std::vector<double> &tau, double inverse_disk,
        const std::vector<PeakCandidate> &candidates, bool &resolved)
{
    std::vector<AdaptivePeakCell> leaves;
    resolved = true;
    if (candidates.empty())
        return leaves;
    const int side = 1 << options.adaptive_base_level;
    const double size = 1.0/side;
    typedef std::pair<int, int> CellIndex;
    std::set<CellIndex> scheduled;
    std::vector<CellIndex> queue;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const int center_x = std::min(side-1, std::max(0,
            static_cast<int>(candidates[i].map_u*side)));
        const int center_y = std::min(side-1, std::max(0,
            static_cast<int>(candidates[i].map_v*side)));
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const int x = center_x+dx;
                const int y = center_y+dy;
                if (x < 0 || x >= side || y < 0 || y >= side)
                    continue;
                if (scheduled.insert(CellIndex(x, y)).second)
                    queue.push_back(CellIndex(x, y));
            }
    }

    std::map<CellIndex, AdaptivePeakCell> base_cells;
    for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
        const CellIndex index = queue[cursor];
        AdaptivePeakCell cell = evaluate_adaptive_cell(
            options, group, sun, incident, incident_cosine, tau,
            inverse_disk, index.first*size, index.second*size, size,
            options.adaptive_base_level);
        base_cells[index] = cell;
        bool contains_candidate = false;
        for (std::size_t i = 0; i < candidates.size(); ++i)
            if (candidate_in_cell(candidates[i], cell)) {
                contains_candidate = true;
                break;
            }
        if (!contains_candidate && cell.luminance <= options.threshold)
            continue;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const int x = index.first+dx;
                const int y = index.second+dy;
                if (x < 0 || x >= side || y < 0 || y >= side)
                    continue;
                if (scheduled.insert(CellIndex(x, y)).second)
                    queue.push_back(CellIndex(x, y));
            }
    }

    for (std::map<CellIndex, AdaptivePeakCell>::const_iterator iterator =
            base_cells.begin(); iterator != base_cells.end(); ++iterator) {
        std::vector<std::size_t> local_candidates;
        for (std::size_t i = 0; i < candidates.size(); ++i)
            if (candidate_in_cell(candidates[i], iterator->second))
                local_candidates.push_back(i);
        bool unresolved = false;
        refine_adaptive_peak_cell(
            options, group, sun, incident, incident_cosine, tau,
            inverse_disk, candidates, local_candidates, iterator->second,
            leaves, unresolved);
        if (unresolved) {
            resolved = false;
            leaves.clear();
            break;
        }
    }
    return leaves;
}

double boundary_map_luminance(
        const Options &options, const WindowGroup &group, const Sun &sun,
        const std::vector<Vec3> &incident,
        const std::vector<double> &incident_cosine,
        const std::vector<double> &tau, double inverse_disk,
        double map_u, double map_v, Vec3 *outgoing = NULL)
{
    const Vec3 direction = equal_solid_angle_hemisphere(
        std::max(0.0, std::min(1.0, map_u)),
        std::max(0.0, std::min(1.0, map_v)));
    if (outgoing != NULL)
        *outgoing = direction;
    return source_luminance_for_outgoing(
        options, group, sun, incident, incident_cosine, tau, inverse_disk,
        direction);
}

double boundary_ray_extent(double map_u, double map_v,
                           double direction_u, double direction_v)
{
    double extent = std::numeric_limits<double>::infinity();
    if (direction_u > kEpsilon)
        extent = std::min(extent, (1.0-map_u)/direction_u);
    else if (direction_u < -kEpsilon)
        extent = std::min(extent, -map_u/direction_u);
    if (direction_v > kEpsilon)
        extent = std::min(extent, (1.0-map_v)/direction_v);
    else if (direction_v < -kEpsilon)
        extent = std::min(extent, -map_v/direction_v);
    return std::max(0.0, extent);
}

bool point_in_boundary(const std::array<double, 2> &point,
                       const std::vector<std::array<double, 2> > &boundary)
{
    if (boundary.size() < 3)
        return false;
    bool inside = false;
    for (std::size_t first = 0, second = boundary.size()-1;
            first < boundary.size(); second = first++) {
        const double first_v = boundary[first][1];
        const double second_v = boundary[second][1];
        const bool crosses = (first_v > point[1]) != (second_v > point[1]);
        if (!crosses)
            continue;
        const double crossing_u =
            (boundary[second][0]-boundary[first][0])*
            (point[1]-first_v)/(second_v-first_v)+boundary[first][0];
        if (point[0] < crossing_u)
            inside = !inside;
    }
    return inside;
}

std::vector<std::vector<std::size_t> > cluster_peak_candidates(
        const Options &options, const std::vector<PeakCandidate> &candidates)
{
    const std::size_t count = candidates.size();
    std::vector<std::size_t> parent(count);
    std::vector<unsigned char> rank(count, 0);
    std::vector<double> radius(count, 0.0);
    for (std::size_t index = 0; index < count; ++index) {
        parent[index] = index;
        radius[index] = equivalent_cap_radius(candidates[index].solid_angle);
    }
    const double max_link = options.peak_max_link_angle*PI/180.0;
    for (std::size_t first = 0; first < count; ++first)
        for (std::size_t second = first+1; second < count; ++second) {
            const double link = std::min(
                max_link, options.peak_cluster_factor*
                (radius[first]+radius[second]));
            if (link > 0.0 && dot(candidates[first].outgoing,
                                  candidates[second].outgoing) >=
                                  std::cos(link))
                join_peaks(parent, rank, first, second);
        }
    std::map<std::size_t, std::vector<std::size_t> > grouped;
    for (std::size_t index = 0; index < count; ++index)
        grouped[peak_root(parent, index)].push_back(index);
    std::vector<std::vector<std::size_t> > clusters;
    for (std::map<std::size_t, std::vector<std::size_t> >::const_iterator
            iterator = grouped.begin(); iterator != grouped.end(); ++iterator)
        clusters.push_back(iterator->second);
    std::sort(clusters.begin(), clusters.end(),
        [&candidates](const std::vector<std::size_t> &first,
                      const std::vector<std::size_t> &second) {
            double first_max = 0.0, second_max = 0.0;
            for (std::size_t i = 0; i < first.size(); ++i)
                first_max = std::max(
                    first_max, candidates[first[i]].luminance);
            for (std::size_t i = 0; i < second.size(); ++i)
                second_max = std::max(
                    second_max, candidates[second[i]].luminance);
            return first_max > second_max;
        });
    return clusters;
}

bool trace_candidate_boundary(
        const Options &options, const WindowGroup &group, const Sun &sun,
        const std::vector<Vec3> &incident,
        const std::vector<double> &incident_cosine,
        const std::vector<double> &tau, double inverse_disk,
        const PeakCandidate &anchor,
        std::vector<std::array<double, 2> > &boundary)
{
    boundary.clear();
    boundary.reserve(options.boundary_rays);
    const double initial_radius =
        0.5/static_cast<double>(1 << options.adaptive_base_level);
    for (int ray = 0; ray < options.boundary_rays; ++ray) {
        const double azimuth = 2.0*PI*ray/options.boundary_rays;
        const double direction_u = std::cos(azimuth);
        const double direction_v = std::sin(azimuth);
        const double extent = boundary_ray_extent(
            anchor.map_u, anchor.map_v, direction_u, direction_v);
        if (extent <= kEpsilon)
            return false;
        double inside = 0.0;
        double outside = std::min(initial_radius, extent);
        double luminance = boundary_map_luminance(
            options, group, sun, incident, incident_cosine, tau,
            inverse_disk, anchor.map_u+outside*direction_u,
            anchor.map_v+outside*direction_v);
        while (luminance > options.threshold &&
                outside < extent-kEpsilon) {
            inside = outside;
            outside = std::min(extent, 2.0*outside);
            luminance = boundary_map_luminance(
                options, group, sun, incident, incident_cosine, tau,
                inverse_disk, anchor.map_u+outside*direction_u,
                anchor.map_v+outside*direction_v);
        }
        double radius;
        if (luminance > options.threshold &&
                outside >= extent-kEpsilon) {
            /* A compact peak must cross the luminance threshold before the
             * edge of the equal-solid-angle map.  A clipped contour does not
             * define a closed physical glare source. */
            return false;
        } else {
            for (int iteration = 0;
                    iteration < options.boundary_bisections; ++iteration) {
                const double middle = 0.5*(inside+outside);
                const double middle_luminance = boundary_map_luminance(
                    options, group, sun, incident, incident_cosine, tau,
                    inverse_disk, anchor.map_u+middle*direction_u,
                    anchor.map_v+middle*direction_v);
                if (middle_luminance > options.threshold)
                    inside = middle;
                else
                    outside = middle;
            }
            radius = 0.5*(inside+outside);
        }
        boundary.push_back(std::array<double, 2>{{
            anchor.map_u+radius*direction_u,
            anchor.map_v+radius*direction_v}});
    }
    return boundary.size() >= 8;
}

double boundary_polygon_area(
        const std::vector<std::array<double, 2> > &boundary)
{
    if (boundary.size() < 3)
        return 0.0;
    double twice_area = 0.0;
    for (std::size_t index = 0; index < boundary.size(); ++index) {
        const std::array<double, 2> &first = boundary[index];
        const std::array<double, 2> &second =
            boundary[(index+1)%boundary.size()];
        twice_area += first[0]*second[1]-second[0]*first[1];
    }
    return 0.5*std::fabs(twice_area);
}

std::vector<BoundaryPeakSample> quadrature_boundary_source(
        const Options &options, const WindowGroup &group, const Sun &sun,
        const std::vector<Vec3> &incident,
        const std::vector<double> &incident_cosine,
        const std::vector<double> &tau, double inverse_disk,
        const std::array<double, 2> &anchor,
        const std::vector<std::array<double, 2> > &boundary)
{
    std::vector<BoundaryPeakSample> samples;
    samples.reserve(3*boundary.size());
    /* Symmetric three-point quadrature on each anchor-boundary triangle.
     * The Shirley-Chiu map has constant 2*pi sr per unit square area, so the
     * triangle weights are physical solid-angle weights without a Jacobian. */
    static const double barycentric[3][3] = {
        {2.0/3.0, 1.0/6.0, 1.0/6.0},
        {1.0/6.0, 2.0/3.0, 1.0/6.0},
        {1.0/6.0, 1.0/6.0, 2.0/3.0}
    };
    for (std::size_t edge = 0; edge < boundary.size(); ++edge) {
        const std::array<double, 2> &first = boundary[edge];
        const std::array<double, 2> &second =
            boundary[(edge+1)%boundary.size()];
        const double twice_map_area = std::fabs(
            (first[0]-anchor[0])*(second[1]-anchor[1])-
            (second[0]-anchor[0])*(first[1]-anchor[1]));
        const double sample_omega = PI*twice_map_area/3.0;
        if (sample_omega <= kEpsilon)
            continue;
        for (int point = 0; point < 3; ++point) {
            const double map_u =
                barycentric[point][0]*anchor[0]+
                barycentric[point][1]*first[0]+
                barycentric[point][2]*second[0];
            const double map_v =
                barycentric[point][0]*anchor[1]+
                barycentric[point][1]*first[1]+
                barycentric[point][2]*second[1];
            BoundaryPeakSample sample;
            sample.luminance = boundary_map_luminance(
                options, group, sun, incident, incident_cosine, tau,
                inverse_disk, map_u, map_v, &sample.outgoing);
            if (sample.luminance <= options.threshold)
                continue;
            sample.solid_angle = sample_omega;
            samples.push_back(sample);
        }
    }
    return samples;
}

std::vector<BoundaryPeakSource> reconstruct_candidate_boundary_sources(
        const Options &options, const WindowGroup &group, const Sun &sun,
        const std::vector<Vec3> &incident,
        const std::vector<double> &incident_cosine,
        const std::vector<double> &tau, double inverse_disk,
        const std::vector<PeakCandidate> &candidates)
{
    std::vector<BoundaryPeakSource> sources;
    const std::vector<std::vector<std::size_t> > clusters =
        cluster_peak_candidates(options, candidates);
    for (std::size_t cluster_index = 0;
            cluster_index < clusters.size(); ++cluster_index) {
        const std::vector<std::size_t> &indices = clusters[cluster_index];
        std::size_t anchor_index = indices.front();
        for (std::size_t i = 1; i < indices.size(); ++i)
            if (candidates[indices[i]].luminance >
                    candidates[anchor_index].luminance)
                anchor_index = indices[i];
        const std::array<double, 2> anchor_map = {{
            candidates[anchor_index].map_u,
            candidates[anchor_index].map_v}};
        std::size_t covered_source = sources.size();
        for (std::size_t source = 0; source < sources.size(); ++source)
            if (sources[source].resolved &&
                    point_in_boundary(anchor_map, sources[source].boundary)) {
                covered_source = source;
                break;
            }
        if (covered_source < sources.size()) {
            sources[covered_source].candidate_indices.insert(
                sources[covered_source].candidate_indices.end(),
                indices.begin(), indices.end());
            continue;
        }
        BoundaryPeakSource source;
        source.candidate_indices = indices;
        source.anchor = anchor_map;
        if (trace_candidate_boundary(
                options, group, sun, incident, incident_cosine, tau,
                inverse_disk, candidates[anchor_index], source.boundary)) {
            source.solid_angle = 2.0*PI*
                boundary_polygon_area(source.boundary);
            source.resolved = source.solid_angle > kEpsilon &&
                (options.boundary_max_solid_angle <= 0.0 ||
                 source.solid_angle <= options.boundary_max_solid_angle);
        }
        sources.push_back(source);
    }
    for (std::size_t source_index = 0;
            source_index < sources.size(); ++source_index) {
        BoundaryPeakSource &source = sources[source_index];
        if (!source.resolved)
            continue;
        double importance_solid_angle = 0.0;
        for (std::size_t index = 0;
                index < source.candidate_indices.size(); ++index)
            importance_solid_angle += candidates[
                source.candidate_indices[index]].solid_angle;
        if (importance_solid_angle <= kEpsilon) {
            source.resolved = false;
            continue;
        }
        source.importance_solid_angle = importance_solid_angle;
        source.scale = source.solid_angle/importance_solid_angle;
        if (source.scale > 1.0) {
            if (!options.boundary_allow_expansion) {
                source.resolved = false;
                continue;
            }
            source.samples = quadrature_boundary_source(
                options, group, sun, incident, incident_cosine, tau,
                inverse_disk, source.anchor, source.boundary);
            source.solid_angle = 0.0;
            for (std::size_t index = 0; index < source.samples.size(); ++index)
                source.solid_angle += source.samples[index].solid_angle;
            if (source.solid_angle <= kEpsilon) {
                source.resolved = false;
                continue;
            }
            source.scale = source.solid_angle/importance_solid_angle;
            continue;
        }
        for (std::size_t index = 0;
                index < source.candidate_indices.size(); ++index) {
            const PeakCandidate &candidate = candidates[
                source.candidate_indices[index]];
            BoundaryPeakSample sample;
            sample.outgoing = candidate.outgoing;
            sample.luminance = candidate.luminance;
            sample.solid_angle = source.scale*candidate.solid_angle;
            source.samples.push_back(sample);
        }
    }
    return sources;
}

void record_boundary_statistics(
        const std::vector<BoundaryPeakSource> &sources,
        BoundaryStatistics *statistics)
{
    if (statistics == NULL)
        return;
    for (std::size_t index = 0; index < sources.size(); ++index) {
        const BoundaryPeakSource &source = sources[index];
        if (!source.resolved)
            continue;
        ++statistics->resolved_sources;
        statistics->scale_sum += source.scale;
        statistics->scale_min = std::min(
            statistics->scale_min, source.scale);
        statistics->scale_max = std::max(
            statistics->scale_max, source.scale);
        std::size_t bin;
        if (source.scale < 0.5)
            bin = 0;
        else if (source.scale < 0.75)
            bin = 1;
        else if (source.scale < 1.0)
            bin = 2;
        else if (source.scale < 1.25)
            bin = 3;
        else if (source.scale < 2.0)
            bin = 4;
        else
            bin = 5;
        ++statistics->scale_bins[bin];
        if (source.solid_angle < 1.0e-5)
            bin = 0;
        else if (source.solid_angle < 1.0e-4)
            bin = 1;
        else if (source.solid_angle < 1.0e-3)
            bin = 2;
        else if (source.solid_angle < 1.0e-2)
            bin = 3;
        else if (source.solid_angle < 1.0e-1)
            bin = 4;
        else
            bin = 5;
        ++statistics->solid_angle_bins[bin];
    }
}

void queue_boundary_peak_sources(
        const Options &options, const WindowGroup &group,
        std::size_t group_index, std::size_t time_index,
        const std::vector<Viewpoint> &views, std::size_t nsteps,
        const std::vector<PeakCandidate> &candidates,
        const std::vector<BoundaryPeakSource> &sources,
        std::vector<double> &contrast,
        std::vector<PendingContribution> &pending,
        std::vector<GlareDirectionSample> *peak_samples)
{
    for (std::size_t source_index = 0;
            source_index < sources.size(); ++source_index) {
        const BoundaryPeakSource &source = sources[source_index];
        std::vector<BoundaryPeakSample> samples = source.samples;
        if (!source.resolved) {
            samples.clear();
            for (std::size_t candidate = 0;
                    candidate < source.candidate_indices.size(); ++candidate) {
                const PeakCandidate &peak = candidates[
                    source.candidate_indices[candidate]];
                BoundaryPeakSample sample;
                sample.outgoing = peak.outgoing;
                sample.luminance = peak.luminance;
                sample.solid_angle = peak.solid_angle;
                samples.push_back(sample);
            }
        }
        for (std::size_t sample_index = 0;
                sample_index < samples.size(); ++sample_index) {
            const BoundaryPeakSample &sample = samples[sample_index];
            const Vec3 outgoing_world = local_to_world(group, sample.outgoing);
            const Vec3 source_direction = {{-outgoing_world[0],
                                             -outgoing_world[1],
                                             -outgoing_world[2]}};
            for (std::size_t view = 0; view < views.size(); ++view) {
                if (dot(source_direction, views[view].direction) <= kEpsilon)
                    continue;
                double target_distance;
                if (!ray_intersects_group(views[view], source_direction,
                                          group, target_distance))
                    continue;
                PendingContribution contribution;
                contribution.origin = views[view].origin;
                contribution.direction = source_direction;
                contribution.result_index = view*nsteps+time_index;
                contribution.target_distance = target_distance;
                GlareDirectionSample peak_sample;
                peak_sample.direction = source_direction;
                peak_sample.result_index = contribution.result_index;
                peak_sample.partition = group_index;
                peak_sample.source_id = source_index;
                peak_sample.luminance = sample.luminance;
                peak_sample.solid_angle = sample.solid_angle;
                peak_sample.coherent_source = source.resolved;
                contribution.peak_sample_index = peak_samples->size();
                contribution.value = 0.0;
                peak_samples->push_back(peak_sample);
                pending.push_back(contribution);
                if (pending.size() >=
                        static_cast<std::size_t>(options.visibility_batch))
                    flush_visibility(options, pending, contrast, peak_samples);
            }
        }
    }
}

PeakReconstructionResult integrate_group_sun(
                         const Options &options, const WindowGroup &group,
                         std::size_t group_index, const Sun &sun,
                         std::size_t time_index,
                         const std::vector<Viewpoint> &views,
                         std::vector<double> &contrast, std::size_t nsteps,
                         std::vector<PendingContribution> &pending,
                         std::vector<GlareDirectionSample> *peak_samples = NULL,
                         BoundaryStatistics *boundary_statistics = NULL)
{
    const std::vector<Vec3> disk_world = solar_disk_directions(
        sun, options.sun_disk_samples);
    std::vector<Vec3> incident(disk_world.size());
    std::vector<double> incident_cosine(disk_world.size(), 0.0);
    std::vector<double> tau(disk_world.size(), 0.0);
    for (std::size_t k = 0; k < disk_world.size(); ++k) {
        incident[k] = world_to_local(group, disk_world[k]);
        if (incident[k][2] >= -kEpsilon)
            continue;
        incident_cosine[k] = -incident[k][2];
        FVECT in_vector;
        copy_to_fvect(in_vector, incident[k]);
        tau[k] = SDdirectHemi(in_vector, SDsampSpT, group.bsdf);
    }

    srandom(static_cast<unsigned long>(options.seed) ^
            static_cast<unsigned long>(time_index*2654435761u) ^
            static_cast<unsigned long>(group.geometry_path.size()*2246822519u));
    const double inverse_samples = 1.0/options.samples;
    const double inverse_disk = 1.0/disk_world.size();
    std::vector<std::size_t> local_index(disk_world.size(), 0);
    std::vector<std::size_t> local_count(disk_world.size(), 0);
    std::vector<PeakCandidate> candidates;
    if (options.adaptive_peak_cells || options.candidate_boundary_peaks ||
            options.candidate_boundary_fallback)
        candidates.reserve(options.samples);
    for (int sample = 0; sample < options.samples; ++sample)
        ++local_count[static_cast<std::size_t>(sample)%disk_world.size()];

    for (int sample = 0; sample < options.samples; ++sample) {
        const std::size_t selected = static_cast<std::size_t>(sample)%disk_world.size();
        if (tau[selected] <= kEpsilon) {
            ++local_index[selected];
            continue;
        }
        const double random_x = (local_index[selected]++ + 0.5)/
                                local_count[selected];
        FVECT io_vector;
        copy_to_fvect(io_vector, incident[selected]);
        SDValue sampled_value;
        const SDError sample_error = SDsampBSDF(
            &sampled_value, io_vector, random_x, SDsampSpT, group.bsdf);
        if (sample_error != SDEnone)
            throw std::runtime_error("BSDF sampling failed for '" +
                                     group.bsdf_path + "'");
        const Vec3 outgoing = copy_from_fvect(io_vector);
        if (outgoing[2] <= kEpsilon)
            continue;
        double proposal = 0.0;
        const double source_luminance = source_luminance_for_outgoing(
            options, group, sun, incident, incident_cosine, tau,
            inverse_disk, outgoing, &proposal);
        if (source_luminance <= options.threshold || proposal <= kEpsilon)
            continue;
        const double sample_solid_angle = inverse_samples/proposal;

        if (options.adaptive_peak_cells || options.candidate_boundary_peaks ||
                options.candidate_boundary_fallback) {
            PeakCandidate candidate;
            candidate.outgoing = outgoing;
            hemisphere_to_equal_solid_angle(
                outgoing, candidate.map_u, candidate.map_v);
            candidate.luminance = source_luminance;
            candidate.solid_angle = sample_solid_angle;
            candidates.push_back(candidate);
            continue;
        }

        const Vec3 outgoing_world = local_to_world(group, outgoing);
        const Vec3 source_direction = {{-outgoing_world[0],
                                        -outgoing_world[1],
                                        -outgoing_world[2]}};
        const double common = source_luminance*source_luminance*
                              sample_solid_angle;
        for (std::size_t view = 0; view < views.size(); ++view) {
            if (dot(source_direction, views[view].direction) <= kEpsilon)
                continue;
            double target_distance;
            if (!ray_intersects_group(views[view], source_direction, group,
                                      target_distance))
                continue;
            PendingContribution contribution;
            contribution.origin = views[view].origin;
            contribution.direction = source_direction;
            contribution.result_index = view*nsteps+time_index;
            contribution.target_distance = target_distance;
            if (peak_samples != NULL) {
                GlareDirectionSample peak_sample;
                peak_sample.direction = source_direction;
                peak_sample.result_index = contribution.result_index;
                peak_sample.partition = group_index;
                peak_sample.luminance = source_luminance;
                peak_sample.solid_angle = sample_solid_angle;
                contribution.peak_sample_index = peak_samples->size();
                contribution.value = 0.0;
                peak_samples->push_back(peak_sample);
            } else {
                const double position = guth_position_index(
                    source_direction, views[view].direction, options.up);
                contribution.value = common/(position*position);
            }
            pending.push_back(contribution);
            if (pending.size() >=
                    static_cast<std::size_t>(options.visibility_batch))
                flush_visibility(options, pending, contrast, peak_samples);
        }
    }

    if (!options.adaptive_peak_cells && !options.candidate_boundary_peaks &&
            !options.candidate_boundary_fallback)
        return PeakReconstructionNone;
    if (peak_samples == NULL)
        throw std::runtime_error(
            "peak reconstruction requires source clustering");
    if (candidates.empty())
        return PeakReconstructionNone;
    if (options.candidate_boundary_peaks) {
        const std::vector<BoundaryPeakSource> sources =
            reconstruct_candidate_boundary_sources(
                options, group, sun, incident, incident_cosine, tau,
                inverse_disk, candidates);
        record_boundary_statistics(sources, boundary_statistics);
        queue_boundary_peak_sources(
            options, group, group_index, time_index, views, nsteps,
            candidates, sources, contrast, pending, peak_samples);
        return PeakReconstructionBoundary;
    }
    bool adaptive_resolved = false;
    const std::vector<AdaptivePeakCell> cells =
        reconstruct_adaptive_peak_cells(
            options, group, sun, incident, incident_cosine, tau,
            inverse_disk, candidates, adaptive_resolved);
    if (!adaptive_resolved) {
        if (options.candidate_boundary_fallback) {
            const std::vector<BoundaryPeakSource> sources =
                reconstruct_candidate_boundary_sources(
                    options, group, sun, incident, incident_cosine, tau,
                    inverse_disk, candidates);
            record_boundary_statistics(sources, boundary_statistics);
            queue_boundary_peak_sources(
                options, group, group_index, time_index, views, nsteps,
                candidates, sources, contrast, pending, peak_samples);
            return PeakReconstructionBoundary;
        }
        for (std::size_t candidate_index = 0;
                candidate_index < candidates.size(); ++candidate_index) {
            const PeakCandidate &candidate = candidates[candidate_index];
            const Vec3 outgoing_world =
                local_to_world(group, candidate.outgoing);
            const Vec3 source_direction = {{-outgoing_world[0],
                                             -outgoing_world[1],
                                             -outgoing_world[2]}};
            for (std::size_t view = 0; view < views.size(); ++view) {
                if (dot(source_direction, views[view].direction) <= kEpsilon)
                    continue;
                double target_distance;
                if (!ray_intersects_group(views[view], source_direction,
                                          group, target_distance))
                    continue;
                PendingContribution contribution;
                contribution.origin = views[view].origin;
                contribution.direction = source_direction;
                contribution.result_index = view*nsteps+time_index;
                contribution.target_distance = target_distance;
                GlareDirectionSample peak_sample;
                peak_sample.direction = source_direction;
                peak_sample.result_index = contribution.result_index;
                peak_sample.partition = group_index;
                peak_sample.luminance = candidate.luminance;
                peak_sample.solid_angle = candidate.solid_angle;
                contribution.peak_sample_index = peak_samples->size();
                contribution.value = 0.0;
                peak_samples->push_back(peak_sample);
                pending.push_back(contribution);
                if (pending.size() >=
                        static_cast<std::size_t>(options.visibility_batch))
                    flush_visibility(options, pending, contrast, peak_samples);
            }
        }
        return PeakReconstructionFallback;
    }
    for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
        const AdaptivePeakCell &cell = cells[cell_index];
        const Vec3 outgoing_world = local_to_world(group, cell.outgoing);
        const Vec3 source_direction = {{-outgoing_world[0],
                                        -outgoing_world[1],
                                        -outgoing_world[2]}};
        const double solid_angle =
            2.0*PI*cell.map_size*cell.map_size;
        for (std::size_t view = 0; view < views.size(); ++view) {
            if (dot(source_direction, views[view].direction) <= kEpsilon)
                continue;
            double target_distance;
            if (!ray_intersects_group(views[view], source_direction, group,
                                      target_distance))
                continue;
            PendingContribution contribution;
            contribution.origin = views[view].origin;
            contribution.direction = source_direction;
            contribution.result_index = view*nsteps+time_index;
            contribution.target_distance = target_distance;
            GlareDirectionSample peak_sample;
            peak_sample.direction = source_direction;
            peak_sample.result_index = contribution.result_index;
            peak_sample.partition = group_index;
            peak_sample.luminance = cell.luminance;
            peak_sample.solid_angle = solid_angle;
            peak_sample.map_u0 = cell.map_u0;
            peak_sample.map_v0 = cell.map_v0;
            peak_sample.map_size = cell.map_size;
            peak_sample.adaptive_cell = true;
            contribution.peak_sample_index = peak_samples->size();
            contribution.value = 0.0;
            peak_samples->push_back(peak_sample);
            pending.push_back(contribution);
            if (pending.size() >=
                    static_cast<std::size_t>(options.visibility_batch))
                flush_visibility(options, pending, contrast, peak_samples);
        }
    }
    return PeakReconstructionAdaptive;
}

std::size_t peak_root(std::vector<std::size_t> &parent, std::size_t index)
{
    std::size_t root = index;
    while (parent[root] != root)
        root = parent[root];
    while (parent[index] != index) {
        const std::size_t next = parent[index];
        parent[index] = root;
        index = next;
    }
    return root;
}

void join_peaks(std::vector<std::size_t> &parent,
                std::vector<unsigned char> &rank,
                std::size_t first, std::size_t second)
{
    first = peak_root(parent, first);
    second = peak_root(parent, second);
    if (first == second)
        return;
    if (rank[first] < rank[second])
        std::swap(first, second);
    parent[second] = first;
    if (rank[first] == rank[second])
        ++rank[first];
}

double equivalent_cap_radius(double solid_angle)
{
    /* Approximate each Monte Carlo measure as a circular directional cell. */
    const double bounded = std::max(0.0, std::min(2.0*PI, solid_angle));
    return std::acos(std::max(-1.0, std::min(1.0,
        1.0-bounded/(2.0*PI))));
}

bool adaptive_cells_touch(const GlareDirectionSample &first,
                          const GlareDirectionSample &second)
{
    if (!first.adaptive_cell || !second.adaptive_cell ||
            first.partition != second.partition)
        return false;
    const double tolerance = 1.0e-10;
    const double first_u1 = first.map_u0+first.map_size;
    const double first_v1 = first.map_v0+first.map_size;
    const double second_u1 = second.map_u0+second.map_size;
    const double second_v1 = second.map_v0+second.map_size;
    const bool overlap_u = std::min(first_u1, second_u1)+tolerance >=
                           std::max(first.map_u0, second.map_u0);
    const bool overlap_v = std::min(first_v1, second_v1)+tolerance >=
                           std::max(first.map_v0, second.map_v0);
    return overlap_u && overlap_v;
}

std::size_t accumulate_peak_clusters(
        const Options &options,
        const std::vector<GlareDirectionSample> &peak_samples,
        const std::vector<Viewpoint> &views, std::size_t nsteps,
        std::vector<double> &contrast)
{
    std::vector<std::size_t> order;
    order.reserve(peak_samples.size());
    for (std::size_t i = 0; i < peak_samples.size(); ++i)
        if (peak_samples[i].visible)
            order.push_back(i);
    std::sort(order.begin(), order.end(),
        [&peak_samples](std::size_t first, std::size_t second) {
            return peak_samples[first].result_index <
                   peak_samples[second].result_index;
        });

    const double max_link = options.peak_max_link_angle*PI/180.0;
    std::size_t source_count = 0;
    for (std::size_t begin = 0; begin < order.size();) {
        std::size_t end = begin+1;
        const std::size_t result_index =
            peak_samples[order[begin]].result_index;
        while (end < order.size() &&
                peak_samples[order[end]].result_index == result_index)
            ++end;
        const std::size_t count = end-begin;
        std::vector<std::size_t> parent(count);
        std::vector<unsigned char> rank(count, 0);
        std::vector<double> radius(count, 0.0);
        for (std::size_t i = 0; i < count; ++i) {
            parent[i] = i;
            radius[i] = equivalent_cap_radius(
                peak_samples[order[begin+i]].solid_angle);
        }
        for (std::size_t i = 0; i < count; ++i)
            for (std::size_t j = i+1; j < count; ++j) {
                const GlareDirectionSample &first =
                    peak_samples[order[begin+i]];
                const GlareDirectionSample &second =
                    peak_samples[order[begin+j]];
                if (first.coherent_source || second.coherent_source) {
                    if (first.coherent_source && second.coherent_source &&
                            first.partition == second.partition &&
                            first.source_id == second.source_id)
                        join_peaks(parent, rank, i, j);
                    continue;
                }
                if (first.adaptive_cell || second.adaptive_cell) {
                    if (adaptive_cells_touch(first, second))
                        join_peaks(parent, rank, i, j);
                    continue;
                }
                const double link = std::min(max_link,
                    options.peak_cluster_factor*(radius[i]+radius[j]));
                if (link > 0.0 &&
                        dot(first.direction, second.direction) >=
                            std::cos(link))
                    join_peaks(parent, rank, i, j);
            }

        std::vector<double> omega(count, 0.0);
        std::vector<double> luminance_integral(count, 0.0);
        std::vector<Vec3> center(count, Vec3{{0.0, 0.0, 0.0}});
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t root = peak_root(parent, i);
            const GlareDirectionSample &sample =
                peak_samples[order[begin+i]];
            omega[root] += sample.solid_angle;
            const double luminous_weight =
                sample.luminance*sample.solid_angle;
            luminance_integral[root] += luminous_weight;
            /* evalglare locates a source using luminance-solid-angle weight. */
            for (int c = 0; c < 3; ++c)
                center[root][c] += luminous_weight*sample.direction[c];
        }
        const std::size_t view_index = result_index/nsteps;
        if (view_index >= views.size())
            throw std::runtime_error("invalid clustered result index");
        for (std::size_t root = 0; root < count; ++root) {
            if (omega[root] <= kEpsilon || parent[root] != root)
                continue;
            const double average_luminance =
                luminance_integral[root]/omega[root];
            const Vec3 source_direction = normalized(
                center[root], "peak luminance-weighted center");
            const double position = guth_position_index(
                source_direction, views[view_index].direction, options.up);
            contrast[result_index] += average_luminance*average_luminance*
                                      omega[root]/(position*position);
            ++source_count;
        }
        begin = end;
    }
    return source_count;
}

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

void usage(FILE *stream)
{
    std::fprintf(stream,
        "Usage: ttsuncontrast -vf views.pts -S suns.rad\n"
        "       [--auto-bsdf scene.oct |\n"
        "        {-wgroup window.rad (-wBSDF system.xml |\n"
        "         -wABsdf materials.rad id)}...] [options]\n\n"
        "Options:\n"
        "  -? | --help              show this help\n"
        "  -h                       suppress Radiance matrix header\n"
        "  --auto-bsdf scene.oct    discover BSDF/aBSDF polygons and frames\n"
        "  -wgroup file.rad         finite window polygons for the next group\n"
        "  -wBSDF file.xml          TensorTree or matrix BSDF for that group\n"
        "  -wABsdf file.rad id      aBSDF material; excludes its through peak\n"
        "  --include-aBSDF-through include aBSDF through peaks in the output\n"
        "  --bsdf-samples n         outgoing importance samples per sun/group (256)\n"
        "  --adaptive-peak-cells    reconstruct resolvable peaks with equal-solid-angle\n"
        "                           cells; fall back for subcell peaks (default)\n"
        "  --peak-cluster-factor d adjacency scale for equivalent sample radii (2)\n"
        "  --peak-max-link-angle d maximum neighbor link angle in degrees (10)\n"
        "  --adaptive-base-level n initial Shirley-Chiu grid level (4)\n"
        "  --adaptive-max-level n  maximum Shirley-Chiu refinement level (7)\n"
        "  --adaptive-contrast-tolerance d\n"
        "                           local contrast convergence tolerance (0.05)\n"
        "  --adaptive-gradient-tolerance d\n"
        "                           normalized luminance-gradient tolerance (0.25)\n"
        "  --sun-disk-samples n     perfect-square incident disk samples (1)\n"
        "  --sample-seed n          deterministic BSDF sampling seed (0)\n"
        "  -i scene.oct             reject rays blocked before the window plane\n"
        "  --rtrace executable      visibility-query executable (rtrace)\n"
        "  -n processes             rtrace worker count (1)\n"
        "  --visibility-batch n     rays per rtrace call (65536)\n"
        "  --visibility-tolerance d hit-distance tolerance in scene units (0.001)\n"
        "  -b luminance             glare-source threshold in cd/m2 (2000)\n"
        "  -vu x y z                global up vector used by windows and Guth index\n"
        "  -o file                  output matrix (default stdout)\n"
        "  -q                       suppress progress messages\n\n"
        "Only non-diffuse direct-sun transmission is integrated. aBSDF groups\n"
        "default to redirected transmission so an external visible-sun term\n"
        "may supply the extracted through peak without double-counting. Rows\n"
        "are viewpoints and columns follow source records in suns.rad.\n");
}

std::string option_value(int &index, int argc, char *argv[],
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
        else if (argument == "-q" || argument == "--quiet")
            options.quiet = true;
        else if (argument == "-vf" || argument == "--views")
            options.views_path = option_value(index, argc, argv, argument);
        else if (argument == "-S" || argument == "--suns")
            options.suns_path = option_value(index, argc, argv, argument);
        else if (argument == "--auto-bsdf")
            options.auto_bsdf_octree =
                option_value(index, argc, argv, argument);
        else if (argument == "-wgroup" || argument == "--window") {
            WindowGroup group;
            group.geometry_path = option_value(index, argc, argv, argument);
            group.bsdf = NULL;
            options.groups.push_back(group);
        } else if (argument == "-wBSDF" || argument == "--bsdf") {
            if (options.groups.empty() || !options.groups.back().bsdf_path.empty() ||
                    !options.groups.back().absdf_path.empty())
                throw std::runtime_error(argument + " must follow its -wgroup");
            options.groups.back().bsdf_path =
                option_value(index, argc, argv, argument);
        } else if (argument == "-wABsdf" || argument == "--absdf") {
            if (options.groups.empty() || !options.groups.back().bsdf_path.empty() ||
                    !options.groups.back().absdf_path.empty())
                throw std::runtime_error(argument + " must follow its -wgroup");
            options.groups.back().absdf_path =
                option_value(index, argc, argv, argument);
            options.groups.back().absdf_identifier =
                option_value(index, argc, argv, argument);
        } else if (argument == "--include-aBSDF-through")
            options.include_absdf_through = true;
        else if (argument == "--adaptive-peak-cells") {
            options.cluster_peaks = true;
            options.adaptive_peak_cells = true;
            options.candidate_boundary_peaks = false;
            options.candidate_boundary_fallback = false;
        } else if (argument == "--bsdf-samples")
            options.samples = parse_nonnegative_int(
                option_value(index, argc, argv, argument), "BSDF sample count");
        else if (argument == "--peak-cluster-factor")
            options.peak_cluster_factor = parse_nonnegative_double(
                option_value(index, argc, argv, argument),
                "peak cluster factor");
        else if (argument == "--peak-max-link-angle")
            options.peak_max_link_angle = parse_nonnegative_double(
                option_value(index, argc, argv, argument),
                "peak maximum link angle");
        else if (argument == "--adaptive-base-level")
            options.adaptive_base_level = parse_nonnegative_int(
                option_value(index, argc, argv, argument),
                "adaptive base level");
        else if (argument == "--adaptive-max-level")
            options.adaptive_max_level = parse_nonnegative_int(
                option_value(index, argc, argv, argument),
                "adaptive maximum level");
        else if (argument == "--adaptive-contrast-tolerance")
            options.adaptive_contrast_tolerance = parse_nonnegative_double(
                option_value(index, argc, argv, argument),
                "adaptive contrast tolerance");
        else if (argument == "--adaptive-gradient-tolerance")
            options.adaptive_gradient_tolerance = parse_nonnegative_double(
                option_value(index, argc, argv, argument),
                "adaptive gradient tolerance");
        else if (argument == "--sun-disk-samples")
            options.sun_disk_samples = parse_nonnegative_int(
                option_value(index, argc, argv, argument),
                "sun-disk sample count");
        else if (argument == "--sample-seed")
            options.seed = parse_nonnegative_int(
                option_value(index, argc, argv, argument), "sample seed");
        else if (argument == "-i" || argument == "--visibility-octree")
            options.visibility_octree =
                option_value(index, argc, argv, argument);
        else if (argument == "--rtrace")
            options.rtrace = option_value(index, argc, argv, argument);
        else if (argument == "-n" || argument == "--processes")
            options.nproc = parse_nonnegative_int(
                option_value(index, argc, argv, argument), "process count");
        else if (argument == "--visibility-batch")
            options.visibility_batch = parse_nonnegative_int(
                option_value(index, argc, argv, argument),
                "visibility batch size");
        else if (argument == "--visibility-tolerance")
            options.visibility_tolerance = parse_nonnegative_double(
                option_value(index, argc, argv, argument),
                "visibility tolerance");
        else if (argument == "-b" || argument == "--threshold")
            options.threshold = parse_nonnegative_double(
                option_value(index, argc, argv, argument),
                "luminance threshold");
        else if (argument == "-vu") {
            for (int c = 0; c < 3; ++c)
                options.up[c] = parse_finite_double(
                    option_value(index, argc, argv, argument), "up vector");
        } else if (argument == "-o" || argument == "--output")
            options.output_path = option_value(index, argc, argv, argument);
        else
            throw std::runtime_error("unknown option '" + argument + "'");
    }
    return options;
}

void validate_options(Options &options, const std::vector<Viewpoint> &views)
{
    if (options.views_path.empty() || options.suns_path.empty())
        throw std::runtime_error("-vf and -S are required");
    if (!options.auto_bsdf_octree.empty() && !options.groups.empty())
        throw std::runtime_error(
            "--auto-bsdf cannot be combined with explicit -wgroup options");
    if (options.auto_bsdf_octree.empty() && options.groups.empty())
        throw std::runtime_error(
            "--auto-bsdf or at least one explicit window group is required");
    if (options.samples < 1 || options.sun_disk_samples < 1)
        throw std::runtime_error("sample counts must be positive");
    if (options.nproc < 1 || options.visibility_batch < 1)
        throw std::runtime_error(
            "process count and visibility batch size must be positive");
    const int disk_side = static_cast<int>(
        std::sqrt(static_cast<double>(options.sun_disk_samples)));
    if (disk_side*disk_side != options.sun_disk_samples)
        throw std::runtime_error("sun-disk sample count must be a perfect square");
    if (options.samples < options.sun_disk_samples)
        throw std::runtime_error("BSDF samples must not be fewer than sun-disk samples");
    if (options.peak_cluster_factor <= 0.0)
        throw std::runtime_error("peak cluster factor must be positive");
    if (options.peak_max_link_angle <= 0.0 ||
            options.peak_max_link_angle > 180.0)
        throw std::runtime_error(
            "peak maximum link angle must be in (0, 180] degrees");
    if (options.adaptive_base_level < 1 ||
            options.adaptive_base_level > 12 ||
            options.adaptive_max_level < options.adaptive_base_level ||
            options.adaptive_max_level > 14)
        throw std::runtime_error(
            "adaptive levels must satisfy 1 <= base <= maximum <= 14");
    if (options.adaptive_contrast_tolerance <= 0.0 ||
            options.adaptive_gradient_tolerance <= 0.0)
        throw std::runtime_error(
            "adaptive tolerances must be positive");
    if (options.boundary_rays < 8 || options.boundary_rays > 256)
        throw std::runtime_error(
            "boundary ray count must be in [8, 256]");
    if (options.boundary_bisections < 1 ||
            options.boundary_bisections > 30)
        throw std::runtime_error(
            "boundary bisection count must be in [1, 30]");
    if (options.boundary_max_solid_angle > 2.0*PI)
        throw std::runtime_error(
            "boundary maximum solid angle must not exceed 2*pi sr");
    options.up = normalized(options.up, "up vector");
    require_file(options.views_path, "viewpoint file");
    require_file(options.suns_path, "suns file");
    if (!options.auto_bsdf_octree.empty()) {
        require_file(options.auto_bsdf_octree, "automatic BSDF octree");
        if (options.visibility_octree.empty())
            options.visibility_octree = options.auto_bsdf_octree;
        discover_bsdf_groups(options, views);
    }
    if (!options.visibility_octree.empty())
        require_file(options.visibility_octree, "visibility octree");
    for (std::size_t i = 0; i < options.groups.size(); ++i) {
        WindowGroup &group = options.groups[i];
        if (group.bsdf_path.empty() && group.absdf_path.empty())
            if (!group.auto_discovered)
                throw std::runtime_error(
                    "every -wgroup requires a following -wBSDF or -wABsdf");
        if (!group.auto_discovered)
            require_file(group.geometry_path, "window geometry file");
        if (group.auto_discovered) {
            /* Material path, type, and local up vector came from the octree. */
        } else if (!group.absdf_path.empty()) {
            require_file(group.absdf_path, "aBSDF material file");
            load_absdf_material(group);
        } else {
            group.up = options.up;
        }
        require_file(group.bsdf_path, "BSDF file");
        orient_and_frame_group(group, views, group.up);
        group.bsdf = SDcacheFile(group.bsdf_path.c_str());
        if (group.bsdf == NULL)
            throw std::runtime_error("cannot load BSDF '" + group.bsdf_path + "'");
        if (group.auto_discovered && !options.quiet)
            std::fprintf(stderr,
                "%s: auto group %lu: %s '%s', %lu polygons, "
                "normal=(%.6g %.6g %.6g), up=(%.6g %.6g %.6g)\n",
                progname, static_cast<unsigned long>(i+1),
                group.is_absdf ? "aBSDF" : "BSDF",
                group.material_name.c_str(),
                static_cast<unsigned long>(group.polygons.size()),
                group.normal[0], group.normal[1], group.normal[2],
                group.up[0], group.up[1], group.up[2]);
    }
}

} // namespace

int main(int argc, char *argv[])
{
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN);
#endif
    progname = fixargv0(argv[0]);
    try {
        Options options = parse_options(argc, argv);
        if (options.views_path.empty())
            throw std::runtime_error("-vf is required");
        const std::vector<Viewpoint> views = load_views(options.views_path);
        validate_options(options, views);
        const std::vector<Sun> suns = load_suns(options.suns_path);
        std::vector<double> contrast(views.size()*suns.size(), 0.0);
        std::vector<PendingContribution> pending;
        pending.reserve(static_cast<std::size_t>(options.visibility_batch));
        std::vector<GlareDirectionSample> peak_samples;
        if (options.cluster_peaks)
            peak_samples.reserve(
                static_cast<std::size_t>(options.visibility_batch));
        std::size_t active = 0;
        for (std::size_t time = 0; time < suns.size(); ++time)
            if (suns[time].active)
                ++active;
        if (!options.quiet)
            std::fprintf(stderr,
                "%s: %lu views, %lu window groups, %lu time steps (%lu active), "
                "%d BSDF samples, %d sun-disk samples\n",
                progname, static_cast<unsigned long>(views.size()),
                static_cast<unsigned long>(options.groups.size()),
                static_cast<unsigned long>(suns.size()),
                 static_cast<unsigned long>(active), options.samples,
                 options.sun_disk_samples);
        if (options.candidate_boundary_fallback && !options.quiet)
            std::fprintf(stderr,
                "%s: adaptive cells with candidate-boundary fallback "
                "(levels %d-%d, %d rays, %d bisections, "
                "maximum solid angle %.6g sr)\n",
                progname, options.adaptive_base_level,
                options.adaptive_max_level, options.boundary_rays,
                options.boundary_bisections,
                options.boundary_max_solid_angle);
        else if (options.candidate_boundary_peaks && !options.quiet)
            std::fprintf(stderr,
                "%s: candidate-centered peak boundaries enabled "
                "(%d rays, %d bisections, maximum solid angle %.6g sr)\n",
                progname, options.boundary_rays,
                options.boundary_bisections,
                options.boundary_max_solid_angle);
        else if (options.adaptive_peak_cells && !options.quiet)
            std::fprintf(stderr,
                "%s: adaptive peak cells enabled (levels %d-%d, "
                "contrast tolerance %.6g, gradient tolerance %.6g)\n",
                progname, options.adaptive_base_level,
                options.adaptive_max_level,
                options.adaptive_contrast_tolerance,
                options.adaptive_gradient_tolerance);
        else if (options.cluster_peaks && !options.quiet)
            std::fprintf(stderr,
                "%s: importance-sample clustering enabled "
                "(factor %.6g, max link %.6g deg)\n",
                progname, options.peak_cluster_factor,
                options.peak_max_link_angle);
        std::size_t completed = 0;
        std::size_t clustered_sources = 0;
        std::size_t adaptive_reconstructions = 0;
        std::size_t importance_fallbacks = 0;
        std::size_t boundary_reconstructions = 0;
        BoundaryStatistics boundary_statistics;
        for (std::size_t time = 0; time < suns.size(); ++time) {
            if (!suns[time].active)
                continue;
            for (std::size_t group = 0; group < options.groups.size(); ++group) {
                const PeakReconstructionResult reconstruction =
                    integrate_group_sun(
                        options, options.groups[group], group, suns[time],
                        time, views, contrast, suns.size(), pending,
                        options.cluster_peaks ? &peak_samples : NULL,
                        &boundary_statistics);
                if (reconstruction == PeakReconstructionAdaptive)
                    ++adaptive_reconstructions;
                else if (reconstruction == PeakReconstructionFallback)
                    ++importance_fallbacks;
                else if (reconstruction == PeakReconstructionBoundary)
                    ++boundary_reconstructions;
            }
            if (options.cluster_peaks &&
                    peak_samples.size() >=
                        static_cast<std::size_t>(options.visibility_batch)) {
                flush_visibility(options, pending, contrast, &peak_samples);
                clustered_sources += accumulate_peak_clusters(
                    options, peak_samples, views, suns.size(), contrast);
                peak_samples.clear();
            }
            ++completed;
            if (!options.quiet && (completed%250 == 0 || completed == active))
                std::fprintf(stderr, "%s: processed %lu/%lu active suns\n",
                    progname, static_cast<unsigned long>(completed),
                    static_cast<unsigned long>(active));
        }
        flush_visibility(options, pending, contrast,
                         options.cluster_peaks ? &peak_samples : NULL);
        if (options.cluster_peaks) {
            clustered_sources += accumulate_peak_clusters(
                options, peak_samples, views, suns.size(), contrast);
            if (!options.quiet)
                std::fprintf(stderr, "%s: accumulated %lu clustered sources\n",
                    progname, static_cast<unsigned long>(clustered_sources));
        }
        if (options.adaptive_peak_cells && !options.quiet)
            std::fprintf(stderr,
                "%s: %lu adaptive reconstructions, %lu importance-sample "
                "fallbacks\n",
                progname,
                static_cast<unsigned long>(adaptive_reconstructions),
                static_cast<unsigned long>(importance_fallbacks));
        if ((options.candidate_boundary_peaks ||
                options.candidate_boundary_fallback) && !options.quiet)
            std::fprintf(stderr, "%s: %lu candidate-boundary "
                "group/time reconstructions\n", progname,
                static_cast<unsigned long>(boundary_reconstructions));
        if (boundary_statistics.resolved_sources > 0 && !options.quiet)
            std::fprintf(stderr,
                "%s: boundary/importance solid-angle scale: mean %.6g, "
                "range %.6g-%.6g; bins "
                "<0.5/0.5-0.75/0.75-1/1-1.25/1.25-2/>=2 = "
                "%lu/%lu/%lu/%lu/%lu/%lu\n",
                progname,
                boundary_statistics.scale_sum/
                    boundary_statistics.resolved_sources,
                boundary_statistics.scale_min,
                boundary_statistics.scale_max,
                static_cast<unsigned long>(
                    boundary_statistics.scale_bins[0]),
                static_cast<unsigned long>(
                    boundary_statistics.scale_bins[1]),
                static_cast<unsigned long>(
                    boundary_statistics.scale_bins[2]),
                static_cast<unsigned long>(
                    boundary_statistics.scale_bins[3]),
                static_cast<unsigned long>(
                    boundary_statistics.scale_bins[4]),
                static_cast<unsigned long>(
                    boundary_statistics.scale_bins[5]));
        if (boundary_statistics.resolved_sources > 0 && !options.quiet)
            std::fprintf(stderr,
                "%s: boundary solid-angle bins "
                "<1e-5/1e-5-1e-4/1e-4-1e-3/1e-3-1e-2/1e-2-1e-1/>=1e-1 = "
                "%lu/%lu/%lu/%lu/%lu/%lu sr\n",
                progname,
                static_cast<unsigned long>(
                    boundary_statistics.solid_angle_bins[0]),
                static_cast<unsigned long>(
                    boundary_statistics.solid_angle_bins[1]),
                static_cast<unsigned long>(
                    boundary_statistics.solid_angle_bins[2]),
                static_cast<unsigned long>(
                    boundary_statistics.solid_angle_bins[3]),
                static_cast<unsigned long>(
                    boundary_statistics.solid_angle_bins[4]),
                static_cast<unsigned long>(
                    boundary_statistics.solid_angle_bins[5]));
        write_matrix(options, contrast, views.size(), suns.size(), argc, argv);
        SDfreeCache(NULL);
        if (nobjects > 0)
            freeobjects(0, nobjects);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s: %s\n",
                     progname ? progname : "ttsuncontrast", error.what());
        SDfreeCache(NULL);
        if (nobjects > 0)
            freeobjects(0, nobjects);
        return 1;
    }
    return 0;
}
