#ifndef lint
static const char RCSid[] = "$Id$";
#endif

/*
 * Compute DGP contrast from direct visible sun and direct-sun-driven,
 * non-diffuse transmission/refraction lobes without an HDR image.  The
 * solar disk and a Shirley-Chiu equal-solid-angle view hemisphere are
 * refined independently for every active solar record.  Mutually exclusive
 * Radiance path filters keep the two explicit components disjoint.
 */

#include "direct_lobe_backend.h"
#include "direct_lobe_proposals.h"
#include "paths.h"
#include "platform.h"
#include "rtio.h"
#include "standard.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

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
    double solid_angle;
    bool active;
};

struct Primitive {
    std::string modifier;
    std::string type;
    std::string identifier;
    std::vector<double> real_args;
};

struct Cell {
    int x0;
    int y0;
    int span;
    Vec3 direction;
    double luminance;
    double solid_angle_override;
    double parent_luminance = 0.0;
    Vec3 parent_direction = {{0.0, 0.0, 1.0}};
    double parent_solid_angle = 0.0;
    bool has_parent_estimate = false;
};

struct TimeTree {
    std::vector<Cell> cells;
    std::vector<DirectLobeProposal> proposals;
    std::vector<double> proposal_luminance;
};

struct SunDiskTree {
    std::vector<Cell> cells;
};

struct PendingCell {
    std::size_t time;
    std::size_t cell;
};

struct PendingProposal {
    std::size_t time;
    std::size_t proposal;
};

struct BatchedPendingCell {
    std::size_t view;
    std::size_t time;
    std::size_t cell;
};

struct BatchedPendingProposal {
    std::size_t view;
    std::size_t time;
    std::size_t proposal;
};

struct ViewLobeState {
    Viewpoint view;
    std::size_t view_index = 0;
    Vec3 right = {{1.0, 0.0, 0.0}};
    Vec3 up = {{0.0, 0.0, 1.0}};
    std::vector<TimeTree> trees;
    std::vector<double> contrast;
    int stable_passes = 0;
    bool finished = false;
};

enum SunMode {
    SUN_BATCH,
    SUN_ADAPTIVE
};

struct Options {
    bool no_header = false;
    bool quiet = false;
    std::string views_path;
    std::string suns_path;
    std::string output_path;
    std::string sun_output_path;
    std::string sun_irradiance_output_path;
    std::string sun_total_irradiance_output_path;
    std::string lobe_output_path;
    std::string lobe_view_mask_path;
    std::string octree;
    bool include_sun = true;
    bool include_lobes = true;
    SunMode sun_mode = SUN_BATCH;
    int nproc = 1;
    int resolution = 4;
    int adaptive_levels = 6;
    int adaptive_guard = 1;
    int accumulation = 8;
    int min_cluster_cells = 1;
    int max_lobe_bounces = 2;
    int lobe_samples = 16;
    int min_adaptive_levels = 2;
    int convergence_passes = 2;
    int sun_disk_resolution = 2;
    int sun_disk_levels = 6;
    int sun_disk_min_levels = 3;
    int sun_disk_guard = 1;
    int lobe_sun_disk_resolution = 8;
    int lobe_sun_disk_pilot_resolution = 0;
    int sun_samples = 10000;
    int view_batch_size = 1;
    double threshold = 2000.0;
    double adaptive_trigger = 500.0;
    double adaptive_gradient = 500.0;
    double adaptive_variance = 500.0;
    double adaptive_uncertainty = 0.01;
    double contrast_tolerance = 0.01;
    double sun_disk_gradient = 500.0;
    double sun_disk_tolerance = 0.005;
    double proposal_normal_tolerance = 0.1;
    double lobe_sun_disk_pilot_guard_angle = 1.0;
    Vec3 up = {{0.0, 0.0, 1.0}};
    std::vector<std::string> render_options = {
        "-lw", "1e-7", "-st", "0", "-dj", "1",
        "-dt", "0", "-dc", "1", "-ds", "0.02"
    };
};

double dot(const Vec3 &a, const Vec3 &b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

Vec3 cross(const Vec3 &a, const Vec3 &b)
{
    return Vec3{{a[1]*b[2]-a[2]*b[1],
                 a[2]*b[0]-a[0]*b[2],
                 a[0]*b[1]-a[1]*b[0]}};
}

Vec3 add_scaled(const Vec3 &a, const Vec3 &b, double scale)
{
    return Vec3{{a[0]+scale*b[0], a[1]+scale*b[1],
                 a[2]+scale*b[2]}};
}

Vec3 scaled(const Vec3 &a, double scale)
{
    return Vec3{{scale*a[0], scale*a[1], scale*a[2]}};
}

double norm(const Vec3 &a)
{
    return std::sqrt(dot(a, a));
}

Vec3 normalized(const Vec3 &a, const std::string &what)
{
    const double length = norm(a);
    if (!(length > kEpsilon))
        throw std::runtime_error("zero-length " + what);
    return scaled(a, 1.0/length);
}

double clamp_unit(double value)
{
    return std::max(-1.0, std::min(1.0, value));
}

double angle(const Vec3 &a, const Vec3 &b)
{
    return std::acos(clamp_unit(dot(a, b)));
}

std::string trim_comment(const std::string &line)
{
    const std::string text = line.substr(0, line.find('#'));
    const std::string::size_type first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return std::string();
    return text.substr(first, text.find_last_not_of(" \t\r\n")-first+1);
}

bool parse_double(const std::string &word, double &value)
{
    char *end = NULL;
    errno = 0;
    value = std::strtod(word.c_str(), &end);
    return !errno && end != word.c_str() && !*end && std::isfinite(value);
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

double parse_nonnegative(const std::string &word, const std::string &what)
{
    double value;
    if (!parse_double(word, value) || value < 0.0)
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

std::vector<unsigned char> load_lobe_view_mask(
    const std::string &path, std::size_t expected_views)
{
    std::vector<unsigned char> mask(expected_views, 1u);
    if (path.empty())
        return mask;
    std::ifstream input(path.c_str());
    if (!input)
        throw std::runtime_error("cannot open lobe view mask '"+path+"'");
    std::vector<unsigned char> values;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = trim_comment(line);
        if (line.empty())
            continue;
        std::istringstream parser(line);
        int value;
        while (parser >> value) {
            if (value != 0 && value != 1)
                throw std::runtime_error(
                    "lobe view mask entries must be 0 or 1 at line "+
                    std::to_string(line_number));
            values.push_back(value ? 1u : 0u);
        }
        if (!parser.eof())
            throw std::runtime_error(
                "invalid lobe view mask entry at line "+
                std::to_string(line_number));
    }
    if (values.size() != expected_views)
        throw std::runtime_error(
            "lobe view mask has "+std::to_string(values.size())+
            " entries; expected "+std::to_string(expected_views));
    return values;
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

std::vector<std::string> take_counted(
    const std::vector<std::string> &tokens, std::size_t &index,
    const std::string &what)
{
    if (index >= tokens.size())
        throw std::runtime_error("unexpected end before " + what + " count");
    const int count = parse_int(tokens[index++], what + " count");
    if (index+static_cast<std::size_t>(count) > tokens.size())
        throw std::runtime_error("invalid " + what + " argument list");
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
        if (index+3 > tokens.size())
            throw std::runtime_error("incomplete primitive in '"+path+"'");
        Primitive primitive;
        primitive.modifier = tokens[index++];
        primitive.type = tokens[index++];
        primitive.identifier = tokens[index++];
        take_counted(tokens, index, "string");
        take_counted(tokens, index, "integer");
        const std::vector<std::string> real =
            take_counted(tokens, index, "real");
        for (std::size_t i = 0; i < real.size(); ++i) {
            double value;
            if (!parse_double(real[i], value))
                throw std::runtime_error("invalid real argument in '"+path+"'");
            primitive.real_args.push_back(value);
        }
        primitives.push_back(primitive);
    }
    return primitives;
}

std::vector<Sun> load_suns(const std::string &path)
{
    const std::vector<Primitive> primitives = load_primitives(path);
    std::vector<Sun> suns;
    std::set<std::string> seen;
    std::vector<std::pair<std::string, Vec3> > lights;
    for (std::size_t i = 0; i < primitives.size(); ++i)
        if (primitives[i].type == "light" &&
                primitives[i].real_args.size() >= 3)
            lights.push_back(std::make_pair(primitives[i].identifier,
                Vec3{{primitives[i].real_args[0], primitives[i].real_args[1],
                      primitives[i].real_args[2]}}));

    for (std::size_t i = 0; i < primitives.size(); ++i) {
        const Primitive &primitive = primitives[i];
        if (primitive.type != "source")
            continue;
        if (primitive.real_args.size() < 4)
            throw std::runtime_error("solar source has fewer than four values");
        if (!seen.insert(primitive.modifier).second)
            throw std::runtime_error("solar source modifiers must be unique");
        Sun sun;
        sun.modifier = primitive.modifier;
        sun.direction = normalized(Vec3{{primitive.real_args[0],
            primitive.real_args[1], primitive.real_args[2]}},
            "solar source direction");
        sun.angular_diameter = primitive.real_args[3]*PI/180.0;
        sun.solid_angle = 2.0*PI*(1.0-
            std::cos(0.5*sun.angular_diameter));
        sun.radiance = Vec3{{1.0, 1.0, 1.0}};
        for (std::size_t light = 0; light < lights.size(); ++light)
            if (lights[light].first == sun.modifier) {
                sun.radiance = lights[light].second;
                break;
            }
        sun.active = std::max(sun.radiance[0],
            std::max(sun.radiance[1], sun.radiance[2])) > 0.0;
        suns.push_back(sun);
    }
    if (suns.empty())
        throw std::runtime_error("no solar sources found in '"+path+"'");
    return suns;
}

void view_basis(const Vec3 &forward, const Vec3 &preferred_up,
                Vec3 &right, Vec3 &up)
{
    up = add_scaled(preferred_up, forward, -dot(preferred_up, forward));
    if (norm(up) < 1.0e-7) {
        const Vec3 fallback = {{0.0, 1.0, 0.0}};
        up = add_scaled(fallback, forward, -dot(fallback, forward));
    }
    up = normalized(up, "view-up vector");
    right = normalized(cross(forward, up), "view-right vector");
    up = normalized(cross(right, forward), "orthogonal view-up vector");
}

void concentric_disk(double a, double b, double &x, double &y)
{
    if (std::fabs(a) < kEpsilon && std::fabs(b) < kEpsilon) {
        x = y = 0.0;
        return;
    }
    double radius, phi;
    if (std::fabs(a) > std::fabs(b)) {
        radius = a;
        phi = (PI/4.0)*(b/a);
    } else {
        radius = b;
        phi = PI/2.0-(PI/4.0)*(a/b);
    }
    x = radius*std::cos(phi);
    y = radius*std::sin(phi);
}

Vec3 sample_direction(double u, double v, const Vec3 &forward,
                      const Vec3 &right, const Vec3 &up)
{
    double disk_x, disk_y;
    concentric_disk(2.0*u-1.0, 2.0*v-1.0, disk_x, disk_y);
    const double radius = std::sqrt(disk_x*disk_x+disk_y*disk_y);
    const double z = std::max(0.0, 1.0-radius*radius);
    double x = 0.0, y = 0.0;
    if (radius > kEpsilon) {
        const double transverse = std::sqrt(std::max(0.0, 1.0-z*z));
        x = transverse*disk_x/radius;
        y = transverse*disk_y/radius;
    }
    Vec3 result = scaled(forward, z);
    result = add_scaled(result, right, x);
    result = add_scaled(result, up, y);
    return normalized(result, "hemisphere sample direction");
}

void solar_basis(const Vec3 &forward, Vec3 &right, Vec3 &up)
{
    Vec3 reference = {{0.0, 0.0, 1.0}};
    if (std::fabs(dot(reference, forward)) > 0.95)
        reference = Vec3{{0.0, 1.0, 0.0}};
    right = normalized(cross(reference, forward), "solar-disk right vector");
    up = normalized(cross(forward, right), "solar-disk up vector");
}

Vec3 sample_solar_disk(double u, double v, const Sun &sun)
{
    double disk_x, disk_y;
    concentric_disk(2.0*u-1.0, 2.0*v-1.0, disk_x, disk_y);
    const double radius_squared = std::min(
        1.0, disk_x*disk_x+disk_y*disk_y);
    const double radius = std::sqrt(radius_squared);
    const double half_angle = 0.5*sun.angular_diameter;
    const double cosine = 1.0-radius_squared*(1.0-std::cos(half_angle));
    const double sine = std::sqrt(std::max(0.0, 1.0-cosine*cosine));
    Vec3 right, up;
    solar_basis(sun.direction, right, up);
    Vec3 result = scaled(sun.direction, cosine);
    if (radius > kEpsilon) {
        result = add_scaled(result, right, sine*disk_x/radius);
        result = add_scaled(result, up, sine*disk_y/radius);
    }
    return normalized(result, "solar-disk sample direction");
}

std::vector<std::vector<int> > build_adjacency(const std::vector<Cell> &cells)
{
    struct Edge {
        int begin;
        int end;
        int cell;
    };
    typedef std::map<int, std::vector<Edge> > EdgeMap;
    EdgeMap left, right, bottom, top;
    std::vector<std::vector<int> > adjacency(cells.size());
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const Cell &cell = cells[i];
        const Edge vertical = {cell.y0, cell.y0+cell.span,
                               static_cast<int>(i)};
        const Edge horizontal = {cell.x0, cell.x0+cell.span,
                                 static_cast<int>(i)};
        left[cell.x0].push_back(vertical);
        right[cell.x0+cell.span].push_back(vertical);
        bottom[cell.y0].push_back(horizontal);
        top[cell.y0+cell.span].push_back(horizontal);
    }
    const auto connect_edges = [&adjacency](EdgeMap &first, EdgeMap &second) {
        for (EdgeMap::iterator line = first.begin(); line != first.end();
                ++line) {
            EdgeMap::iterator opposite = second.find(line->first);
            if (opposite == second.end())
                continue;
            const auto order = [](const Edge &a, const Edge &b) {
                return a.begin < b.begin ||
                    (a.begin == b.begin && a.end < b.end);
            };
            std::sort(line->second.begin(), line->second.end(), order);
            std::sort(opposite->second.begin(), opposite->second.end(), order);
            std::size_t i = 0, j = 0;
            while (i < line->second.size() && j < opposite->second.size()) {
                const Edge &a = line->second[i];
                const Edge &b = opposite->second[j];
                if (std::max(a.begin, b.begin) < std::min(a.end, b.end) &&
                        a.cell != b.cell) {
                    adjacency[a.cell].push_back(b.cell);
                    adjacency[b.cell].push_back(a.cell);
                }
                if (a.end <= b.end)
                    ++i;
                else
                    ++j;
            }
        }
    };
    connect_edges(right, left);
    connect_edges(top, bottom);
    return adjacency;
}

double guth_position_index(const Vec3 &source_direction,
                           const Vec3 &forward, const Vec3 &preferred_up)
{
    Vec3 right, up;
    view_basis(forward, preferred_up, right, up);
    const Vec3 source = normalized(source_direction, "glare-source direction");
    double sigma = std::acos(clamp_unit(dot(source, forward)));
    if (sigma < 1.0e-9)
        return 1.0;
    Vec3 projected = add_scaled(source, forward, -dot(source, forward));
    if (norm(projected) < kEpsilon)
        return 16.0;
    projected = normalized(projected, "projected glare-source direction");
    double tau = std::acos(clamp_unit(dot(projected, up)));
    double position;
    if (dot(projected, up) >= 0.0) {
        tau *= 180.0/PI;
        sigma *= 180.0/PI;
        position = std::exp(
            (35.2-0.31889*tau-1.22*std::exp(-2.0*tau/9.0))*
                1.0e-3*sigma +
            (21.0+0.26667*tau-0.002963*tau*tau)*
                1.0e-5*sigma*sigma);
    } else {
        const double beta = std::atan(std::tan(sigma)*std::sqrt(
            1.0+0.3225*std::cos(tau)*std::cos(tau)))*180.0/PI;
        position = std::exp(6.49e-3*beta+21.0e-5*beta*beta);
    }
    return std::max(1.0, std::min(16.0, position));
}

double rgb_luminance(const float *rgb)
{
    return kLuminousEfficacy*std::max(0.0,
        kBrightness[0]*rgb[0]+kBrightness[1]*rgb[1]+kBrightness[2]*rgb[2]);
}

void trace_pending(DirectLobeBackend &backend,
                   const std::vector<std::string> &modifiers,
                   const Viewpoint &view, std::vector<TimeTree> &trees,
                   const std::vector<PendingCell> &pending)
{
    std::vector<DirectLobeRay> rays(pending.size());
    for (std::size_t i = 0; i < pending.size(); ++i) {
        rays[i].origin = view.origin;
        rays[i].direction = trees[pending[i].time].cells[pending[i].cell].direction;
        rays[i].target_modifier = pending[i].time;
    }
    const std::vector<float> values = backend.trace(modifiers, rays);
    if (values.size() != 3*pending.size())
        throw std::runtime_error("unexpected direct-lobe trace size");
    for (std::size_t i = 0; i < pending.size(); ++i)
        trees[pending[i].time].cells[pending[i].cell].luminance =
            rgb_luminance(values.data()+3*i);
}

void trace_proposals(DirectLobeBackend &backend,
                     const std::vector<std::string> &modifiers,
                     const Viewpoint &view, std::vector<TimeTree> &trees,
                     const std::vector<PendingProposal> &pending)
{
    std::vector<DirectLobeRay> rays(pending.size());
    for (std::size_t i = 0; i < pending.size(); ++i) {
        const DirectLobeProposal &proposal =
            trees[pending[i].time].proposals[pending[i].proposal];
        rays[i].origin = view.origin;
        rays[i].direction = proposal.direction;
        rays[i].target_modifier = pending[i].time;
    }
    const std::vector<float> values = backend.trace(modifiers, rays);
    if (values.size() != 3*pending.size())
        throw std::runtime_error("unexpected proposal trace size");
    for (std::size_t i = 0; i < pending.size(); ++i) {
        const double luminance = rgb_luminance(values.data()+3*i);
        trees[pending[i].time].proposal_luminance[
            pending[i].proposal] = luminance;
        if (std::getenv("DIRECTLOBE_DEBUG_VALUES")) {
            const DirectLobeProposal &proposal =
                trees[pending[i].time].proposals[pending[i].proposal];
            std::fprintf(stderr,
                "directlobe proposal: time=%lu proposal=%lu kinds=0x%x "
                "omega=%.9g direction=%.9g %.9g %.9g luminance=%.9g\n",
                static_cast<unsigned long>(pending[i].time),
                static_cast<unsigned long>(pending[i].proposal),
                proposal.kinds, proposal.solid_angle,
                proposal.direction[0], proposal.direction[1],
                proposal.direction[2], luminance);
        }
    }
}

void trace_proposals_and_pending(
    DirectLobeBackend &backend, const std::vector<std::string> &modifiers,
    const Viewpoint &view, std::vector<TimeTree> &trees,
    const std::vector<PendingProposal> &proposals,
    const std::vector<PendingCell> &pending)
{
    std::vector<DirectLobeRay> rays(proposals.size()+pending.size());
    for (std::size_t i = 0; i < proposals.size(); ++i) {
        const DirectLobeProposal &proposal =
            trees[proposals[i].time].proposals[proposals[i].proposal];
        rays[i].origin = view.origin;
        rays[i].direction = proposal.direction;
        rays[i].target_modifier = proposals[i].time;
    }
    for (std::size_t i = 0; i < pending.size(); ++i) {
        const std::size_t ray = proposals.size()+i;
        rays[ray].origin = view.origin;
        rays[ray].direction =
            trees[pending[i].time].cells[pending[i].cell].direction;
        rays[ray].target_modifier = pending[i].time;
    }
    const std::vector<float> values = backend.trace(modifiers, rays);
    if (values.size() != 3*rays.size())
        throw std::runtime_error("unexpected combined lobe trace size");
    for (std::size_t i = 0; i < proposals.size(); ++i) {
        const double luminance = rgb_luminance(values.data()+3*i);
        trees[proposals[i].time].proposal_luminance[
            proposals[i].proposal] = luminance;
    }
    for (std::size_t i = 0; i < pending.size(); ++i)
        trees[pending[i].time].cells[pending[i].cell].luminance =
            rgb_luminance(values.data()+3*(proposals.size()+i));
}

void trace_pending_batch(DirectLobeBackend &backend,
                         const std::vector<std::string> &modifiers,
                         std::vector<ViewLobeState> &states,
                         const std::vector<BatchedPendingCell> &pending)
{
    std::vector<DirectLobeRay> rays(pending.size());
    for (std::size_t i = 0; i < pending.size(); ++i) {
        ViewLobeState &state = states[pending[i].view];
        rays[i].origin = state.view.origin;
        rays[i].direction = state.trees[pending[i].time]
            .cells[pending[i].cell].direction;
        rays[i].target_modifier = pending[i].time;
    }
    const std::vector<float> values = backend.trace(modifiers, rays);
    if (values.size() != 3*pending.size())
        throw std::runtime_error("unexpected batched direct-lobe trace size");
    for (std::size_t i = 0; i < pending.size(); ++i) {
        ViewLobeState &state = states[pending[i].view];
        state.trees[pending[i].time].cells[pending[i].cell].luminance =
            rgb_luminance(values.data()+3*i);
    }
}

void trace_proposals_batch(DirectLobeBackend &backend,
                           const std::vector<std::string> &modifiers,
                           std::vector<ViewLobeState> &states,
                           const std::vector<BatchedPendingProposal> &pending)
{
    std::vector<DirectLobeRay> rays(pending.size());
    for (std::size_t i = 0; i < pending.size(); ++i) {
        ViewLobeState &state = states[pending[i].view];
        const DirectLobeProposal &proposal = state.trees[pending[i].time]
            .proposals[pending[i].proposal];
        rays[i].origin = state.view.origin;
        rays[i].direction = proposal.direction;
        rays[i].target_modifier = pending[i].time;
    }
    const std::vector<float> values = backend.trace(modifiers, rays);
    if (values.size() != 3*pending.size())
        throw std::runtime_error("unexpected batched proposal trace size");
    for (std::size_t i = 0; i < pending.size(); ++i) {
        ViewLobeState &state = states[pending[i].view];
        const double luminance = rgb_luminance(values.data()+3*i);
        state.trees[pending[i].time].proposal_luminance[
            pending[i].proposal] = luminance;
        if (std::getenv("DIRECTLOBE_DEBUG_VALUES")) {
            const DirectLobeProposal &proposal =
                state.trees[pending[i].time].proposals[pending[i].proposal];
            std::fprintf(stderr,
                "directlobe proposal: view=%lu time=%lu proposal=%lu "
                "kinds=0x%x omega=%.9g direction=%.9g %.9g %.9g "
                "luminance=%.9g\n",
                static_cast<unsigned long>(state.view_index+1),
                static_cast<unsigned long>(pending[i].time),
                static_cast<unsigned long>(pending[i].proposal),
                proposal.kinds, proposal.solid_angle,
                proposal.direction[0], proposal.direction[1],
                proposal.direction[2], luminance);
        }
    }
}

void trace_proposals_and_pending_batch(
    DirectLobeBackend &backend, const std::vector<std::string> &modifiers,
    std::vector<ViewLobeState> &states,
    const std::vector<BatchedPendingProposal> &proposals,
    const std::vector<BatchedPendingCell> &pending)
{
    std::vector<DirectLobeRay> rays(proposals.size()+pending.size());
    for (std::size_t i = 0; i < proposals.size(); ++i) {
        ViewLobeState &state = states[proposals[i].view];
        const DirectLobeProposal &proposal = state.trees[proposals[i].time]
            .proposals[proposals[i].proposal];
        rays[i].origin = state.view.origin;
        rays[i].direction = proposal.direction;
        rays[i].target_modifier = proposals[i].time;
    }
    for (std::size_t i = 0; i < pending.size(); ++i) {
        const std::size_t ray = proposals.size()+i;
        ViewLobeState &state = states[pending[i].view];
        rays[ray].origin = state.view.origin;
        rays[ray].direction = state.trees[pending[i].time]
            .cells[pending[i].cell].direction;
        rays[ray].target_modifier = pending[i].time;
    }
    const std::vector<float> values = backend.trace(modifiers, rays);
    if (values.size() != 3*rays.size())
        throw std::runtime_error(
            "unexpected combined batched lobe trace size");
    for (std::size_t i = 0; i < proposals.size(); ++i) {
        ViewLobeState &state = states[proposals[i].view];
        state.trees[proposals[i].time].proposal_luminance[
            proposals[i].proposal] = rgb_luminance(values.data()+3*i);
    }
    for (std::size_t i = 0; i < pending.size(); ++i) {
        ViewLobeState &state = states[pending[i].view];
        state.trees[pending[i].time].cells[pending[i].cell].luminance =
            rgb_luminance(values.data()+3*(proposals.size()+i));
    }
}

void trace_sun_pending(DirectLobeBackend &backend,
                       const std::vector<std::string> &modifiers,
                       const Viewpoint &view,
                       std::vector<SunDiskTree> &trees,
                       const std::vector<PendingCell> &pending)
{
    std::vector<DirectLobeRay> rays(pending.size());
    for (std::size_t i = 0; i < pending.size(); ++i) {
        rays[i].origin = view.origin;
        rays[i].direction =
            trees[pending[i].time].cells[pending[i].cell].direction;
        rays[i].target_modifier = pending[i].time;
    }
    const std::vector<float> values = backend.trace(
        modifiers, rays, DLT_STRAIGHT_SUN);
    if (values.size() != 3*pending.size())
        throw std::runtime_error("unexpected direct-sun trace size");
    for (std::size_t i = 0; i < pending.size(); ++i)
        trees[pending[i].time].cells[pending[i].cell].luminance =
            rgb_luminance(values.data()+3*i);
}

std::vector<unsigned char> select_sun_refinement(
    const SunDiskTree &tree, const Options &options, bool force_all)
{
    const std::vector<std::vector<int> > adjacency =
        build_adjacency(tree.cells);
    std::vector<unsigned char> refine(tree.cells.size(), 0);
    for (std::size_t i = 0; i < tree.cells.size(); ++i) {
        if (tree.cells[i].span <= 1)
            continue;
        if (force_all) {
            refine[i] = 1;
            continue;
        }
        for (std::size_t n = 0; n < adjacency[i].size(); ++n) {
            const int neighbor = adjacency[i][n];
            const bool threshold_crossing =
                (tree.cells[i].luminance > options.threshold) !=
                (tree.cells[neighbor].luminance > options.threshold);
            if (threshold_crossing ||
                    std::fabs(tree.cells[i].luminance-
                              tree.cells[neighbor].luminance) >=
                        options.sun_disk_gradient) {
                refine[i] = 1;
                if (tree.cells[neighbor].span > 1)
                    refine[neighbor] = 1;
            }
        }
    }
    for (int guard = 0; guard < options.sun_disk_guard; ++guard) {
        std::vector<unsigned char> expanded = refine;
        for (std::size_t i = 0; i < refine.size(); ++i)
            if (refine[i])
                for (std::size_t n = 0; n < adjacency[i].size(); ++n) {
                    const int neighbor = adjacency[i][n];
                    if (tree.cells[neighbor].span > 1)
                        expanded[neighbor] = 1;
                }
        refine.swap(expanded);
    }
    return refine;
}

std::size_t refine_sun_tree(SunDiskTree &tree,
                            const std::vector<unsigned char> &refine,
                            int dimension, const Sun &sun,
                            const Viewpoint &view, std::size_t time,
                            std::vector<PendingCell> &pending)
{
    std::vector<Cell> next;
    next.reserve(tree.cells.size());
    std::size_t parents = 0;
    for (std::size_t i = 0; i < tree.cells.size(); ++i) {
        if (!refine[i]) {
            next.push_back(tree.cells[i]);
            continue;
        }
        ++parents;
        const int child_span = tree.cells[i].span/2;
        for (int child_y = 0; child_y < 2; ++child_y)
            for (int child_x = 0; child_x < 2; ++child_x) {
                Cell child;
                child.x0 = tree.cells[i].x0+child_x*child_span;
                child.y0 = tree.cells[i].y0+child_y*child_span;
                child.span = child_span;
                child.direction = sample_solar_disk(
                    (child.x0+0.5*child.span)/dimension,
                    (child.y0+0.5*child.span)/dimension, sun);
                child.luminance = 0.0;
                child.solid_angle_override = 0.0;
                next.push_back(child);
                if (dot(child.direction, view.direction) > 0.0) {
                    PendingCell entry;
                    entry.time = time;
                    entry.cell = next.size()-1;
                    pending.push_back(entry);
                }
            }
    }
    tree.cells.swap(next);
    return parents;
}

double integrate_sun_tree(const SunDiskTree &tree, int dimension,
                          const Sun &sun, const Viewpoint &view,
                          const Options &options)
{
    const std::vector<std::vector<int> > adjacency =
        build_adjacency(tree.cells);
    std::vector<unsigned char> visited(tree.cells.size(), 0);
    double contrast = 0.0;
    for (std::size_t start = 0; start < tree.cells.size(); ++start) {
        if (visited[start] || tree.cells[start].luminance <= options.threshold)
            continue;
        std::queue<int> queue;
        queue.push(static_cast<int>(start));
        visited[start] = 1;
        int count = 0;
        double omega_sum = 0.0;
        double luminance_integral = 0.0;
        Vec3 centroid = {{0.0, 0.0, 0.0}};
        while (!queue.empty()) {
            const int current = queue.front();
            queue.pop();
            const Cell &cell = tree.cells[current];
            const double fraction = static_cast<double>(cell.span)/dimension;
            const double omega = sun.solid_angle*fraction*fraction;
            ++count;
            omega_sum += omega;
            luminance_integral += cell.luminance*omega;
            centroid = add_scaled(centroid, cell.direction,
                                  cell.luminance*omega);
            for (std::size_t n = 0; n < adjacency[current].size(); ++n) {
                const int neighbor = adjacency[current][n];
                if (!visited[neighbor] &&
                        tree.cells[neighbor].luminance > options.threshold) {
                    visited[neighbor] = 1;
                    queue.push(neighbor);
                }
            }
        }
        if (count < options.min_cluster_cells || omega_sum <= 0.0 ||
                norm(centroid) <= kEpsilon)
            continue;
        const double average = luminance_integral/omega_sum;
        const double position = guth_position_index(
            normalized(centroid, "visible-sun centroid"),
            view.direction, options.up);
        contrast += average*average*omega_sum/(position*position);
    }
    return contrast;
}

bool has_unresolved_sun_boundary(const SunDiskTree &tree,
                                 const Options &options)
{
    const std::vector<unsigned char> refine =
        select_sun_refinement(tree, options, false);
    return std::find(refine.begin(), refine.end(), 1) != refine.end();
}

std::vector<double> evaluate_direct_sun(
    DirectLobeBackend &backend, const std::vector<std::string> &modifiers,
    const std::vector<Sun> &suns, const Viewpoint &view,
    std::size_t view_index, const Options &options)
{
    const int remaining_levels =
        options.sun_disk_levels-options.sun_disk_min_levels;
    const int scale = 1 << remaining_levels;
    const int initial_resolution = options.sun_disk_resolution*
        (1 << options.sun_disk_min_levels);
    const int dimension = initial_resolution*scale;
    std::vector<SunDiskTree> trees(suns.size());
    std::vector<PendingCell> pending;
    const std::size_t initial_count =
        static_cast<std::size_t>(initial_resolution)*initial_resolution;
    for (std::size_t time = 0; time < suns.size(); ++time) {
        if (!suns[time].active)
            continue;
        if (dot(suns[time].direction, view.direction) <=
                -std::sin(0.5*suns[time].angular_diameter))
            continue;
        trees[time].cells.reserve(initial_count);
        for (int row = 0; row < initial_resolution; ++row)
            for (int column = 0; column < initial_resolution;
                    ++column) {
                Cell cell;
                cell.x0 = column*scale;
                cell.y0 = row*scale;
                cell.span = scale;
                cell.direction = sample_solar_disk(
                    (cell.x0+0.5*cell.span)/dimension,
                    (cell.y0+0.5*cell.span)/dimension, suns[time]);
                cell.luminance = 0.0;
                cell.solid_angle_override = 0.0;
                trees[time].cells.push_back(cell);
                if (dot(cell.direction, view.direction) > 0.0) {
                    PendingCell entry;
                    entry.time = time;
                    entry.cell = trees[time].cells.size()-1;
                    pending.push_back(entry);
                }
            }
    }
    if (!options.quiet)
        std::fprintf(stderr,
            "directlobecontrast: view %lu initial solar disks, "
            "%lu targeted rays\n",
            static_cast<unsigned long>(view_index+1),
            static_cast<unsigned long>(pending.size()));
    trace_sun_pending(backend, modifiers, view, trees, pending);

    std::vector<double> contrast(suns.size(), 0.0);
    for (std::size_t time = 0; time < suns.size(); ++time)
        if (!trees[time].cells.empty())
            contrast[time] = integrate_sun_tree(
                trees[time], dimension, suns[time], view, options);

    for (int level = options.sun_disk_min_levels;
            level < options.sun_disk_levels; ++level) {
        pending.clear();
        std::size_t parents = 0;
        for (std::size_t time = 0; time < suns.size(); ++time) {
            if (trees[time].cells.empty())
                continue;
            const std::vector<unsigned char> refine =
                select_sun_refinement(trees[time], options, false);
            parents += refine_sun_tree(
                trees[time], refine, dimension, suns[time], view,
                time, pending);
        }
        if (!parents)
            break;
        if (!options.quiet)
            std::fprintf(stderr,
                "directlobecontrast: view %lu solar-disk level %d, "
                "%lu parents, %lu targeted rays\n",
                static_cast<unsigned long>(view_index+1), level+1,
                static_cast<unsigned long>(parents),
                static_cast<unsigned long>(pending.size()));
        trace_sun_pending(backend, modifiers, view, trees, pending);

        std::vector<double> next(suns.size(), 0.0);
        double maximum_change = 0.0;
        bool unresolved = false;
        for (std::size_t time = 0; time < suns.size(); ++time) {
            if (trees[time].cells.empty())
                continue;
            next[time] = integrate_sun_tree(
                trees[time], dimension, suns[time], view, options);
            const double scale_value = std::max(
                std::fabs(next[time]), 1.0e-12);
            maximum_change = std::max(maximum_change,
                std::fabs(next[time]-contrast[time])/scale_value);
            unresolved = unresolved ||
                has_unresolved_sun_boundary(trees[time], options);
        }
        contrast.swap(next);
        if (!options.quiet)
            std::fprintf(stderr,
                "directlobecontrast: view %lu solar-disk level %d "
                "maximum contrast change %.6g%s\n",
                static_cast<unsigned long>(view_index+1), level+1,
                maximum_change, unresolved ? " (boundary unresolved)" : "");
        if (!unresolved && maximum_change < options.sun_disk_tolerance)
            break;
    }
    return contrast;
}

std::vector<double> evaluate_direct_sun_batch(
    DirectLobeBackend &backend, const std::vector<std::string> &modifiers,
    const std::vector<Sun> &suns, const std::vector<Viewpoint> &views,
    const Options &options, std::vector<float> *irradiance_output,
    std::vector<float> *total_irradiance_output)
{
    std::vector<DirectLobeSensor> sensors(views.size());
    for (std::size_t view = 0; view < views.size(); ++view) {
        sensors[view].origin = views[view].origin;
        sensors[view].direction = views[view].direction;
    }
    if (!options.quiet)
        std::fprintf(stderr,
            "directlobecontrast: batch direct-sun sampling, %lu sensors, "
            "%d samples per sensor\n",
            static_cast<unsigned long>(sensors.size()), options.sun_samples);
    std::vector<float> total_irradiance;
    std::vector<float> visible_fraction;
    std::vector<float> irradiance = backend.trace_sun_irradiance(
        modifiers, sensors, options.sun_samples,
        total_irradiance_output ? &total_irradiance : NULL,
        &visible_fraction);
    const std::size_t expected = 3*views.size()*suns.size();
    if (irradiance.size() != expected ||
            visible_fraction.size() != views.size()*suns.size() ||
            (total_irradiance_output &&
             total_irradiance.size() != expected))
        throw std::runtime_error("unexpected batch direct-sun trace size");

    std::vector<double> contrast(views.size()*suns.size(), 0.0);
    for (std::size_t view = 0; view < views.size(); ++view)
        for (std::size_t time = 0; time < suns.size(); ++time) {
            if (!suns[time].active || suns[time].solid_angle <= kEpsilon)
                continue;
            const double cosine = dot(
                views[view].direction, suns[time].direction);
            if (cosine <= kEpsilon)
                continue;
            const std::size_t index = view*suns.size()+time;
            const double solar_illuminance = rgb_luminance(
                irradiance.data()+3*index);
            if (solar_illuminance <= kEpsilon)
                continue;
            const double fraction = std::max(0.0, std::min(1.0,
                static_cast<double>(visible_fraction[index])));
            const double visible_solid_angle =
                fraction*suns[time].solid_angle;
            if (visible_solid_angle <= kEpsilon)
                continue;
            /* Irradiance already averages the blocked and unblocked solar-disk
             * samples.  Divide by their visible solid angle so obstruction
             * changes source area rather than diluting source luminance. */
            const double average_luminance = solar_illuminance/
                (visible_solid_angle*cosine);
            if (average_luminance <= options.threshold)
                continue;
            const double position = guth_position_index(
                suns[time].direction, views[view].direction, options.up);
            contrast[index] = average_luminance*average_luminance*
                visible_solid_angle/(position*position);
        }
    if (irradiance_output)
        irradiance_output->swap(irradiance);
    if (total_irradiance_output)
        total_irradiance_output->swap(total_irradiance);
    return contrast;
}

std::size_t nearest_cell(const std::vector<Cell> &cells, const Vec3 &direction)
{
    std::size_t nearest = 0;
    double best = -2.0;
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const double cosine = dot(cells[i].direction, direction);
        if (cosine > best) {
            best = cosine;
            nearest = i;
        }
    }
    return nearest;
}

double cell_solid_angle(const Cell &cell, int dimension)
{
    if (cell.solid_angle_override > 0.0)
        return cell.solid_angle_override;
    const double fraction = static_cast<double>(cell.span)/dimension;
    return 2.0*PI*fraction*fraction;
}

double directional_contrast(double luminance, double solid_angle,
                            const Vec3 &direction, const Viewpoint &view,
                            const Options &options)
{
    if (luminance <= options.threshold || solid_angle <= 0.0)
        return 0.0;
    const double position = guth_position_index(
        direction, view.direction, options.up);
    return luminance*luminance*solid_angle/(position*position);
}

double grouped_contrast(const TimeTree &tree,
                        const std::vector<std::size_t> &indices,
                        int dimension, const Viewpoint &view,
                        const Options &options)
{
    double omega_sum = 0.0;
    double luminance_integral = 0.0;
    Vec3 centroid = {{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const Cell &cell = tree.cells[indices[i]];
        if (cell.luminance <= options.threshold)
            continue;
        const double omega = cell_solid_angle(cell, dimension);
        omega_sum += omega;
        luminance_integral += cell.luminance*omega;
        centroid = add_scaled(centroid, cell.direction,
                              cell.luminance*omega);
    }
    if (omega_sum <= 0.0 || norm(centroid) <= kEpsilon)
        return 0.0;
    return directional_contrast(
        luminance_integral/omega_sum, omega_sum,
        normalized(centroid, "refinement centroid"), view, options);
}

struct RefinementCandidate {
    std::vector<std::size_t> cells;
    double uncertainty;
};

std::vector<unsigned char> select_refinement(
    const TimeTree &tree, int dimension, const Viewpoint &view,
    double current_contrast, const Options &options,
    double *estimated_uncertainty, double *residual_uncertainty)
{
    const std::vector<std::vector<int> > adjacency =
        build_adjacency(tree.cells);
    std::vector<unsigned char> refine(tree.cells.size(), 0);
    for (std::size_t i = 0; i < tree.cells.size(); ++i) {
        if (tree.cells[i].span <= 1)
            continue;
        const bool pilot_cell = !tree.cells[i].has_parent_estimate;
        if (pilot_cell &&
                tree.cells[i].luminance >= options.adaptive_trigger)
            refine[i] = 1;
        for (std::size_t n = 0; n < adjacency[i].size(); ++n) {
            const int neighbor = adjacency[i][n];
            const bool threshold_crossing =
                (tree.cells[i].luminance <= options.threshold) !=
                (tree.cells[neighbor].luminance <= options.threshold);
            if (threshold_crossing) {
                refine[i] = 1;
                if (tree.cells[neighbor].span > 1)
                    refine[neighbor] = 1;
            } else if ((pilot_cell ||
                        !tree.cells[neighbor].has_parent_estimate) &&
                    std::fabs(tree.cells[i].luminance-
                        tree.cells[neighbor].luminance) >=
                        options.adaptive_gradient) {
                refine[i] = 1;
                if (tree.cells[neighbor].span > 1)
                    refine[neighbor] = 1;
            }
        }
    }

    typedef std::tuple<int, int, int> ParentKey;
    std::map<ParentKey, std::vector<std::size_t> > sibling_groups;
    for (std::size_t i = 0; i < tree.cells.size(); ++i) {
        const Cell &cell = tree.cells[i];
        if (cell.span <= 1)
            continue;
        const int parent_span = 2*cell.span;
        sibling_groups[ParentKey(
            cell.x0/parent_span, cell.y0/parent_span, cell.span)].push_back(i);
    }
    std::vector<RefinementCandidate> candidates;
    double total_uncertainty = 0.0;
    for (std::map<ParentKey, std::vector<std::size_t> >::const_iterator group =
            sibling_groups.begin(); group != sibling_groups.end(); ++group) {
        if (group->second.size() != 4)
            continue;
        const Cell &first = tree.cells[group->second[0]];
        if (first.has_parent_estimate) {
            /* Compare the parent's coarse DGP contribution with the source
             * reconstructed from its four equal-solid-angle children. */
            bool mixed_threshold = false;
            const bool first_bright = first.luminance > options.threshold;
            for (std::size_t i = 1; i < group->second.size(); ++i)
                mixed_threshold = mixed_threshold ||
                    ((tree.cells[group->second[i]].luminance >
                        options.threshold) != first_bright);
            const bool parent_bright =
                first.parent_luminance > options.threshold;
            if (mixed_threshold || parent_bright != first_bright)
                for (std::size_t i = 0; i < group->second.size(); ++i)
                    refine[group->second[i]] = 1;

            const double coarse = directional_contrast(
                first.parent_luminance, first.parent_solid_angle,
                first.parent_direction, view, options);
            const double fine = grouped_contrast(
                tree, group->second, dimension, view, options);
            const double uncertainty = std::fabs(fine-coarse);
            if (uncertainty > 0.0) {
                RefinementCandidate candidate;
                candidate.cells = group->second;
                candidate.uncertainty = uncertainty;
                candidates.push_back(candidate);
                total_uncertainty += uncertainty;
            }
            continue;
        }

        double mean = 0.0;
        for (std::size_t i = 0; i < group->second.size(); ++i)
            mean += tree.cells[group->second[i]].luminance;
        mean /= group->second.size();
        double variance = 0.0;
        for (std::size_t i = 0; i < group->second.size(); ++i) {
            const double difference =
                tree.cells[group->second[i]].luminance-mean;
            variance += difference*difference;
        }
        const double deviation = std::sqrt(variance/group->second.size());
        if (deviation >= options.adaptive_variance)
            for (std::size_t i = 0; i < group->second.size(); ++i)
                refine[group->second[i]] = 1;
    }
    for (std::size_t proposal = 0;
            proposal < tree.proposals.size(); ++proposal) {
        const double luminance = tree.proposal_luminance[proposal];
        const std::size_t seed = nearest_cell(
            tree.cells, tree.proposals[proposal].direction);
        if (tree.cells[seed].span <= 1)
            continue;
        const bool threshold_crossing =
            (tree.cells[seed].luminance <= options.threshold) !=
            (luminance <= options.threshold);
        const unsigned int sharp_kinds =
            tree.proposals[proposal].kinds & (DLP_REFRACTION|DLP_PRISM);
        if (threshold_crossing ||
                (!tree.cells[seed].has_parent_estimate &&
                 luminance >= options.adaptive_trigger) ||
                (sharp_kinds && luminance >= options.adaptive_trigger &&
                 cell_solid_angle(tree.cells[seed], dimension) >
                    tree.proposals[proposal].solid_angle))
            refine[seed] = 1;
    }

    std::sort(candidates.begin(), candidates.end(),
        [](const RefinementCandidate &a, const RefinementCandidate &b) {
            return a.uncertainty > b.uncertainty;
        });
    double unresolved_uncertainty = 0.0;
    for (std::size_t candidate = 0; candidate < candidates.size(); ++candidate) {
        bool already_selected = false;
        for (std::size_t i = 0; i < candidates[candidate].cells.size(); ++i)
            already_selected = already_selected ||
                refine[candidates[candidate].cells[i]];
        if (already_selected) {
            for (std::size_t i = 0; i < candidates[candidate].cells.size(); ++i)
                refine[candidates[candidate].cells[i]] = 1;
        } else
            unresolved_uncertainty += candidates[candidate].uncertainty;
    }
    /* Spend the relative error budget on the largest unresolved contributors
     * first. Mandatory threshold boundaries have already been selected. */
    const double uncertainty_scale = std::max(
        std::fabs(current_contrast), kEpsilon);
    const double uncertainty_budget =
        options.adaptive_uncertainty*uncertainty_scale;
    for (std::size_t candidate = 0;
            candidate < candidates.size() &&
            unresolved_uncertainty > uncertainty_budget; ++candidate) {
        bool already_selected = false;
        for (std::size_t i = 0; i < candidates[candidate].cells.size(); ++i)
            already_selected = already_selected ||
                refine[candidates[candidate].cells[i]];
        if (already_selected)
            continue;
        for (std::size_t i = 0; i < candidates[candidate].cells.size(); ++i)
            refine[candidates[candidate].cells[i]] = 1;
        unresolved_uncertainty = std::max(0.0,
            unresolved_uncertainty-candidates[candidate].uncertainty);
    }
    if (estimated_uncertainty)
        *estimated_uncertainty = total_uncertainty;
    if (residual_uncertainty)
        *residual_uncertainty = unresolved_uncertainty;

    for (int guard = 0; guard < options.adaptive_guard; ++guard) {
        std::vector<unsigned char> expanded = refine;
        for (std::size_t i = 0; i < refine.size(); ++i)
            if (refine[i])
                for (std::size_t n = 0; n < adjacency[i].size(); ++n) {
                    const int neighbor = adjacency[i][n];
                    if (tree.cells[neighbor].span > 1)
                        expanded[neighbor] = 1;
                }
        refine.swap(expanded);
    }
    return refine;
}

void seed_finest_cells(TimeTree &tree, int dimension)
{
    std::map<std::size_t, std::size_t> selected;
    for (std::size_t proposal = 0;
            proposal < tree.proposals.size(); ++proposal) {
        if (tree.proposal_luminance[proposal] <= 0.0)
            continue;
        const std::size_t cell = nearest_cell(
            tree.cells, tree.proposals[proposal].direction);
        if (tree.cells[cell].span != 1)
            continue;
        const std::map<std::size_t, std::size_t>::const_iterator existing =
            selected.find(cell);
        if (existing == selected.end() ||
                tree.proposal_luminance[proposal] >
                tree.proposal_luminance[existing->second])
            selected[cell] = proposal;
    }
    for (std::map<std::size_t, std::size_t>::const_iterator entry =
            selected.begin(); entry != selected.end(); ++entry) {
        tree.cells[entry->first].direction =
            tree.proposals[entry->second].direction;
        if (tree.proposals[entry->second].kinds &
                (DLP_REFRACTION|DLP_PRISM)) {
            const double fraction =
                static_cast<double>(tree.cells[entry->first].span)/dimension;
            tree.cells[entry->first].solid_angle_override = std::min(
                2.0*PI*fraction*fraction,
                tree.proposals[entry->second].solid_angle);
        }
    }
}

void apply_finest_proposal_weights(TimeTree &tree)
{
    /* A refracted solar disk is a sharp output-domain source and therefore
     * owns its physical solid angle. Rough-transmission disk probes only
     * locate a broad output lobe; their input-domain weights must not replace
     * a Shirley-Chiu cell's output solid angle. */
    struct Aggregate {
        double omega = 0.0;
        double luminance_integral = 0.0;
        double brightest = -1.0;
        std::size_t brightest_proposal = 0;
    };
    std::map<std::size_t, Aggregate> aggregates;
    for (std::size_t proposal = 0;
            proposal < tree.proposals.size(); ++proposal) {
        const unsigned int kinds = tree.proposals[proposal].kinds;
        if (!(kinds & (DLP_REFRACTION|DLP_PRISM)) ||
                tree.proposal_luminance[proposal] <= 0.0)
            continue;
        const std::size_t cell = nearest_cell(
            tree.cells, tree.proposals[proposal].direction);
        if (tree.cells[cell].span != 1)
            continue;
        Aggregate &aggregate = aggregates[cell];
        const double omega = tree.proposals[proposal].solid_angle;
        aggregate.omega += omega;
        aggregate.luminance_integral +=
            tree.proposal_luminance[proposal]*omega;
        if (tree.proposal_luminance[proposal] > aggregate.brightest) {
            aggregate.brightest = tree.proposal_luminance[proposal];
            aggregate.brightest_proposal = proposal;
        }
    }
    for (std::map<std::size_t, Aggregate>::const_iterator entry =
            aggregates.begin(); entry != aggregates.end(); ++entry) {
        if (!(entry->second.omega > 0.0))
            continue;
        Cell &cell = tree.cells[entry->first];
        cell.direction = tree.proposals[
            entry->second.brightest_proposal].direction;
        cell.luminance = entry->second.luminance_integral/
            entry->second.omega;
        cell.solid_angle_override = entry->second.omega;
    }
}

std::size_t refine_tree(TimeTree &tree,
                        const std::vector<unsigned char> &refine,
                        int dimension, const Viewpoint &view,
                        const Vec3 &right, const Vec3 &up,
                        std::size_t time, std::vector<PendingCell> &pending)
{
    std::vector<Cell> next;
    std::size_t parents = 0;
    for (std::size_t i = 0; i < tree.cells.size(); ++i) {
        if (!refine[i]) {
            next.push_back(tree.cells[i]);
            continue;
        }
        ++parents;
        const int child_span = tree.cells[i].span/2;
        for (int child_y = 0; child_y < 2; ++child_y)
            for (int child_x = 0; child_x < 2; ++child_x) {
                Cell child;
                child.x0 = tree.cells[i].x0+child_x*child_span;
                child.y0 = tree.cells[i].y0+child_y*child_span;
                child.span = child_span;
                child.direction = sample_direction(
                    (child.x0+0.5*child.span)/dimension,
                    (child.y0+0.5*child.span)/dimension,
                    view.direction, right, up);
                child.luminance = 0.0;
                child.solid_angle_override = 0.0;
                child.parent_luminance = tree.cells[i].luminance;
                child.parent_direction = tree.cells[i].direction;
                child.parent_solid_angle =
                    cell_solid_angle(tree.cells[i], dimension);
                child.has_parent_estimate = true;
                next.push_back(child);
                PendingCell entry;
                entry.time = time;
                entry.cell = next.size()-1;
                pending.push_back(entry);
            }
    }
    tree.cells.swap(next);
    return parents;
}

double integrate_tree(const TimeTree &tree, int dimension,
                      const Viewpoint &view, const Options &options)
{
    const std::vector<std::vector<int> > adjacency =
        build_adjacency(tree.cells);
    std::vector<unsigned char> visited(tree.cells.size(), 0);
    double contrast = 0.0;
    for (std::size_t start = 0; start < tree.cells.size(); ++start) {
        if (visited[start] || tree.cells[start].luminance <= options.threshold)
            continue;
        std::queue<int> queue;
        queue.push(static_cast<int>(start));
        visited[start] = 1;
        int count = 0;
        double omega_sum = 0.0;
        double luminance_integral = 0.0;
        Vec3 centroid = {{0.0, 0.0, 0.0}};
        while (!queue.empty()) {
            const int current = queue.front();
            queue.pop();
            const Cell &cell = tree.cells[current];
            const double omega = cell_solid_angle(cell, dimension);
            ++count;
            omega_sum += omega;
            luminance_integral += cell.luminance*omega;
            centroid = add_scaled(centroid, cell.direction,
                                  cell.luminance*omega);
            for (std::size_t n = 0; n < adjacency[current].size(); ++n) {
                const int neighbor = adjacency[current][n];
                if (!visited[neighbor] &&
                        tree.cells[neighbor].luminance > options.threshold) {
                    visited[neighbor] = 1;
                    queue.push(neighbor);
                }
            }
        }
        if (count < options.min_cluster_cells || omega_sum <= 0.0 ||
                norm(centroid) <= kEpsilon)
            continue;
        const double average = luminance_integral/omega_sum;
        const double position = guth_position_index(
            normalized(centroid, "lobe centroid"), view.direction, options.up);
        contrast += average*average*omega_sum/(position*position);
    }
    return contrast;
}

bool has_unresolved_proposal(const TimeTree &tree, int dimension,
                             const Options &options)
{
    for (std::size_t proposal = 0;
            proposal < tree.proposals.size(); ++proposal) {
        const std::size_t cell = nearest_cell(
            tree.cells, tree.proposals[proposal].direction);
        if (tree.cells[cell].span <= 1)
            continue;
        const double luminance = tree.proposal_luminance[proposal];
        const bool threshold_crossing =
            (tree.cells[cell].luminance <= options.threshold) !=
            (luminance <= options.threshold);
        if (threshold_crossing)
            return true;
        const unsigned int sharp_kinds =
            tree.proposals[proposal].kinds & (DLP_REFRACTION|DLP_PRISM);
        if (sharp_kinds && luminance >= options.adaptive_trigger &&
                cell_solid_angle(tree.cells[cell], dimension) >
                    tree.proposals[proposal].solid_angle)
            return true;
        if (!sharp_kinds && !tree.cells[cell].has_parent_estimate &&
                luminance >= options.adaptive_trigger)
            return true;
    }
    return false;
}

void append_transmission_disk_probes(
    TimeTree &tree, const DirectLobeProposal &base, const Sun &sun,
    const Viewpoint &view, int resolution, std::size_t time,
    std::vector<PendingProposal> &pending)
{
    const double sample_omega = base.solid_angle/
        (static_cast<double>(resolution)*resolution);
    for (int row = 0; row < resolution; ++row)
        for (int column = 0; column < resolution; ++column) {
            DirectLobeProposal probe = base;
            probe.direction = sample_solar_disk(
                (column+0.5)/resolution, (row+0.5)/resolution, sun);
            probe.solid_angle = sample_omega;
            probe.kinds = DLP_TRANSMISSION;
            if (dot(probe.direction, view.direction) <= 0.0)
                continue;
            tree.proposals.push_back(probe);
            tree.proposal_luminance.push_back(0.0);
            PendingProposal entry;
            entry.time = time;
            entry.proposal = tree.proposals.size()-1;
            pending.push_back(entry);
        }
}

void append_transmission_disk_probes_batch(
    TimeTree &tree, const DirectLobeProposal &base, const Sun &sun,
    const Viewpoint &view, int resolution, std::size_t state_index,
    std::size_t time, std::vector<BatchedPendingProposal> &pending)
{
    const double sample_omega = base.solid_angle/
        (static_cast<double>(resolution)*resolution);
    for (int row = 0; row < resolution; ++row)
        for (int column = 0; column < resolution; ++column) {
            DirectLobeProposal probe = base;
            probe.direction = sample_solar_disk(
                (column+0.5)/resolution, (row+0.5)/resolution, sun);
            probe.solid_angle = sample_omega;
            probe.kinds = DLP_TRANSMISSION;
            if (dot(probe.direction, view.direction) <= 0.0)
                continue;
            tree.proposals.push_back(probe);
            tree.proposal_luminance.push_back(0.0);
            BatchedPendingProposal entry;
            entry.view = state_index;
            entry.time = time;
            entry.proposal = tree.proposals.size()-1;
            pending.push_back(entry);
        }
}

bool needs_full_transmission_disk(const TimeTree &tree,
                                  const Options &options)
{
    std::size_t samples = 0;
    std::size_t positive = 0;
    bool above_threshold = false;
    bool below_threshold = false;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = 0.0;
    for (std::size_t proposal = 0;
            proposal < tree.proposals.size(); ++proposal) {
        if (!(tree.proposals[proposal].kinds & DLP_TRANSMISSION))
            continue;
        const double luminance = tree.proposal_luminance[proposal];
        ++samples;
        positive += luminance > 0.0 ? 1u : 0u;
        above_threshold = above_threshold || luminance > options.threshold;
        below_threshold = below_threshold || luminance <= options.threshold;
        minimum = std::min(minimum, luminance);
        maximum = std::max(maximum, luminance);
    }
    if (!samples)
        return false;

    /* Complete the solar disk only when the pilot found a boundary or a
     * strongly varying signal.  A coarse view-hemisphere hit also rescues a
     * lobe whose small visible solar fragment fell between all pilot rays. */
    if (positive && positive < samples)
        return true;
    if (above_threshold && below_threshold)
        return true;
    if (maximum-minimum >= options.adaptive_gradient)
        return true;
    if (!positive)
        for (std::size_t cell = 0; cell < tree.cells.size(); ++cell)
            if (tree.cells[cell].luminance > 0.0)
                return true;
    return false;
}

bool has_transmission_evidence(const TimeTree &tree)
{
    for (std::size_t proposal = 0;
            proposal < tree.proposals.size(); ++proposal)
        if ((tree.proposals[proposal].kinds & DLP_TRANSMISSION) &&
                tree.proposal_luminance[proposal] > 0.0)
            return true;
    for (std::size_t cell = 0; cell < tree.cells.size(); ++cell)
        if (tree.cells[cell].luminance > 0.0)
            return true;
    return false;
}

std::vector<unsigned char> select_full_transmission_disks(
    const std::vector<TimeTree> &trees, const std::vector<Sun> &suns,
    const Options &options)
{
    std::vector<unsigned char> selected(trees.size(), 0u);
    std::vector<std::size_t> evidence;
    for (std::size_t time = 0; time < trees.size(); ++time) {
        selected[time] = needs_full_transmission_disk(
            trees[time], options) ? 1u : 0u;
        if (has_transmission_evidence(trees[time]))
            evidence.push_back(time);
    }
    if (options.lobe_sun_disk_pilot_guard_angle <= 0.0 || evidence.empty())
        return selected;
    const double guard_cosine = std::cos(
        options.lobe_sun_disk_pilot_guard_angle*PI/180.0);
    for (std::size_t time = 0; time < trees.size(); ++time) {
        if (selected[time] || trees[time].cells.empty())
            continue;
        for (std::size_t seed = 0; seed < evidence.size(); ++seed)
            if (dot(suns[time].direction,
                    suns[evidence[seed]].direction) >= guard_cosine) {
                selected[time] = 1u;
                break;
            }
    }
    return selected;
}

std::vector<double> evaluate_view(DirectLobeBackend &backend,
                                  const std::vector<std::string> &modifiers,
                                  const std::vector<Sun> &suns,
                                  const std::vector<std::vector<
                                      DirectLobeProposal> > &proposal_sets,
                                  const Viewpoint &view,
                                  std::size_t view_index,
                                  const Options &options)
{
    const int scale = 1 << options.adaptive_levels;
    const int dimension = options.resolution*scale;
    Vec3 right, up;
    view_basis(view.direction, options.up, right, up);
    std::vector<TimeTree> trees(suns.size());
    std::vector<PendingCell> pending;
    std::vector<PendingProposal> proposal_pending;
    const std::size_t initial_count =
        static_cast<std::size_t>(options.resolution)*options.resolution;
    for (std::size_t time = 0; time < suns.size(); ++time) {
        if (!suns[time].active)
            continue;
        for (std::size_t proposal = 0;
                proposal < proposal_sets[time].size(); ++proposal) {
            const DirectLobeProposal &base = proposal_sets[time][proposal];
            const unsigned int non_transmission =
                base.kinds & ~static_cast<unsigned int>(DLP_TRANSMISSION);
            if (non_transmission &&
                    dot(base.direction, view.direction) > 0.0) {
                DirectLobeProposal retained = base;
                retained.kinds = non_transmission;
                trees[time].proposals.push_back(retained);
                trees[time].proposal_luminance.push_back(0.0);
                PendingProposal probe;
                probe.time = time;
                probe.proposal = trees[time].proposals.size()-1;
                proposal_pending.push_back(probe);
            }
            if (!(base.kinds & DLP_TRANSMISSION))
                continue;
            const int pilot_resolution =
                options.lobe_sun_disk_pilot_resolution > 0 &&
                options.lobe_sun_disk_pilot_resolution <
                    options.lobe_sun_disk_resolution ?
                options.lobe_sun_disk_pilot_resolution :
                options.lobe_sun_disk_resolution;
            append_transmission_disk_probes(
                trees[time], base, suns[time], view, pilot_resolution,
                time, proposal_pending);
        }
        trees[time].cells.reserve(initial_count);
        for (int row = 0; row < options.resolution; ++row)
            for (int column = 0; column < options.resolution; ++column) {
                Cell cell;
                cell.x0 = column*scale;
                cell.y0 = row*scale;
                cell.span = scale;
                cell.direction = sample_direction(
                    (cell.x0+0.5*cell.span)/dimension,
                    (cell.y0+0.5*cell.span)/dimension,
                    view.direction, right, up);
                cell.luminance = 0.0;
                cell.solid_angle_override = 0.0;
                trees[time].cells.push_back(cell);
                PendingCell entry;
                entry.time = time;
                entry.cell = trees[time].cells.size()-1;
                pending.push_back(entry);
            }
    }
    if (!options.quiet)
        std::fprintf(stderr,
            "directlobecontrast: view %lu proposal pass, %lu targeted rays\n",
            static_cast<unsigned long>(view_index+1),
            static_cast<unsigned long>(proposal_pending.size()));
    if (!options.quiet)
        std::fprintf(stderr,
            "directlobecontrast: view %lu initial grid, %lu targeted rays\n",
            static_cast<unsigned long>(view_index+1),
            static_cast<unsigned long>(pending.size()));
    trace_proposals_and_pending(
        backend, modifiers, view, trees, proposal_pending, pending);

    if (options.lobe_sun_disk_pilot_resolution > 0 &&
            options.lobe_sun_disk_pilot_resolution <
                options.lobe_sun_disk_resolution) {
        proposal_pending.clear();
        const std::vector<unsigned char> complete =
            select_full_transmission_disks(trees, suns, options);
        for (std::size_t time = 0; time < suns.size(); ++time) {
            if (!complete[time])
                continue;
            for (std::size_t proposal = 0;
                    proposal < proposal_sets[time].size(); ++proposal)
                if (proposal_sets[time][proposal].kinds & DLP_TRANSMISSION)
                    append_transmission_disk_probes(
                        trees[time], proposal_sets[time][proposal],
                        suns[time], view, options.lobe_sun_disk_resolution,
                        time, proposal_pending);
        }
        if (!options.quiet)
            std::fprintf(stderr,
                "directlobecontrast: view %lu completed solar-disk probes, "
                "%lu targeted rays\n",
                static_cast<unsigned long>(view_index+1),
                static_cast<unsigned long>(proposal_pending.size()));
        trace_proposals(
            backend, modifiers, view, trees, proposal_pending);
    }
    for (std::size_t time = 0; time < suns.size(); ++time)
        if (!trees[time].cells.empty())
            seed_finest_cells(trees[time], dimension);
    for (std::size_t time = 0; time < suns.size(); ++time)
        if (!trees[time].cells.empty())
            apply_finest_proposal_weights(trees[time]);

    std::vector<double> contrast(suns.size(), 0.0);
    for (std::size_t time = 0; time < suns.size(); ++time)
        if (!trees[time].cells.empty())
            contrast[time] = integrate_tree(
                trees[time], dimension, view, options);
    int stable_passes = 0;
    for (int level = 0; level < options.adaptive_levels; ++level) {
        pending.clear();
        std::size_t parents = 0;
        double maximum_estimated_uncertainty = 0.0;
        double maximum_residual_uncertainty = 0.0;
        for (std::size_t time = 0; time < suns.size(); ++time) {
            if (trees[time].cells.empty())
                continue;
            double estimated_uncertainty = 0.0;
            double residual_uncertainty = 0.0;
            const std::vector<unsigned char> refine = select_refinement(
                trees[time], dimension, view, contrast[time], options,
                &estimated_uncertainty, &residual_uncertainty);
            const double uncertainty_scale = std::max(
                std::max(std::fabs(contrast[time]), estimated_uncertainty),
                kEpsilon);
            maximum_estimated_uncertainty = std::max(
                maximum_estimated_uncertainty,
                estimated_uncertainty/uncertainty_scale);
            maximum_residual_uncertainty = std::max(
                maximum_residual_uncertainty,
                residual_uncertainty/uncertainty_scale);
            parents += refine_tree(trees[time], refine, dimension, view,
                                   right, up, time, pending);
        }
        if (pending.empty())
            break;
        for (std::size_t time = 0; time < suns.size(); ++time)
            if (!trees[time].cells.empty())
                seed_finest_cells(trees[time], dimension);
        if (!options.quiet)
            std::fprintf(stderr,
                "directlobecontrast: view %lu level %d, %lu parents, "
                "%lu targeted rays, angular uncertainty %.6g -> %.6g\n",
                static_cast<unsigned long>(view_index+1), level+1,
                static_cast<unsigned long>(parents),
                static_cast<unsigned long>(pending.size()),
                maximum_estimated_uncertainty,
                maximum_residual_uncertainty);
        trace_pending(backend, modifiers, view, trees, pending);
        for (std::size_t time = 0; time < suns.size(); ++time)
            if (!trees[time].cells.empty())
                apply_finest_proposal_weights(trees[time]);

        std::vector<double> next(suns.size(), 0.0);
        double maximum_change = 0.0;
        bool unresolved = false;
        for (std::size_t time = 0; time < suns.size(); ++time) {
            if (trees[time].cells.empty())
                continue;
            next[time] = integrate_tree(
                trees[time], dimension, view, options);
            const double scale = std::max(
                std::fabs(next[time]), 1.0e-12);
            maximum_change = std::max(maximum_change,
                std::fabs(next[time]-contrast[time])/scale);
            unresolved = unresolved || has_unresolved_proposal(
                trees[time], dimension, options);
        }
        contrast.swap(next);
        if (!options.quiet)
            std::fprintf(stderr,
                "directlobecontrast: view %lu level %d maximum contrast "
                "change %.6g%s\n",
                static_cast<unsigned long>(view_index+1), level+1,
                maximum_change, unresolved ? " (proposal unresolved)" : "");
        if (level+1 >= options.min_adaptive_levels && !unresolved &&
                maximum_change < options.contrast_tolerance)
            ++stable_passes;
        else
            stable_passes = 0;
        if (stable_passes >= options.convergence_passes)
            break;
    }
    return contrast;
}

std::vector<double> evaluate_view_batch(
    DirectLobeBackend &backend,
    const std::vector<std::string> &modifiers,
    const std::vector<Sun> &suns,
    const std::vector<std::vector<DirectLobeProposal> > &proposal_sets,
    const std::vector<Viewpoint> &views,
    const std::vector<std::size_t> &view_indices,
    const Options &options)
{
    if (view_indices.empty())
        return std::vector<double>();

    const int scale = 1 << options.adaptive_levels;
    const int dimension = options.resolution*scale;
    const std::size_t initial_count =
        static_cast<std::size_t>(options.resolution)*options.resolution;
    std::vector<ViewLobeState> states(view_indices.size());
    std::vector<BatchedPendingCell> pending;
    std::vector<BatchedPendingProposal> proposal_pending;

    for (std::size_t state_index = 0;
            state_index < states.size(); ++state_index) {
        ViewLobeState &state = states[state_index];
        state.view_index = view_indices[state_index];
        state.view = views[state.view_index];
        view_basis(state.view.direction, options.up, state.right, state.up);
        state.trees.resize(suns.size());
        state.contrast.assign(suns.size(), 0.0);

        for (std::size_t time = 0; time < suns.size(); ++time) {
            if (!suns[time].active)
                continue;
            TimeTree &tree = state.trees[time];
            for (std::size_t proposal = 0;
                    proposal < proposal_sets[time].size(); ++proposal) {
                const DirectLobeProposal &base =
                    proposal_sets[time][proposal];
                const unsigned int non_transmission = base.kinds &
                    ~static_cast<unsigned int>(DLP_TRANSMISSION);
                if (non_transmission &&
                        dot(base.direction, state.view.direction) > 0.0) {
                    DirectLobeProposal retained = base;
                    retained.kinds = non_transmission;
                    tree.proposals.push_back(retained);
                    tree.proposal_luminance.push_back(0.0);
                    BatchedPendingProposal probe;
                    probe.view = state_index;
                    probe.time = time;
                    probe.proposal = tree.proposals.size()-1;
                    proposal_pending.push_back(probe);
                }
                if (!(base.kinds & DLP_TRANSMISSION))
                    continue;
                const int pilot_resolution =
                    options.lobe_sun_disk_pilot_resolution > 0 &&
                    options.lobe_sun_disk_pilot_resolution <
                        options.lobe_sun_disk_resolution ?
                    options.lobe_sun_disk_pilot_resolution :
                    options.lobe_sun_disk_resolution;
                append_transmission_disk_probes_batch(
                    tree, base, suns[time], state.view, pilot_resolution,
                    state_index, time, proposal_pending);
            }

            tree.cells.reserve(initial_count);
            for (int row = 0; row < options.resolution; ++row)
                for (int column = 0; column < options.resolution; ++column) {
                    Cell cell;
                    cell.x0 = column*scale;
                    cell.y0 = row*scale;
                    cell.span = scale;
                    cell.direction = sample_direction(
                        (cell.x0+0.5*cell.span)/dimension,
                        (cell.y0+0.5*cell.span)/dimension,
                        state.view.direction, state.right, state.up);
                    cell.luminance = 0.0;
                    cell.solid_angle_override = 0.0;
                    tree.cells.push_back(cell);
                    BatchedPendingCell entry;
                    entry.view = state_index;
                    entry.time = time;
                    entry.cell = tree.cells.size()-1;
                    pending.push_back(entry);
                }
        }
    }

    if (!options.quiet)
        std::fprintf(stderr,
            "directlobecontrast: views %lu-%lu proposal pass, "
            "%lu targeted rays\n",
            static_cast<unsigned long>(states.front().view_index+1),
            static_cast<unsigned long>(states.back().view_index+1),
            static_cast<unsigned long>(proposal_pending.size()));
    if (!options.quiet)
        std::fprintf(stderr,
            "directlobecontrast: views %lu-%lu initial grids, "
            "%lu targeted rays\n",
            static_cast<unsigned long>(states.front().view_index+1),
            static_cast<unsigned long>(states.back().view_index+1),
            static_cast<unsigned long>(pending.size()));
    trace_proposals_and_pending_batch(
        backend, modifiers, states, proposal_pending, pending);

    if (options.lobe_sun_disk_pilot_resolution > 0 &&
            options.lobe_sun_disk_pilot_resolution <
                options.lobe_sun_disk_resolution) {
        proposal_pending.clear();
        for (std::size_t state_index = 0;
                state_index < states.size(); ++state_index) {
            ViewLobeState &state = states[state_index];
            const std::vector<unsigned char> complete =
                select_full_transmission_disks(
                    state.trees, suns, options);
            for (std::size_t time = 0; time < suns.size(); ++time) {
                TimeTree &tree = state.trees[time];
                if (!complete[time])
                    continue;
                for (std::size_t proposal = 0;
                        proposal < proposal_sets[time].size(); ++proposal)
                    if (proposal_sets[time][proposal].kinds &
                            DLP_TRANSMISSION)
                        append_transmission_disk_probes_batch(
                            tree, proposal_sets[time][proposal], suns[time],
                            state.view, options.lobe_sun_disk_resolution,
                            state_index, time, proposal_pending);
            }
        }
        if (!options.quiet)
            std::fprintf(stderr,
                "directlobecontrast: views %lu-%lu completed solar-disk "
                "probes, %lu targeted rays\n",
                static_cast<unsigned long>(states.front().view_index+1),
                static_cast<unsigned long>(states.back().view_index+1),
                static_cast<unsigned long>(proposal_pending.size()));
        trace_proposals_batch(
            backend, modifiers, states, proposal_pending);
    }
    for (std::size_t state_index = 0;
            state_index < states.size(); ++state_index)
        for (std::size_t time = 0; time < suns.size(); ++time)
            if (!states[state_index].trees[time].cells.empty())
                seed_finest_cells(
                    states[state_index].trees[time], dimension);
    for (std::size_t state_index = 0;
            state_index < states.size(); ++state_index) {
        ViewLobeState &state = states[state_index];
        for (std::size_t time = 0; time < suns.size(); ++time)
            if (!state.trees[time].cells.empty()) {
                apply_finest_proposal_weights(state.trees[time]);
                state.contrast[time] = integrate_tree(
                    state.trees[time], dimension, state.view, options);
            }
    }

    for (int level = 0; level < options.adaptive_levels; ++level) {
        pending.clear();
        std::vector<unsigned char> has_work(states.size(), 0u);
        std::size_t total_parents = 0;
        double maximum_estimated_uncertainty = 0.0;
        double maximum_residual_uncertainty = 0.0;

        for (std::size_t state_index = 0;
                state_index < states.size(); ++state_index) {
            ViewLobeState &state = states[state_index];
            if (state.finished)
                continue;
            std::vector<PendingCell> state_pending;
            std::size_t parents = 0;
            for (std::size_t time = 0; time < suns.size(); ++time) {
                if (state.trees[time].cells.empty())
                    continue;
                double estimated_uncertainty = 0.0;
                double residual_uncertainty = 0.0;
                const std::vector<unsigned char> refine = select_refinement(
                    state.trees[time], dimension, state.view,
                    state.contrast[time], options,
                    &estimated_uncertainty, &residual_uncertainty);
                const double uncertainty_scale = std::max(
                    std::max(std::fabs(state.contrast[time]),
                             estimated_uncertainty), kEpsilon);
                maximum_estimated_uncertainty = std::max(
                    maximum_estimated_uncertainty,
                    estimated_uncertainty/uncertainty_scale);
                maximum_residual_uncertainty = std::max(
                    maximum_residual_uncertainty,
                    residual_uncertainty/uncertainty_scale);
                parents += refine_tree(
                    state.trees[time], refine, dimension, state.view,
                    state.right, state.up, time, state_pending);
            }
            if (state_pending.empty()) {
                state.finished = true;
                continue;
            }
            has_work[state_index] = 1u;
            total_parents += parents;
            for (std::size_t i = 0; i < state_pending.size(); ++i) {
                BatchedPendingCell entry;
                entry.view = state_index;
                entry.time = state_pending[i].time;
                entry.cell = state_pending[i].cell;
                pending.push_back(entry);
            }
        }
        if (pending.empty())
            break;

        for (std::size_t state_index = 0;
                state_index < states.size(); ++state_index)
            if (has_work[state_index])
                for (std::size_t time = 0; time < suns.size(); ++time)
                    if (!states[state_index].trees[time].cells.empty())
                        seed_finest_cells(
                            states[state_index].trees[time], dimension);
        if (!options.quiet)
            std::fprintf(stderr,
                "directlobecontrast: views %lu-%lu level %d, "
                "%lu parents, %lu targeted rays, angular uncertainty "
                "%.6g -> %.6g\n",
                static_cast<unsigned long>(states.front().view_index+1),
                static_cast<unsigned long>(states.back().view_index+1),
                level+1, static_cast<unsigned long>(total_parents),
                static_cast<unsigned long>(pending.size()),
                maximum_estimated_uncertainty,
                maximum_residual_uncertainty);
        trace_pending_batch(backend, modifiers, states, pending);

        for (std::size_t state_index = 0;
                state_index < states.size(); ++state_index) {
            if (!has_work[state_index])
                continue;
            ViewLobeState &state = states[state_index];
            for (std::size_t time = 0; time < suns.size(); ++time)
                if (!state.trees[time].cells.empty())
                    apply_finest_proposal_weights(state.trees[time]);

            std::vector<double> next(suns.size(), 0.0);
            double maximum_change = 0.0;
            bool unresolved = false;
            for (std::size_t time = 0; time < suns.size(); ++time) {
                if (state.trees[time].cells.empty())
                    continue;
                next[time] = integrate_tree(
                    state.trees[time], dimension, state.view, options);
                const double contrast_scale = std::max(
                    std::fabs(next[time]), 1.0e-12);
                maximum_change = std::max(maximum_change,
                    std::fabs(next[time]-state.contrast[time])/
                        contrast_scale);
                unresolved = unresolved || has_unresolved_proposal(
                    state.trees[time], dimension, options);
            }
            state.contrast.swap(next);
            if (level+1 >= options.min_adaptive_levels && !unresolved &&
                    maximum_change < options.contrast_tolerance)
                ++state.stable_passes;
            else
                state.stable_passes = 0;
            if (state.stable_passes >= options.convergence_passes)
                state.finished = true;
        }
    }

    std::vector<double> result(states.size()*suns.size(), 0.0);
    for (std::size_t state_index = 0;
            state_index < states.size(); ++state_index)
        std::copy(states[state_index].contrast.begin(),
                  states[state_index].contrast.end(),
                  result.begin()+state_index*suns.size());
    return result;
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

std::string require_value(int &index, int argc, char **argv,
                          const std::string &option)
{
    if (++index >= argc)
        throw std::runtime_error("missing argument for " + option);
    return argv[index];
}

void usage(FILE *stream, const char *program)
{
    std::fprintf(stream,
        "Usage: %s -vf views.pts -S suns.rad [options] scene.oct\n\n"
        "Options:\n"
        "  -h                         suppress Radiance matrix header\n"
        "  -q                         suppress progress messages\n"
        "  -r count                   base Shirley-Chiu width (default 4)\n"
        "  --adaptive-levels count    quadtree levels (default 6)\n"
        "  --adaptive-trigger value   luminance refinement trigger (default 500)\n"
        "  --adaptive-gradient value  neighbor-gradient trigger (default 500)\n"
        "  --adaptive-variance value  sibling-cell deviation trigger (default 500)\n"
        "  --adaptive-uncertainty v   unresolved relative DGP-contrast budget (default 0.01)\n"
        "  --adaptive-guard count     neighboring refinement rings (default 1)\n"
        "  --contrast-tolerance value relative G convergence (default 0.01)\n"
        "  --convergence-passes count stable levels required (default 2)\n"
        "  --min-adaptive-levels n    levels before convergence (default 2)\n"
        "  --normal-tolerance value   normal deduplication degrees (default 0.1)\n"
        "  -c count                   repeated source samples (default 8)\n"
        "  -t value                   glare threshold in cd/m2 (default 2000)\n"
        "  --min-cluster-cells count  minimum connected lobe size (default 1)\n"
        "  --max-lobe-bounces n       Radiance recursion limit (default 2)\n"
        "  --lobe-samples n           Radiance rough-lobe samples (default 16)\n"
        "  --sun-disk-resolution n    base solar-disk width (default 2)\n"
        "  --sun-disk-levels n        maximum solar-disk levels (default 6)\n"
        "  --sun-disk-min-levels n    mandatory solar-disk levels (default 3)\n"
        "  --sun-disk-gradient value  solar luminance-gradient trigger (default 500)\n"
        "  --sun-disk-guard n         solar boundary guard rings (default 1)\n"
        "  --sun-disk-tolerance value relative solar G convergence (default 0.005)\n"
        "  --lobe-sun-disk-resolution n\n"
        "                             transmission seed grid width (default 8)\n"
        "  --lobe-sun-disk-pilot-resolution n\n"
        "                             adaptive pilot width; 0 traces the full grid (default 0)\n"
        "  --lobe-sun-disk-pilot-guard-angle degrees\n"
        "                             complete nearby solar positions (default 1)\n"
        "  --sun-mode mode            batch or adaptive (default batch)\n"
        "  --sun-samples n            batch irradiance samples (default 10000)\n"
        "  --view-batch-size n        lobe viewpoints per trace batch (default 1)\n"
        "  --sun-only                 calculate only directly visible sun\n"
        "  --lobe-only                calculate only transmission/refraction lobes\n"
        "  --sun-output file          write direct-sun component matrix\n"
        "  --sun-irradiance-output f  write batch direct-sun RGB irradiance\n"
        "  --sun-total-irradiance-output f\n"
        "                             include mirror virtual-source irradiance\n"
        "  --lobe-output file         write direct-lobe component matrix\n"
        "  --lobe-view-mask file      one 0/1 entry per viewpoint; 0 skips lobe sampling\n"
        "  --render-options \"...\"   additional built-in Radiance options\n"
        "  -vu x y z                  preferred view-up vector (default 0 0 1)\n"
        "  -n count                   built-in Radiance workers (default 1)\n"
        "  -o file                    output matrix (default stdout)\n\n"
        "Rows are viewpoints and columns follow source records in suns.rad.\n"
        "The default output is the sum of directly visible sun and clustered "
        "non-diffuse transmission/refraction lobes. Mirror paths and XML "
        "BSDF peaks are excluded.\n",
        program);
}

Options parse_options(int argc, char **argv)
{
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "-?" || argument == "--help") {
            usage(stdout, argv[0]);
            std::exit(0);
        } else if (argument == "-h")
            options.no_header = true;
        else if (argument == "-q")
            options.quiet = true;
        else if (argument == "-vf")
            options.views_path = require_value(index, argc, argv, argument);
        else if (argument == "-S")
            options.suns_path = require_value(index, argc, argv, argument);
        else if (argument == "-r")
            options.resolution = parse_int(
                require_value(index, argc, argv, argument), "base resolution");
        else if (argument == "--adaptive-levels")
            options.adaptive_levels = parse_int(
                require_value(index, argc, argv, argument), "adaptive levels");
        else if (argument == "--adaptive-trigger")
            options.adaptive_trigger = parse_nonnegative(
                require_value(index, argc, argv, argument), "adaptive trigger");
        else if (argument == "--adaptive-gradient")
            options.adaptive_gradient = parse_nonnegative(
                require_value(index, argc, argv, argument), "gradient trigger");
        else if (argument == "--adaptive-variance")
            options.adaptive_variance = parse_nonnegative(
                require_value(index, argc, argv, argument), "variance trigger");
        else if (argument == "--adaptive-uncertainty")
            options.adaptive_uncertainty = parse_nonnegative(
                require_value(index, argc, argv, argument),
                "adaptive uncertainty");
        else if (argument == "--adaptive-guard")
            options.adaptive_guard = parse_int(
                require_value(index, argc, argv, argument), "adaptive guard");
        else if (argument == "--contrast-tolerance")
            options.contrast_tolerance = parse_nonnegative(
                require_value(index, argc, argv, argument),
                "contrast tolerance");
        else if (argument == "--convergence-passes")
            options.convergence_passes = parse_int(
                require_value(index, argc, argv, argument),
                "convergence passes");
        else if (argument == "--min-adaptive-levels")
            options.min_adaptive_levels = parse_int(
                require_value(index, argc, argv, argument),
                "minimum adaptive levels");
        else if (argument == "--normal-tolerance")
            options.proposal_normal_tolerance = parse_nonnegative(
                require_value(index, argc, argv, argument),
                "normal tolerance");
        else if (argument == "-c")
            options.accumulation = parse_int(
                require_value(index, argc, argv, argument), "sample count");
        else if (argument == "-t")
            options.threshold = parse_nonnegative(
                require_value(index, argc, argv, argument), "glare threshold");
        else if (argument == "--min-cluster-cells")
            options.min_cluster_cells = parse_int(
                require_value(index, argc, argv, argument), "cluster size");
        else if (argument == "--max-lobe-bounces" ||
                argument == "--max-specular-bounces")
            options.max_lobe_bounces = parse_int(
                require_value(index, argc, argv, argument), "recursion limit");
        else if (argument == "--lobe-samples" ||
                argument == "--rough-samples")
            options.lobe_samples = parse_int(
                require_value(index, argc, argv, argument), "lobe samples");
        else if (argument == "--sun-disk-resolution")
            options.sun_disk_resolution = parse_int(
                require_value(index, argc, argv, argument),
                "solar-disk resolution");
        else if (argument == "--sun-disk-levels")
            options.sun_disk_levels = parse_int(
                require_value(index, argc, argv, argument),
                "solar-disk levels");
        else if (argument == "--sun-disk-min-levels")
            options.sun_disk_min_levels = parse_int(
                require_value(index, argc, argv, argument),
                "minimum solar-disk levels");
        else if (argument == "--sun-disk-gradient")
            options.sun_disk_gradient = parse_nonnegative(
                require_value(index, argc, argv, argument),
                "solar-disk gradient");
        else if (argument == "--sun-disk-guard")
            options.sun_disk_guard = parse_int(
                require_value(index, argc, argv, argument),
                "solar-disk guard");
        else if (argument == "--sun-disk-tolerance")
            options.sun_disk_tolerance = parse_nonnegative(
                require_value(index, argc, argv, argument),
                "solar-disk tolerance");
        else if (argument == "--lobe-sun-disk-resolution")
            options.lobe_sun_disk_resolution = parse_int(
                require_value(index, argc, argv, argument),
                "transmission seed grid width");
        else if (argument == "--lobe-sun-disk-pilot-resolution")
            options.lobe_sun_disk_pilot_resolution = parse_int(
                require_value(index, argc, argv, argument),
                "transmission pilot grid width");
        else if (argument == "--lobe-sun-disk-pilot-guard-angle")
            options.lobe_sun_disk_pilot_guard_angle = parse_nonnegative(
                require_value(index, argc, argv, argument),
                "transmission pilot guard angle");
        else if (argument == "--sun-mode") {
            const std::string mode =
                require_value(index, argc, argv, argument);
            if (mode == "batch")
                options.sun_mode = SUN_BATCH;
            else if (mode == "adaptive")
                options.sun_mode = SUN_ADAPTIVE;
            else
                throw std::runtime_error(
                    "sun mode must be 'batch' or 'adaptive'");
        }
        else if (argument == "--sun-samples")
            options.sun_samples = parse_int(
                require_value(index, argc, argv, argument),
                "direct-sun samples");
        else if (argument == "--view-batch-size")
            options.view_batch_size = parse_int(
                require_value(index, argc, argv, argument),
                "view batch size");
        else if (argument == "--sun-only") {
            options.include_sun = true;
            options.include_lobes = false;
        }
        else if (argument == "--lobe-only") {
            options.include_sun = false;
            options.include_lobes = true;
        }
        else if (argument == "--sun-output")
            options.sun_output_path =
                require_value(index, argc, argv, argument);
        else if (argument == "--sun-irradiance-output")
            options.sun_irradiance_output_path =
                require_value(index, argc, argv, argument);
        else if (argument == "--sun-total-irradiance-output")
            options.sun_total_irradiance_output_path =
                require_value(index, argc, argv, argument);
        else if (argument == "--lobe-output")
            options.lobe_output_path =
                require_value(index, argc, argv, argument);
        else if (argument == "--lobe-view-mask")
            options.lobe_view_mask_path =
                require_value(index, argc, argv, argument);
        else if (argument == "--render-options") {
            const std::vector<std::string> extra = split_words(
                require_value(index, argc, argv, argument));
            options.render_options.insert(options.render_options.end(),
                                          extra.begin(), extra.end());
        }
        else if (argument == "-vu") {
            if (index+3 >= argc)
                throw std::runtime_error("missing values for -vu");
            for (int component = 0; component < 3; ++component)
                if (!parse_double(argv[++index], options.up[component]))
                    throw std::runtime_error("invalid -vu vector");
        } else if (argument == "-n")
            options.nproc = parse_int(
                require_value(index, argc, argv, argument), "worker count");
        else if (argument == "-o")
            options.output_path = require_value(index, argc, argv, argument);
        else if (!argument.empty() && argument[0] == '-')
            throw std::runtime_error("unknown option '" + argument + "'");
        else if (options.octree.empty())
            options.octree = argument;
        else
            throw std::runtime_error("unexpected argument '" + argument + "'");
    }
    if (options.views_path.empty() || options.suns_path.empty() ||
            options.octree.empty())
        throw std::runtime_error("-vf, -S, and a scene octree are required");
    if (options.resolution < 2 || options.adaptive_levels < 0 ||
            options.adaptive_levels > 10 || options.adaptive_guard < 0 ||
            options.accumulation < 1 || options.min_cluster_cells < 1 ||
            options.max_lobe_bounces < 1 || options.lobe_samples < 0 ||
            options.sun_disk_resolution < 1 ||
            options.sun_disk_levels < 0 || options.sun_disk_levels > 10 ||
            options.sun_disk_min_levels < 0 ||
            options.sun_disk_min_levels > options.sun_disk_levels ||
            options.sun_disk_guard < 0 ||
            options.sun_disk_tolerance <= 0.0 ||
            options.lobe_sun_disk_resolution < 1 ||
            options.lobe_sun_disk_resolution > 64 ||
            options.lobe_sun_disk_pilot_resolution < 0 ||
            options.lobe_sun_disk_pilot_resolution > 64 ||
            options.lobe_sun_disk_pilot_guard_angle > 90.0 ||
            options.sun_samples < 1 ||
            options.view_batch_size < 1 ||
            options.nproc < 1 ||
            options.min_adaptive_levels < 0 ||
            options.min_adaptive_levels > options.adaptive_levels ||
            options.convergence_passes < 1 ||
            options.adaptive_uncertainty <= 0.0 ||
            options.adaptive_uncertainty > 1.0 ||
            options.contrast_tolerance <= 0.0 ||
            options.proposal_normal_tolerance <= 0.0 ||
            options.proposal_normal_tolerance > 90.0)
        throw std::runtime_error("invalid numeric option");
    if (static_cast<long long>(options.resolution)*
            (1LL << options.adaptive_levels) > 4096)
        throw std::runtime_error("maximum adaptive grid exceeds 4096 x 4096");
    if (static_cast<long long>(options.sun_disk_resolution)*
            (1LL << options.sun_disk_levels) > 4096)
        throw std::runtime_error(
            "maximum solar-disk grid exceeds 4096 x 4096");
    if ((!options.sun_output_path.empty() && !options.include_sun) ||
            (!options.sun_irradiance_output_path.empty() &&
                !options.include_sun) ||
            (!options.sun_total_irradiance_output_path.empty() &&
                !options.include_sun) ||
            (!options.lobe_output_path.empty() && !options.include_lobes))
        throw std::runtime_error(
            "component output requested for a disabled calculation");
    if ((!options.sun_irradiance_output_path.empty() ||
            !options.sun_total_irradiance_output_path.empty()) &&
            options.sun_mode != SUN_BATCH)
        throw std::runtime_error(
            "sun irradiance output requires --sun-mode batch");
    std::set<std::string> output_paths;
    if (!options.output_path.empty())
        output_paths.insert(options.output_path);
    if (!options.sun_output_path.empty() &&
            !output_paths.insert(options.sun_output_path).second)
        throw std::runtime_error("output matrix paths must be distinct");
    if (!options.sun_irradiance_output_path.empty() &&
            !output_paths.insert(
                options.sun_irradiance_output_path).second)
        throw std::runtime_error("output matrix paths must be distinct");
    if (!options.sun_total_irradiance_output_path.empty() &&
            !output_paths.insert(
                options.sun_total_irradiance_output_path).second)
        throw std::runtime_error("output matrix paths must be distinct");
    if (!options.lobe_output_path.empty() &&
            !output_paths.insert(options.lobe_output_path).second)
        throw std::runtime_error("output matrix paths must be distinct");
    options.up = normalized(options.up, "view-up vector");
    const char *forbidden[] = {"-ab", "-lr", "-ss", "-c", "-n", "-M",
                               "-m", "--path-output", "--path-components",
                               "--direct-specular-only"};
    for (std::size_t i = 0; i < options.render_options.size(); ++i)
        for (std::size_t j = 0; j < sizeof(forbidden)/sizeof(forbidden[0]); ++j)
            if (options.render_options[i] == forbidden[j])
                throw std::runtime_error("render option '"+
                    options.render_options[i]+"' is managed internally");
    options.render_options.push_back("-ab");
    options.render_options.push_back("0");
    options.render_options.push_back("-lr");
    options.render_options.push_back(std::to_string(options.max_lobe_bounces));
    options.render_options.push_back("-ss");
    options.render_options.push_back(std::to_string(options.lobe_samples));
    return options;
}

enum OutputComponent {
    OUTPUT_TOTAL,
    OUTPUT_DIRECT_SUN,
    OUTPUT_DIRECT_LOBE
};

void write_matrix(const Options &options, const std::string &path,
                  OutputComponent component,
                  const std::vector<double> &contrast,
                  std::size_t nviews, std::size_t nsteps,
                  int argc, char **argv)
{
    FILE *output = stdout;
    if (!path.empty()) {
        output = std::fopen(path.c_str(), "w");
        if (!output)
            throw std::runtime_error("cannot open output file '"+
                                     path+"'");
    }
    if (!options.no_header) {
        const char *sun_method = options.sun_mode == SUN_BATCH ?
            "batch_irradiance" : "adaptive_solar_disk";
        const char *sun_filter = options.sun_mode == SUN_BATCH ?
            "DIRECT_SOURCE_SAMPLING_AB0" : "N_DIFFUSE_0_N_REDIRECT_0";
        newheader("RADIANCE", output);
        printargs(argc, argv, output);
        fputnow(output);
        std::fprintf(output, "NROWS=%lu\nNCOLS=%lu\nNCOMP=1\n",
            static_cast<unsigned long>(nviews),
            static_cast<unsigned long>(nsteps));
        if (component == OUTPUT_TOTAL)
            std::fprintf(output,
                "PATH_REPLACEMENT=direct_visible_sun_plus_direct_transmission_refraction_lobes\n"
                "PATH_FILTER=UNION(%s,N_DIFFUSE_0_N_DIRECT_LOBE_GE_1_N_MIRROR_0_N_BSDF_0)\n"
                "DIRECT_VISIBLE_SUN=included_%s\n"
                "DIRECT_LOBES=included_adaptive_view_hemisphere\n",
                sun_filter, sun_method);
        else if (component == OUTPUT_DIRECT_SUN)
            std::fprintf(output,
                "PATH_REPLACEMENT=direct_visible_sun\n"
                "PATH_FILTER=%s\n"
                "DIRECT_VISIBLE_SUN=included_%s\n"
                "DIRECT_LOBES=excluded\n", sun_filter, sun_method);
        else
            std::fprintf(output,
                "PATH_REPLACEMENT=direct_transmission_refraction_lobes\n"
                "PATH_FILTER=N_DIFFUSE_0_N_DIRECT_LOBE_GE_1_N_MIRROR_0_N_BSDF_0\n"
                "DIRECT_VISIBLE_SUN=excluded\n"
                "DIRECT_LOBES=included_adaptive_view_hemisphere\n");
        std::fprintf(output,
            "PURE_MIRROR_PATHS=excluded_use_specularcontrast\n"
            "MIXED_DIRECT_LOBE_MIRROR_PATHS=excluded\n"
            "BSDF_PEAKS=excluded_use_ttsuncontrast\n"
            "TRANSMISSION_PROPOSAL_SOLAR_DISK=%dx%d\n"
            "TRANSMISSION_PROPOSAL_PILOT=%dx%d\n"
            "TRANSMISSION_PROPOSAL_PILOT_GUARD_DEG=%.9g\n"
            "ADAPTIVE_DGP_UNCERTAINTY=%.9g\n"
            "VIEW_BATCH_SIZE=%d\n",
            options.lobe_sun_disk_resolution,
            options.lobe_sun_disk_resolution,
            options.lobe_sun_disk_pilot_resolution,
            options.lobe_sun_disk_pilot_resolution,
            options.lobe_sun_disk_pilot_guard_angle,
            options.adaptive_uncertainty,
            options.view_batch_size);
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

void write_irradiance_matrix(const Options &options, const std::string &path,
                             const std::vector<float> &irradiance,
                             bool includes_virtual_sources,
                             std::size_t nviews, std::size_t nsteps,
                             int argc, char **argv)
{
    FILE *output = std::fopen(path.c_str(), "w");
    if (!output)
        throw std::runtime_error("cannot open output file '"+path+"'");
    if (!options.no_header) {
        newheader("RADIANCE", output);
        printargs(argc, argv, output);
        fputnow(output);
        std::fprintf(output,
            "NROWS=%lu\nNCOLS=%lu\nNCOMP=3\n"
            "QUANTITY=%s\n"
            "UNITS=Radiance_RGB_W_per_m2\n"
            "PATH_FILTER=%s\n",
            static_cast<unsigned long>(nviews),
            static_cast<unsigned long>(nsteps),
            includes_virtual_sources ?
                "total_direct_source_irradiance" :
                "physical_direct_solar_irradiance",
            includes_virtual_sources ?
                "DIRECT_SOURCE_SAMPLING_AB0_PHYSICAL_PLUS_VIRTUAL" :
                "DIRECT_SOURCE_SAMPLING_AB0_ZERO_REDIRECT");
        fputformat("ascii", output);
        std::fputc('\n', output);
    }
    for (std::size_t view = 0; view < nviews; ++view)
        for (std::size_t time = 0; time < nsteps; ++time) {
            const std::size_t index = 3*(view*nsteps+time);
            std::fprintf(output, "%.9e\t%.9e\t%.9e%c",
                irradiance[index], irradiance[index+1], irradiance[index+2],
                time+1 == nsteps ? '\n' : '\t');
        }
    if (std::fflush(output) == EOF || std::fclose(output) == EOF)
        throw std::runtime_error("error writing output matrix");
}

} // namespace

int main(int argc, char **argv)
{
    progname = fixargv0(argv[0]);
    try {
        const Options options = parse_options(argc, argv);
        const std::vector<Viewpoint> views = load_views(options.views_path);
        const std::vector<unsigned char> lobe_view_mask =
            load_lobe_view_mask(options.lobe_view_mask_path, views.size());
        const std::vector<Sun> suns = load_suns(options.suns_path);
        std::vector<std::string> modifiers;
        modifiers.reserve(suns.size());
        std::size_t active = 0;
        for (std::size_t time = 0; time < suns.size(); ++time) {
            modifiers.push_back(suns[time].modifier);
            active += suns[time].active ? 1u : 0u;
        }
        if (!options.quiet)
            std::fprintf(stderr,
                "directlobecontrast: %lu views, %lu solar records "
                "(%lu active), built-in Radiance backend\n",
                static_cast<unsigned long>(views.size()),
                static_cast<unsigned long>(suns.size()),
                static_cast<unsigned long>(active));

        DirectLobeBackend backend(options.nproc, options.accumulation,
                                  options.render_options);
        backend.load_scene(options.octree);
        std::unique_ptr<DirectLobeProposalGenerator> proposal_generator;
        if (options.include_lobes) {
            DirectLobeProposalOptions proposal_options;
            proposal_options.normal_tolerance_degrees =
                options.proposal_normal_tolerance;
            proposal_generator.reset(new DirectLobeProposalGenerator(
                options.octree, proposal_options));
        }
        if (!options.quiet && proposal_generator.get()) {
            std::fprintf(stderr,
                "directlobecontrast: proposal catalog: %lu rough "
                "transmission surfaces, %lu refractive normals, "
                "%lu prism surfaces\n",
                static_cast<unsigned long>(
                    proposal_generator->rough_transmission_surface_count()),
                static_cast<unsigned long>(
                    proposal_generator->refractive_normal_count()),
                static_cast<unsigned long>(
                    proposal_generator->prism_surface_count()));
            if (proposal_generator->skipped_instance_count() ||
                    proposal_generator->skipped_mesh_count())
                std::fprintf(stderr,
                    "directlobecontrast: warning: proposal scan skipped "
                    "%lu instances and %lu meshes; use an expanded -f octree "
                    "when these contain redirecting materials\n",
                    static_cast<unsigned long>(
                        proposal_generator->skipped_instance_count()),
                    static_cast<unsigned long>(
                        proposal_generator->skipped_mesh_count()));
        }
        bool evaluate_lobes = options.include_lobes;
        if (proposal_generator.get() &&
                !proposal_generator->rough_transmission_surface_count() &&
                !proposal_generator->refractive_normal_count() &&
                !proposal_generator->prism_surface_count() &&
                !proposal_generator->skipped_instance_count() &&
                !proposal_generator->skipped_mesh_count()) {
            evaluate_lobes = false;
            if (!options.quiet)
                std::fprintf(stderr,
                    "directlobecontrast: no transmission/refraction lobe "
                    "candidates; skipping adaptive view sampling\n");
        }
        std::vector<double> sun_contrast(
            views.size()*suns.size(), 0.0);
        std::vector<float> sun_irradiance;
        std::vector<float> sun_total_irradiance;
        std::vector<double> lobe_contrast(
            views.size()*suns.size(), 0.0);
        std::vector<double> contrast(
            views.size()*suns.size(), 0.0);
        std::vector<std::vector<DirectLobeProposal> > proposal_sets(
            suns.size());
        if (proposal_generator.get())
            for (std::size_t time = 0; time < suns.size(); ++time)
                if (suns[time].active)
                    proposal_sets[time] = proposal_generator->generate(
                        suns[time].direction, suns[time].solid_angle);
        if (options.include_sun && options.sun_mode == SUN_BATCH)
            sun_contrast = evaluate_direct_sun_batch(
                backend, modifiers, suns, views, options,
                options.sun_irradiance_output_path.empty() ?
                    NULL : &sun_irradiance,
                options.sun_total_irradiance_output_path.empty() ?
                    NULL : &sun_total_irradiance);
        if (options.include_sun && options.sun_mode == SUN_ADAPTIVE)
            for (std::size_t view = 0; view < views.size(); ++view) {
                const std::vector<double> row = evaluate_direct_sun(
                    backend, modifiers, suns, views[view], view, options);
                std::copy(row.begin(), row.end(),
                    sun_contrast.begin()+view*suns.size());
            }

        if (evaluate_lobes) {
            std::vector<std::size_t> active_views;
            active_views.reserve(views.size());
            for (std::size_t view = 0; view < views.size(); ++view)
                if (lobe_view_mask[view])
                    active_views.push_back(view);
            if (options.view_batch_size == 1)
                for (std::size_t active = 0;
                        active < active_views.size(); ++active) {
                    const std::size_t view = active_views[active];
                    const std::vector<double> row = evaluate_view(
                        backend, modifiers, suns, proposal_sets,
                        views[view], view, options);
                    std::copy(row.begin(), row.end(),
                        lobe_contrast.begin()+view*suns.size());
                }
            else
                for (std::size_t begin = 0; begin < active_views.size();
                        begin += options.view_batch_size) {
                    const std::size_t end = std::min(
                        active_views.size(), begin+static_cast<std::size_t>(
                            options.view_batch_size));
                    const std::vector<std::size_t> batch_indices(
                        active_views.begin()+begin, active_views.begin()+end);
                    const std::vector<double> batch = evaluate_view_batch(
                        backend, modifiers, suns, proposal_sets,
                        views, batch_indices, options);
                    for (std::size_t local = 0;
                            local < batch_indices.size(); ++local)
                        std::copy(batch.begin()+local*suns.size(),
                                  batch.begin()+(local+1)*suns.size(),
                                  lobe_contrast.begin()+
                                      batch_indices[local]*suns.size());
                }
        }

        for (std::size_t view = 0; view < views.size(); ++view)
            for (std::size_t time = 0; time < suns.size(); ++time) {
                const std::size_t index = view*suns.size()+time;
                contrast[index] = sun_contrast[index]+lobe_contrast[index];
            }
        const OutputComponent primary_component =
            options.include_sun && options.include_lobes ? OUTPUT_TOTAL :
            (options.include_sun ? OUTPUT_DIRECT_SUN : OUTPUT_DIRECT_LOBE);
        write_matrix(options, options.output_path, primary_component,
                     contrast, views.size(), suns.size(), argc, argv);
        if (!options.sun_output_path.empty())
            write_matrix(options, options.sun_output_path, OUTPUT_DIRECT_SUN,
                         sun_contrast, views.size(), suns.size(), argc, argv);
        if (!options.sun_irradiance_output_path.empty())
            write_irradiance_matrix(
                options, options.sun_irradiance_output_path, sun_irradiance,
                false,
                                    views.size(), suns.size(), argc, argv);
        if (!options.sun_total_irradiance_output_path.empty())
            write_irradiance_matrix(
                options, options.sun_total_irradiance_output_path,
                sun_total_irradiance, true,
                views.size(), suns.size(), argc, argv);
        if (!options.lobe_output_path.empty())
            write_matrix(options, options.lobe_output_path,
                         OUTPUT_DIRECT_LOBE, lobe_contrast,
                         views.size(), suns.size(), argc, argv);
        if (!options.quiet)
            std::fprintf(stderr, "directlobecontrast: done\n");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s: %s\n", argc ? argv[0] :
                     "directlobecontrast", error.what());
        if (argc)
            usage(stderr, argv[0]);
        return 1;
    }
}
