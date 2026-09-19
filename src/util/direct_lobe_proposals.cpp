#include "direct_lobe_proposals.h"

#include "standard.h"
#include "object.h"
#include "face.h"
#include "otypes.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const double kEpsilon = 1.0e-12;
typedef std::array<double, 3> Vec3;

struct RefractiveNormal {
    Vec3 normal;
    double ratio;
};

double dot(const Vec3 &a, const Vec3 &b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

Vec3 scaled(const Vec3 &a, double scale)
{
    return Vec3{{a[0]*scale, a[1]*scale, a[2]*scale}};
}

Vec3 add_scaled(const Vec3 &a, const Vec3 &b, double scale)
{
    return Vec3{{a[0]+scale*b[0],
                 a[1]+scale*b[1],
                 a[2]+scale*b[2]}};
}

Vec3 normalized(const Vec3 &value, const char *label)
{
    const double length = std::sqrt(dot(value, value));
    if (!(length > kEpsilon))
        throw std::runtime_error(std::string("zero-length ")+label);
    return scaled(value, 1.0/length);
}

Vec3 canonical_normal(const Vec3 &candidate)
{
    Vec3 normal = normalized(candidate, "proposal surface normal");
    for (int component = 0; component < 3; ++component) {
        if (std::fabs(normal[component]) <= kEpsilon)
            continue;
        if (normal[component] < 0.0)
            normal = scaled(normal, -1.0);
        break;
    }
    return normal;
}

OBJREC *resolve_surface_material(OBJREC *surface)
{
    OBJECT object = surface->omod;
    int aliases = 0;
    while (object != OVOID) {
        OBJREC *record = objptr(object);
        if (record->otype == MOD_ALIAS) {
            if (++aliases > 64)
                throw std::runtime_error(
                    "modifier alias loop while scanning direct-lobe surfaces");
            object = record->oargs.nsargs ?
                lastmod(object, record->oargs.sarg[0]) : record->omod;
            continue;
        }
        aliases = 0;
        if (ismaterial(record->otype))
            return record;
        object = record->omod;
    }
    return NULL;
}

bool rough_specular_transmission(const OBJREC *material)
{
    if (!material)
        return false;
    if (material->otype == MAT_TRANS && material->oargs.nfargs >= 7)
        return material->oargs.farg[4] > kEpsilon &&
               material->oargs.farg[5] > kEpsilon &&
               material->oargs.farg[6] > kEpsilon;
    if (material->otype == MAT_TRANS2 && material->oargs.nfargs >= 8)
        return (material->oargs.farg[4] > kEpsilon ||
                material->oargs.farg[5] > kEpsilon) &&
               material->oargs.farg[6] > kEpsilon &&
               material->oargs.farg[7] > kEpsilon;
    return false;
}

bool material_refracts(const OBJREC *material, double &ratio)
{
    if (!material)
        return false;
    if (material->otype == MAT_DIELECTRIC && material->oargs.nfargs >= 4) {
        ratio = material->oargs.farg[3];
        return ratio > kEpsilon;
    }
    if (material->otype == MAT_INTERFACE && material->oargs.nfargs >= 8) {
        ratio = material->oargs.farg[3]/material->oargs.farg[7];
        return ratio > kEpsilon;
    }
    return false;
}

bool inverse_refracted_direction(const Vec3 &transmitted,
                                 const Vec3 &normal, double tangent_scale,
                                 Vec3 &incident)
{
    const double normal_component = dot(transmitted, normal);
    const Vec3 tangent = add_scaled(transmitted, normal, -normal_component);
    const Vec3 incident_tangent = scaled(tangent, tangent_scale);
    const double tangent_length2 = dot(incident_tangent, incident_tangent);
    if (tangent_length2 >= 1.0-kEpsilon)
        return false;
    const double sign = normal_component < 0.0 ? -1.0 : 1.0;
    incident = add_scaled(incident_tangent, normal,
        sign*std::sqrt(std::max(0.0, 1.0-tangent_length2)));
    incident = normalized(incident, "refracted proposal direction");
    return true;
}

void append_unique_proposal(std::vector<DirectLobeProposal> &proposals,
                            const Vec3 &candidate, double solid_angle,
                            unsigned int kind)
{
    const Vec3 direction = normalized(candidate, "direct-lobe proposal");
    const double duplicate_cosine = std::cos(1.0e-5);
    for (std::size_t i = 0; i < proposals.size(); ++i)
        if (dot(direction, proposals[i].direction) >= duplicate_cosine) {
            proposals[i].kinds |= kind;
            proposals[i].solid_angle = std::min(
                proposals[i].solid_angle, solid_angle);
            return;
        }
    DirectLobeProposal proposal;
    proposal.direction = direction;
    proposal.solid_angle = solid_angle;
    proposal.kinds = kind;
    proposals.push_back(proposal);
}

} // namespace

class DirectLobeProposalGenerator::Impl {
public:
    explicit Impl(const DirectLobeProposalOptions &options)
        : rough_transmission_surfaces_(0), prism_surfaces_(0),
          skipped_instances_(0), skipped_meshes_(0)
    {
        const double tolerance = options.normal_tolerance_degrees*PI/180.0;
        const double cosine_tolerance = std::cos(tolerance);
        for (OBJECT index = 0; index < nobjects; ++index) {
            OBJREC *surface = objptr(index);
            if (surface->otype == OBJ_INSTANCE) {
                ++skipped_instances_;
                continue;
            }
            if (surface->otype == OBJ_MESH) {
                ++skipped_meshes_;
                continue;
            }
            if (surface->otype != OBJ_FACE)
                continue;
            FACE *face = getface(surface);
            if (!face || face->nv < 3 || face->area <= kEpsilon)
                continue;
            OBJREC *material = resolve_surface_material(surface);
            if (!material)
                continue;
            if (rough_specular_transmission(material))
                ++rough_transmission_surfaces_;
            if (material->otype == MAT_DIRECT1 ||
                    material->otype == MAT_DIRECT2)
                ++prism_surfaces_;

            double ratio = 0.0;
            if (!material_refracts(material, ratio))
                continue;
            const Vec3 normal = canonical_normal(Vec3{{
                face->norm[0], face->norm[1], face->norm[2]}});
            bool duplicate = false;
            for (std::size_t entry = 0;
                    entry < refractive_normals_.size(); ++entry)
                if (std::fabs(dot(normal,
                                  refractive_normals_[entry].normal)) >=
                            cosine_tolerance &&
                        std::fabs(ratio-refractive_normals_[entry].ratio) <
                            1.0e-8) {
                    duplicate = true;
                    break;
                }
            if (!duplicate) {
                RefractiveNormal entry;
                entry.normal = normal;
                entry.ratio = ratio;
                refractive_normals_.push_back(entry);
            }
        }
    }

    std::vector<DirectLobeProposal> generate(
        const Vec3 &sun_direction, double sun_solid_angle) const
    {
        const Vec3 sun = normalized(sun_direction,
                                    "solar proposal direction");
        const double omega = std::max(sun_solid_angle, 1.0e-9);
        std::vector<DirectLobeProposal> proposals;
        if (rough_transmission_surfaces_)
            append_unique_proposal(proposals, sun, omega, DLP_TRANSMISSION);

        for (std::size_t entry = 0;
                entry < refractive_normals_.size(); ++entry) {
            const RefractiveNormal &surface = refractive_normals_[entry];
            const double ratios[2] = {surface.ratio, 1.0/surface.ratio};
            for (int orientation = -1; orientation <= 1; orientation += 2)
                for (int ratio = 0; ratio < 2; ++ratio) {
                    Vec3 candidate;
                    if (inverse_refracted_direction(
                            sun, scaled(surface.normal, orientation),
                            ratios[ratio], candidate))
                        append_unique_proposal(proposals, candidate, omega,
                                               DLP_REFRACTION);
                }
        }
        return proposals;
    }

    std::size_t rough_transmission_surfaces_;
    std::size_t prism_surfaces_;
    std::size_t skipped_instances_;
    std::size_t skipped_meshes_;
    std::vector<RefractiveNormal> refractive_normals_;
};

DirectLobeProposalGenerator::DirectLobeProposalGenerator(
    const std::string &octree_path,
    const DirectLobeProposalOptions &options)
    : impl_(new Impl(options))
{
    (void)octree_path;
}

DirectLobeProposalGenerator::~DirectLobeProposalGenerator()
{
}

std::vector<DirectLobeProposal> DirectLobeProposalGenerator::generate(
    const std::array<double, 3> &sun_direction,
    double sun_solid_angle) const
{
    return impl_->generate(sun_direction, sun_solid_angle);
}

std::size_t
DirectLobeProposalGenerator::rough_transmission_surface_count() const
{
    return impl_->rough_transmission_surfaces_;
}

std::size_t DirectLobeProposalGenerator::refractive_normal_count() const
{
    return impl_->refractive_normals_.size();
}

std::size_t DirectLobeProposalGenerator::prism_surface_count() const
{
    return impl_->prism_surfaces_;
}

std::size_t DirectLobeProposalGenerator::skipped_instance_count() const
{
    return impl_->skipped_instances_;
}

std::size_t DirectLobeProposalGenerator::skipped_mesh_count() const
{
    return impl_->skipped_meshes_;
}
