#include "specular_rcontrib_backend.h"

#include "RcontribSimulManager.h"
#include "func.h"
#include "object.h"
#include "otspecial.h"
#include "otypes.h"
#include "ray.h"
#include "rterror.h"
#include "source.h"
#include "standard.h"

#include <cerrno>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

std::set<std::string> path_reflection_modifiers;

int reject_specular_path(RAY *ray)
{
    /* Some direct-light callers do not inspect rayorigin()'s return value.
     * Preserve rejection in the ray state so the terminal contribution
     * callback cannot accept this branch. */
    ray->rpath_depth = ~0u;
    return 0;
}

int check_specular_path_spawn(RAY *ray, const RAY *parent)
{
    if (!ray || !parent || !parent->rpath_orders)
        return 1;

    if (!(ray->rtype & RAYREFL))
        return 1;
    if (ray->rtype & RAMBIENT)
        return reject_specular_path(ray);

    OBJREC *material = parent->ro ? findmaterial(parent->ro) : NULL;
    if (!material || !material->oname)
        return reject_specular_path(ray);
    const std::string name(material->oname);
    if (!path_reflection_modifiers.count(name))
        return reject_specular_path(ray);

    if (parent->rpath_depth == ~0u)
        return reject_specular_path(ray);
    const unsigned int next_depth = parent->rpath_depth + 1u;
    if (next_depth > 8u*sizeof(parent->rpath_orders) ||
            !(parent->rpath_orders >> (next_depth-1u)))
        return reject_specular_path(ray);
    ray->rpath_depth = next_depth;
    return 1;
}

RdataShare *anonymous_data_share(const char *, RCOutputOp, size_t size)
{
    return new RdataShareMap(
        NULL, RDSread | RDSwrite | RDSextend, size);
}

void set_rcontrib_defaults()
{
    do_irrad = 0;
    rand_samp = 1;
    dstrsrc = 0.9;
    shadthresh = 0.0;
    shadcert = 0.75;
    directrelay = 3;
    vspretest = 512;
    directvis = 1;
    srcsizerat = 0.2;
    specthresh = 0.02;
    specjitter = 1.0;
    backvis = 1;
    maxdepth = -10;
    minweight = 2.0e-3;
    ambacc = 0.0;
    ambres = 256;
    ambdiv = 350;
    ambssamp = 0;
    ambounce = 1;
}

void apply_render_options(const std::vector<std::string> &options)
{
    std::vector<char *> arguments;
    arguments.reserve(options.size());
    for (std::size_t i = 0; i < options.size(); ++i)
        arguments.push_back(const_cast<char *>(options[i].c_str()));

    for (std::size_t i = 0; i < arguments.size();) {
        const int consumed = getrenderopt(
            static_cast<int>(arguments.size() - i), arguments.data() + i);
        if (consumed < 0)
            throw std::runtime_error(
                "unsupported built-in rcontrib option '" + options[i] + "'");
        i += static_cast<std::size_t>(consumed + 1);
    }

    shadthresh = 0.0;
    ambssamp = 0;
    ambacc = 0.0;
}

double maximum3(double a, double b, double c)
{
    return std::max(a, std::max(b, c));
}

SpecularMaterialInfo inspect_material(const OBJREC *material)
{
    SpecularMaterialInfo info;
    info.name = material && material->oname ? material->oname : "";
    info.type = material ? ofun[material->otype].funame : "unknown";
    info.material_class = SPECULAR_MATERIAL_NONE;
    info.roughness = 0.0;
    info.specular_fraction = 0.0;
    if (!material)
        return info;

    const FUNARGS &args = material->oargs;
    switch (material->otype) {
    case MAT_MIRROR:
    case MAT_GLASS:
    case MAT_DIELECTRIC:
    case MAT_INTERFACE:
        info.material_class = SPECULAR_MATERIAL_IDEAL;
        info.specular_fraction = 1.0;
        break;

    case MAT_PLASTIC:
    case MAT_METAL:
    case MAT_TRANS:
        if (args.nfargs < 5) {
            info.material_class = SPECULAR_MATERIAL_UNSUPPORTED;
            break;
        }
        info.specular_fraction = args.farg[3];
        info.roughness = std::max(0.0, static_cast<double>(args.farg[4]));
        if (info.specular_fraction > FTINY)
            info.material_class = info.roughness > FTINY ?
                SPECULAR_MATERIAL_ROUGH : SPECULAR_MATERIAL_IDEAL;
        break;

    case MAT_PLASTIC2:
    case MAT_METAL2:
    case MAT_TRANS2:
        if (args.nfargs < 6) {
            info.material_class = SPECULAR_MATERIAL_UNSUPPORTED;
            break;
        }
        info.specular_fraction = args.farg[3];
        info.roughness = std::max(
            std::max(0.0, static_cast<double>(args.farg[4])),
            std::max(0.0, static_cast<double>(args.farg[5])));
        if (info.specular_fraction > FTINY)
            info.material_class = info.roughness > FTINY ?
                SPECULAR_MATERIAL_ROUGH : SPECULAR_MATERIAL_IDEAL;
        break;

    case MAT_ASHIKHMIN:
        if (args.nfargs < 8) {
            info.material_class = SPECULAR_MATERIAL_UNSUPPORTED;
            break;
        }
        info.specular_fraction = maximum3(
            args.farg[3], args.farg[4], args.farg[5]);
        if (info.specular_fraction > FTINY) {
            const double u_power = std::max(
                0.0, static_cast<double>(args.farg[6]));
            const double v_power = std::max(
                0.0, static_cast<double>(args.farg[7]));
            info.roughness = std::max(
                std::sqrt(2.0/(u_power+2.0)),
                std::sqrt(2.0/(v_power+2.0)));
            info.material_class = info.roughness > FTINY ?
                SPECULAR_MATERIAL_ROUGH : SPECULAR_MATERIAL_IDEAL;
        }
        break;

    case MAT_BSDF:
    case MAT_ABSDF:
        info.material_class = SPECULAR_MATERIAL_EXCLUDED;
        break;

    case MAT_PFUNC:
    case MAT_MFUNC:
    case MAT_PDATA:
    case MAT_MDATA:
        if (args.nfargs >= 4 && args.farg[3] > FTINY) {
            info.specular_fraction = args.farg[3];
            info.material_class = SPECULAR_MATERIAL_UNSUPPORTED;
        }
        break;

    case MAT_TFUNC:
    case MAT_TDATA:
        if (args.nfargs >= 4 && args.farg[3] > FTINY) {
            info.specular_fraction = args.farg[3];
            info.material_class = SPECULAR_MATERIAL_UNSUPPORTED;
        }
        break;

    case MAT_BRTDF:
    case MAT_WGMDF:
        info.material_class = SPECULAR_MATERIAL_UNSUPPORTED;
        break;
    }
    return info;
}

} // namespace

class SpecularRcontribBackend::Impl {
public:
    Impl(int worker_count, const std::vector<std::string> &render_options,
         bool specular_only, bool integrated_path_check,
         const std::vector<std::string> &reflection_modifiers)
        : worker_count_(worker_count),
          previous_direct_specular_only_(direct_specular_only),
          previous_spawn_check_(ray_spawn_check),
          integrated_path_check_(integrated_path_check)
    {
        initfunc();
        calcontext(RCCONTEXT);
        set_rcontrib_defaults();
        apply_render_options(render_options);
        direct_specular_only = specular_only ? 1 : 0;
        if (integrated_path_check) {
            path_reflection_modifiers.insert(reflection_modifiers.begin(),
                                             reflection_modifiers.end());
            ray_spawn_check = &check_specular_path_spawn;
        }
        manager_.SetFlag(RCcontrib, true);
        manager_.SetTargetOnlyOutput(true);
        manager_.cdsF = &anonymous_data_share;
        manager_.outOp = RCOforce;
    }

    ~Impl()
    {
        manager_.Cleanup(true);
        direct_specular_only = previous_direct_specular_only_;
        ray_spawn_check = previous_spawn_check_;
        path_reflection_modifiers.clear();
    }

    void load_scene(const std::string &octree)
    {
        if (!manager_.LoadOctree(octree.c_str()))
            throw std::runtime_error("cannot load octree '" + octree + "'");
        build_source_map();
    }

    std::vector<SpecularMaterialInfo> inspect_specular_materials() const
    {
        std::map<std::string, SpecularMaterialInfo> materials;
        for (OBJECT index = 0; index < nobjects; ++index) {
            const OBJREC *object = objptr(index);
            if (!object || !ismaterial(object->otype) || !object->oname)
                continue;
            const SpecularMaterialInfo info = inspect_material(object);
            if (info.material_class != SPECULAR_MATERIAL_NONE)
                materials[info.name] = info;
        }
        std::vector<SpecularMaterialInfo> result;
        result.reserve(materials.size());
        for (std::map<std::string, SpecularMaterialInfo>::const_iterator it =
                materials.begin(); it != materials.end(); ++it)
            result.push_back(it->second);
        return result;
    }

    void set_reflection_modifiers(
        const std::vector<std::string> &reflection_modifiers)
    {
        if (!integrated_path_check_)
            throw std::runtime_error(
                "cannot update reflection modifiers without integrated path checking");
        path_reflection_modifiers.clear();
        path_reflection_modifiers.insert(reflection_modifiers.begin(),
                                         reflection_modifiers.end());
    }

    std::vector<float> trace(
        const std::vector<std::string> &modifiers,
        const std::vector<SpecularContribRay> &rays)
    {
        if (rays.empty())
            return std::vector<float>();
        if (modifiers.empty())
            throw std::runtime_error(
                "built-in rcontrib trace has no solar modifiers");

        std::vector<int> target_sources(modifiers.size(), -1);
        for (std::size_t i = 0; i < modifiers.size(); ++i) {
            const std::map<std::string, int>::const_iterator found =
                source_by_modifier_.find(modifiers[i]);
            if (found == source_by_modifier_.end())
                throw std::runtime_error("solar modifier '" + modifiers[i] +
                    "' does not identify a distant source in the octree");
            if (found->second < 0)
                throw std::runtime_error("solar modifier '" + modifiers[i] +
                    "' identifies more than one source in the octree");
            target_sources[i] = found->second;
        }

        manager_.ClearModifiers();
        manager_.xres = 0;
        manager_.yres = static_cast<int>(rays.size());
        manager_.accum = 1;
        if (!manager_.SetDataFormat('f'))
            throw std::runtime_error(
                "cannot select float output for built-in rcontrib");

        const char output_name[] = "specularcontrast-memory";
        for (std::size_t i = 0; i < modifiers.size(); ++i)
            if (!manager_.AddModifier(
                    modifiers[i].c_str(), output_name, NULL, NULL, 1))
                throw std::runtime_error(
                    "cannot add solar modifier '" + modifiers[i] + "'");

        if (manager_.PrepOutput() < 0)
            throw std::runtime_error(
                "cannot prepare built-in rcontrib output");
        if (manager_.SetThreadCount(worker_count_) != worker_count_)
            throw std::runtime_error(
                "cannot start requested built-in rcontrib workers");

        for (std::size_t i = 0; i < rays.size(); ++i) {
            if (rays[i].target_modifier >= target_sources.size())
                throw std::runtime_error(
                    "built-in ray has an invalid target solar modifier");
            FVECT origin_direction[2];
            for (int component = 0; component < 3; ++component) {
                origin_direction[0][component] = static_cast<float>(
                    rays[i].origin[component]);
                origin_direction[1][component] = static_cast<float>(
                    rays[i].direction[component]);
            }
            if (manager_.ComputeRecordTarget(
                    origin_direction,
                    target_sources[rays[i].target_modifier],
                    integrated_path_check_ ?
                        rays[i].reflection_orders : 0u) != 1)
                throw std::runtime_error(
                    "built-in rcontrib failed to trace a targeted ray");
        }
        if (!manager_.FlushQueue() ||
                manager_.GetRowFinished() != static_cast<int>(rays.size()))
            throw std::runtime_error(
                "built-in rcontrib did not finish all targeted rays");

        RcontribOutput *output = const_cast<RcontribOutput *>(
            manager_.GetOutput());
        if (!output || output->Next())
            throw std::runtime_error(
                "unexpected built-in rcontrib output layout");

        if (output->rowBytes != 3*sizeof(float))
            throw std::runtime_error(
                "unexpected built-in rcontrib row size");
        std::vector<float> values(3*rays.size(), 0.0f);
        for (std::size_t row = 0; row < rays.size(); ++row) {
            const float *source_values = static_cast<const float *>(
                output->GetRow(static_cast<int>(row)));
            if (!source_values)
                throw std::runtime_error(
                    "cannot read built-in rcontrib output row");
            std::memcpy(values.data()+3*row,
                        source_values, 3*sizeof(float));
        }
        output->DoneRow();
        manager_.ClearModifiers();
        return values;
    }

private:
    void build_source_map()
    {
        source_by_modifier_.clear();
        for (int source_index = 0; source_index < nsources; ++source_index) {
            if (source[source_index].sflags & SVIRTUAL ||
                    !source[source_index].so)
                continue;
            OBJREC *material = findmaterial(source[source_index].so);
            if (!material || !material->oname)
                continue;
            const std::string name(material->oname);
            const std::map<std::string, int>::iterator existing =
                source_by_modifier_.find(name);
            if (existing == source_by_modifier_.end())
                source_by_modifier_[name] = source_index;
            else
                existing->second = -1;
        }
    }

    int worker_count_;
    int previous_direct_specular_only_;
    int (*previous_spawn_check_)(RAY *, const RAY *);
    bool integrated_path_check_;
    std::map<std::string, int> source_by_modifier_;
    RcontribSimulManager manager_;
};

SpecularRcontribBackend::SpecularRcontribBackend(
    int worker_count, const std::vector<std::string> &render_options,
    bool specular_only, bool integrated_path_check,
    const std::vector<std::string> &reflection_modifiers)
    : impl_(new Impl(worker_count, render_options, specular_only,
                     integrated_path_check, reflection_modifiers))
{
}

SpecularRcontribBackend::~SpecularRcontribBackend()
{
}

void SpecularRcontribBackend::load_scene(const std::string &octree)
{
    impl_->load_scene(octree);
}

std::vector<SpecularMaterialInfo>
SpecularRcontribBackend::inspect_specular_materials() const
{
    return impl_->inspect_specular_materials();
}

void SpecularRcontribBackend::set_reflection_modifiers(
    const std::vector<std::string> &reflection_modifiers)
{
    impl_->set_reflection_modifiers(reflection_modifiers);
}

std::vector<float> SpecularRcontribBackend::trace(
    const std::vector<std::string> &modifiers,
    const std::vector<SpecularContribRay> &rays)
{
    return impl_->trace(modifiers, rays);
}

extern "C" void quit(int code)
{
    std::exit(code);
}

extern "C" void eputs(const char *message)
{
    std::fputs(message, stderr);
}

extern "C" void wputs(const char *message)
{
    const int saved_errno = errno;
    if (erract[WARNING].pf != NULL)
        std::fputs(message, stderr);
    errno = saved_errno;
}
