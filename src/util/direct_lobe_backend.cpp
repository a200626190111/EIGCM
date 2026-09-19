#include "direct_lobe_backend.h"

#include "RcontribSimulManager.h"
#include "func.h"
#include "otspecial.h"
#include "otypes.h"
#include "ray.h"
#include "rterror.h"
#include "rtotypes.h"
#include "source.h"
#include "standard.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace {

const unsigned int kDirectLobeEvent = 6u;

struct DirectLobeFilterContext {
    DirectLobeTraceMode mode;
};

void append_path_event(RAY *ray, unsigned int event)
{
    if (ray->rpath_event_count < 2*sizeof(ray->rpath_signature))
        ray->rpath_signature |= (event & 0xfu) <<
            (4*ray->rpath_event_count);
    if (ray->rpath_event_count != static_cast<unsigned short>(~0u))
        ray->rpath_event_count++;
    switch (event) {
    case RPE_MIRROR:
        if (ray->rpath_mirror_count != static_cast<unsigned short>(~0u))
            ray->rpath_mirror_count++;
        break;
    case RPE_BSDF_REFL:
    case RPE_BSDF_TRANS:
        if (ray->rpath_bsdf_count != static_cast<unsigned short>(~0u))
            ray->rpath_bsdf_count++;
        break;
    case RPE_OTHER_SPEC:
    case kDirectLobeEvent:
        if (ray->rpath_other_count != static_cast<unsigned short>(~0u))
            ray->rpath_other_count++;
        break;
    case RPE_DIFFUSE:
        if (ray->rpath_diffuse_count != static_cast<unsigned short>(~0u))
            ray->rpath_diffuse_count++;
        break;
    }
}

void mark_redirect(RAY *ray)
{
    if (ray->rpath_redirect_count != static_cast<unsigned short>(~0u))
        ray->rpath_redirect_count++;
}

int classify_direct_lobe_spawn(RAY *ray, const RAY *parent)
{
    if (!ray || !parent || !parent->ro)
        return 1;
    OBJREC *material = findmaterial(parent->ro);
    if (!material)
        return 1;
    const int rtype = ray->rtype;
    if (std::getenv("DIRECTLOBE_DEBUG_SPAWNS"))
        std::fprintf(stderr,
            "directlobe spawn: rtype=%o primitive=%s modifier=%s "
            "material=%s\n",
            rtype,
            parent->ro->oname ? parent->ro->oname : "(unnamed)",
            material->oname ? material->oname : "(unnamed)",
            ofun[material->otype].funame);
    if (rtype & AMBIENT) {
        mark_redirect(ray);
        append_path_event(ray, RPE_DIFFUSE);
        return 1;
    }
    if ((material->otype == MAT_BSDF || material->otype == MAT_ABSDF) &&
            (rtype & (RSHADOW|TSHADOW|REFLECTED|REFRACTED|
                      RSPECULAR|TSPECULAR))) {
        if ((rtype & RAYREFL) || !(rtype & TRANS))
            mark_redirect(ray);
        append_path_event(ray, rtype & RAYREFL ?
                          RPE_BSDF_REFL : RPE_BSDF_TRANS);
        return 1;
    }
    /* Every reflected branch belongs to specularcontrast, independent of
     * whether Radiance represents it as mirror, metal, plastic, dielectric,
     * or another reflective material. */
    if (rtype & RAYREFL) {
        mark_redirect(ray);
        append_path_event(ray, RPE_MIRROR);
        return 1;
    }
    /* Thin glass is straight unless a texture perturbs its normal.  The
     * transmitted child direction is assigned after rayorigin(), so mirror
     * glass.c's parent-state test rather than comparing child directions. */
    if (material->otype == MAT_GLASS && (rtype & (TRANS|TSHADOW))) {
        const bool textured = DOT(parent->pert, parent->pert) >
            FTINY*FTINY;
        const double refractive_index = material->oargs.nfargs == 4 ?
            material->oargs.farg[3] : 1.52;
        if (textured && std::fabs(refractive_index-1.0) > 1.0e-7 &&
                !(parent->crtype & (SHADOW|AMBIENT)) &&
                !usesPhongSmoothing(parent->ro)) {
            mark_redirect(ray);
            append_path_event(ray, kDirectLobeEvent);
        }
        return 1;
    }
    if (material->otype == MAT_TRANS && material->oargs.nfargs >= 7 &&
            material->oargs.farg[4] > 1.0e-7 &&
            material->oargs.farg[5] > 1.0e-7 &&
            material->oargs.farg[6] > 1.0e-7 &&
            (rtype & (TRANS|TSHADOW|REFRACTED|TSPECULAR))) {
        mark_redirect(ray);
        append_path_event(ray, kDirectLobeEvent);
        return 1;
    }
    if (material->otype == MAT_TRANS2 && material->oargs.nfargs >= 8 &&
            material->oargs.farg[4] > 1.0e-7 &&
            material->oargs.farg[5] > 1.0e-7 &&
            material->oargs.farg[6] > 1.0e-7 &&
            material->oargs.farg[7] > 1.0e-7 &&
            (rtype & (TRANS|TSHADOW|REFRACTED|TSPECULAR))) {
        mark_redirect(ray);
        append_path_event(ray, kDirectLobeEvent);
        return 1;
    }
    if ((material->otype == MAT_DIELECTRIC ||
            material->otype == MAT_INTERFACE) &&
            (rtype & REFRACTED)) {
        mark_redirect(ray);
        append_path_event(ray, kDirectLobeEvent);
        return 1;
    }
    if ((material->otype == MAT_DIRECT1 ||
            material->otype == MAT_DIRECT2) &&
            (rtype & (TRANS|TSHADOW|REFRACTED|TSPECULAR))) {
        mark_redirect(ray);
        append_path_event(ray, kDirectLobeEvent);
        return 1;
    }
    /* Other transmission families are not inferred from an uninitialized
     * child direction.  Radiance materials commonly assign rdir only after
     * rayorigin() invokes this callback. */
    return 1;
}

int accept_solar_terminal(const RAY *ray, void *data)
{
    const DirectLobeFilterContext *context =
        static_cast<const DirectLobeFilterContext *>(data);
    if (ray && std::getenv("DIRECTLOBE_DEBUG_PATHS"))
        std::fprintf(stderr,
            "directlobe path: rlvl=%d rtype=%o crtype=%o events=%u "
            "signature=0x%llx redirect=%u mirror=%u bsdf=%u "
            "other=%u diffuse=%u\n",
            ray->rlvl, ray->rtype, ray->crtype,
            static_cast<unsigned int>(ray->rpath_event_count),
            static_cast<unsigned long long>(ray->rpath_signature),
            static_cast<unsigned int>(ray->rpath_redirect_count),
            static_cast<unsigned int>(ray->rpath_mirror_count),
            static_cast<unsigned int>(ray->rpath_bsdf_count),
            static_cast<unsigned int>(ray->rpath_other_count),
            static_cast<unsigned int>(ray->rpath_diffuse_count));
    if (!ray || !context || ray->rpath_diffuse_count)
        return 0;
    if (context->mode == DLT_STRAIGHT_SUN)
        return ray->rpath_redirect_count == 0;
    /* Keep this partition disjoint from specularcontrast and
     * ttsuncontrast.  Mixed mirror/lobe paths require a dedicated matched
     * replacement term and are not included here. */
    return !ray->rpath_mirror_count && !ray->rpath_bsdf_count &&
           ray->rpath_other_count > 0 &&
           ray->rpath_redirect_count > 0;
}

RdataShare *anonymous_data_share(const char *, RCOutputOp, size_t size)
{
    return new RdataShareMap(NULL, RDSread | RDSwrite | RDSextend, size);
}

void set_rcontrib_defaults()
{
    do_irrad = 0;
    rand_samp = 1;
    dstrsrc = 1.0;
    shadthresh = 0.0;
    shadcert = 1.0;
    /* Match the explicit annual-sun calculation used for Ev.  Higher relay
     * depths generate a combinatorial set of mirror virtual sources for each
     * hourly sun and also change the physical path family being replaced. */
    directrelay = 1;
    vspretest = 0;
    directvis = 1;
    srcsizerat = 0.2;
    specthresh = 0.02;
    specjitter = 1.0;
    backvis = 1;
    maxdepth = -4;
    minweight = 1.0e-7;
    ambacc = 0.0;
    ambres = 256;
    ambdiv = 350;
    ambssamp = 0;
    ambounce = 0;
}

void apply_render_options(const std::vector<std::string> &options)
{
    std::vector<char *> arguments;
    arguments.reserve(options.size());
    for (std::size_t i = 0; i < options.size(); ++i)
        arguments.push_back(const_cast<char *>(options[i].c_str()));

    for (std::size_t i = 0; i < arguments.size();) {
        const int consumed = getrenderopt(
            static_cast<int>(arguments.size()-i), arguments.data()+i);
        if (consumed < 0)
            throw std::runtime_error(
                "unsupported built-in Radiance option '" + options[i] + "'");
        i += static_cast<std::size_t>(consumed+1);
    }
    ambounce = 0;
    ambacc = 0.0;
    ambssamp = 0;
    shadthresh = 0.0;
}

} // namespace

class DirectLobeBackend::Impl {
public:
    Impl(int worker_count, int accumulation,
         const std::vector<std::string> &render_options)
        : worker_count_(worker_count), accumulation_(accumulation),
          filter_context_{DLT_DIRECT_LOBE},
          previous_direct_specular_only_(direct_specular_only),
          previous_spawn_check_(ray_spawn_check)
    {
        initfunc();
        calcontext(RCCONTEXT);
        set_rcontrib_defaults();
        apply_render_options(render_options);

        /* This backend supplies two disjoint partitions: zero-redirect direct
         * sun and paths containing ordinary direction-changing transmission
         * or refraction. Pure mirror paths belong to specularcontrast and XML
         * BSDF peaks to ttsuncontrast. */
        direct_specular_only = 1;
        if (ray_spawn_check && ray_spawn_check != &classify_direct_lobe_spawn)
            throw std::runtime_error("Radiance ray-spawn filter is already set");
        ray_spawn_check = &classify_direct_lobe_spawn;

        manager_.SetFlag(RCcontrib, true);
        manager_.SetTargetOnlyOutput(true);
        manager_.SetRayFilter(&accept_solar_terminal, &filter_context_);
        manager_.cdsF = &anonymous_data_share;
        manager_.outOp = RCOforce;
    }

    ~Impl()
    {
        manager_.Cleanup(true);
        direct_specular_only = previous_direct_specular_only_;
        ray_spawn_check = previous_spawn_check_;
    }

    void load_scene(const std::string &octree)
    {
        if (!manager_.LoadOctree(octree.c_str()))
            throw std::runtime_error("cannot load octree '" + octree + "'");
        build_source_map();
    }

    std::vector<float> trace(const std::vector<std::string> &modifiers,
                             const std::vector<DirectLobeRay> &rays,
                             DirectLobeTraceMode mode)
    {
        if (rays.empty())
            return std::vector<float>();
        if (modifiers.empty())
            throw std::runtime_error("direct-lobe trace has no solar sources");

        std::vector<unsigned char> used(modifiers.size(), 0);
        for (std::size_t i = 0; i < rays.size(); ++i) {
            if (rays[i].target_modifier >= modifiers.size())
                throw std::runtime_error("invalid targeted solar modifier");
            used[rays[i].target_modifier] = 1;
        }
        std::vector<int> target_sources(modifiers.size(), -1);
        for (std::size_t i = 0; i < modifiers.size(); ++i) {
            if (!used[i])
                continue;
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

        filter_context_.mode = mode;
        manager_.ClearModifiers();
        manager_.xres = 0;
        manager_.yres = static_cast<int>(rays.size());
        const int sample_count = mode == DLT_STRAIGHT_SUN ?
            1 : accumulation_;
        manager_.accum = static_cast<unsigned int>(sample_count);
        if (!manager_.SetDataFormat('f'))
            throw std::runtime_error("cannot select float output");

        const char output_name[] = "directlobecontrast-memory";
        for (std::size_t i = 0; i < modifiers.size(); ++i) {
            if (!used[i])
                continue;
            if (!manager_.AddModifier(modifiers[i].c_str(), output_name,
                                      NULL, NULL, 1))
                throw std::runtime_error("cannot add solar modifier '" +
                                         modifiers[i] + "'");
        }
        if (manager_.PrepOutput() < 0)
            throw std::runtime_error("cannot prepare direct-lobe output");
        if (manager_.SetThreadCount(worker_count_) != worker_count_)
            throw std::runtime_error("cannot start requested Radiance workers");

        std::vector<FVECT> bundle(2*static_cast<std::size_t>(sample_count));
        for (std::size_t i = 0; i < rays.size(); ++i) {
            if (rays[i].target_modifier >= target_sources.size())
                throw std::runtime_error("invalid targeted solar modifier");
            for (int sample = 0; sample < sample_count; ++sample)
                for (int component = 0; component < 3; ++component) {
                    bundle[2*sample][component] =
                        static_cast<float>(rays[i].origin[component]);
                    bundle[2*sample+1][component] =
                        static_cast<float>(rays[i].direction[component]);
                }
            if (manager_.ComputeRecordTarget(
                    bundle.data(), target_sources[rays[i].target_modifier],
                    0u) != sample_count)
                throw std::runtime_error("built-in Radiance trace failed");
        }
        if (!manager_.FlushQueue() ||
                manager_.GetRowFinished() != static_cast<int>(rays.size()))
            throw std::runtime_error("built-in Radiance trace did not finish");

        RcontribOutput *output = const_cast<RcontribOutput *>(
            manager_.GetOutput());
        if (!output || output->Next() || output->rowBytes != 3*sizeof(float))
            throw std::runtime_error("unexpected direct-lobe output layout");

        std::vector<float> values(3*rays.size(), 0.0f);
        for (std::size_t row = 0; row < rays.size(); ++row) {
            const float *source_values = static_cast<const float *>(
                output->GetRow(static_cast<int>(row)));
            if (!source_values)
                throw std::runtime_error("cannot read direct-lobe output row");
            std::memcpy(values.data()+3*row, source_values, 3*sizeof(float));
        }
        output->DoneRow();
        manager_.ClearModifiers();
        return values;
    }

    std::vector<float> trace_sun_irradiance(
        const std::vector<std::string> &modifiers,
        const std::vector<DirectLobeSensor> &sensors, int samples,
        std::vector<float> *total_irradiance,
        std::vector<float> *visible_fraction)
    {
        if (sensors.empty())
            return std::vector<float>();
        if (modifiers.empty())
            throw std::runtime_error("direct-sun trace has no solar sources");
        if (samples < 1)
            throw std::runtime_error("direct-sun sample count must be positive");

        manager_.ClearModifiers();
        if (!manager_.SetTargetOnlyOutput(false) ||
                !manager_.SetRayFilter(NULL, NULL) ||
                !manager_.SetFlag(RTimmIrrad, true))
            throw std::runtime_error("cannot select irradiance source sampling");
        direct_specular_only = 0;
        if (!manager_.SetSourceComponentOutput(true))
            throw std::runtime_error(
                "cannot select direct-source component output");
        if (!manager_.SetSourceVisibilityOutput(true))
            throw std::runtime_error(
                "cannot select direct-source visibility output");
        rand_samp = 0;
        /* Split only enough records to keep all workers occupied.  With many
         * sensors, the sensors themselves provide the parallel work; making
         * worker_count_ chunks for every sensor multiplies shared output and
         * IPC traffic without adding concurrency. */
        const int sensor_count = static_cast<int>(sensors.size());
        const int chunks = std::min(samples, std::max(1,
            (worker_count_+sensor_count-1)/sensor_count));
        std::vector<int> chunk_samples(chunks, samples/chunks);
        for (int chunk = 0; chunk < samples%chunks; ++chunk)
            ++chunk_samples[chunk];
        manager_.xres = 0;
        manager_.yres = static_cast<int>(sensors.size())*chunks;
        manager_.accum = static_cast<unsigned int>(chunk_samples[0]);
        if (!manager_.SetDataFormat('f'))
            throw std::runtime_error("cannot select float output");

        const char output_name[] = "directlobecontrast-sun-memory";
        for (std::size_t i = 0; i < modifiers.size(); ++i)
            if (!manager_.AddModifier(modifiers[i].c_str(), output_name,
                                      NULL, NULL, 1))
                throw std::runtime_error("cannot add solar modifier '"+
                                         modifiers[i]+"'");
        if (manager_.PrepOutput() < 0)
            throw std::runtime_error("cannot prepare direct-sun output");
        if (manager_.SetThreadCount(worker_count_) != worker_count_)
            throw std::runtime_error("cannot start requested Radiance workers");

        std::vector<FVECT> bundle(2*static_cast<std::size_t>(
            *std::max_element(chunk_samples.begin(), chunk_samples.end())));
        for (std::size_t sensor = 0; sensor < sensors.size(); ++sensor) {
            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int count = chunk_samples[chunk];
                for (int sample = 0; sample < count; ++sample)
                    for (int component = 0; component < 3; ++component) {
                        bundle[2*sample][component] = static_cast<float>(
                            sensors[sensor].origin[component]);
                        bundle[2*sample+1][component] = static_cast<float>(
                            sensors[sensor].direction[component]);
                    }
                manager_.accum = static_cast<unsigned int>(count);
                if (manager_.ComputeRecord(bundle.data()) != count)
                    throw std::runtime_error(
                        "built-in Radiance direct-sun trace failed");
            }
        }
        if (!manager_.FlushQueue() ||
                manager_.GetRowFinished() !=
                    static_cast<int>(sensors.size())*chunks)
            throw std::runtime_error(
                "built-in Radiance direct-sun trace did not finish");

        RcontribOutput *output = const_cast<RcontribOutput *>(
            manager_.GetOutput());
        const std::size_t final_row_values = 3*modifiers.size();
        const std::size_t row_values = final_row_values*3u;
        if (!output || output->Next() ||
                output->rowBytes != row_values*sizeof(float))
            throw std::runtime_error("unexpected direct-sun output layout");
        std::vector<float> values(
            final_row_values*sensors.size(), 0.0f);
        if (total_irradiance)
            total_irradiance->assign(
                final_row_values*sensors.size(), 0.0f);
        if (visible_fraction)
            visible_fraction->assign(
                modifiers.size()*sensors.size(), 0.0f);
        for (std::size_t sensor = 0; sensor < sensors.size(); ++sensor)
            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int row = static_cast<int>(sensor)*chunks+chunk;
                const float *source_values = static_cast<const float *>(
                    output->GetRow(row));
                if (!source_values)
                    throw std::runtime_error(
                        "cannot read direct-sun output row");
                const double weight =
                    static_cast<double>(chunk_samples[chunk])/samples;
                for (std::size_t modifier = 0;
                        modifier < modifiers.size(); ++modifier) {
                    const std::size_t source_offset = 9u*modifier;
                    const std::size_t target_offset =
                        sensor*final_row_values+3*modifier;
                    for (int component = 0; component < 3; ++component) {
                        if (total_irradiance)
                            (*total_irradiance)[target_offset+component] +=
                                static_cast<float>(weight*
                                    source_values[
                                        source_offset+component]);
                        values[target_offset+component] +=
                            static_cast<float>(weight*
                                source_values[source_offset+3u+component]);
                    }
                    if (visible_fraction)
                        (*visible_fraction)[sensor*modifiers.size()+modifier] +=
                            static_cast<float>(weight*
                                source_values[source_offset+6u]);
                }
            }
        output->DoneRow();
        manager_.ClearModifiers();
        if (!manager_.SetSourceComponentOutput(false))
            throw std::runtime_error(
                "cannot restore direct-source component output");
        if (!manager_.SetSourceVisibilityOutput(false))
            throw std::runtime_error(
                "cannot restore direct-source visibility output");
        manager_.SetFlag(RTimmIrrad, false);
        direct_specular_only = 1;
        rand_samp = 1;
        if (!manager_.SetTargetOnlyOutput(true) ||
                !manager_.SetRayFilter(
                    &accept_solar_terminal, &filter_context_))
            throw std::runtime_error("cannot restore direct-lobe trace mode");
        return values;
    }

private:
    void build_source_map()
    {
        source_by_modifier_.clear();
        for (int source_index = 0; source_index < nsources; ++source_index) {
            if ((source[source_index].sflags & SVIRTUAL) ||
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
    int accumulation_;
    DirectLobeFilterContext filter_context_;
    int previous_direct_specular_only_;
    int (*previous_spawn_check_)(RAY *, const RAY *);
    std::map<std::string, int> source_by_modifier_;
    RcontribSimulManager manager_;
};

DirectLobeBackend::DirectLobeBackend(
    int worker_count, int accumulation,
    const std::vector<std::string> &render_options)
    : impl_(new Impl(worker_count, accumulation, render_options))
{
}

DirectLobeBackend::~DirectLobeBackend()
{
}

void DirectLobeBackend::load_scene(const std::string &octree)
{
    impl_->load_scene(octree);
}

std::vector<float> DirectLobeBackend::trace(
    const std::vector<std::string> &modifiers,
    const std::vector<DirectLobeRay> &rays,
    DirectLobeTraceMode mode)
{
    return impl_->trace(modifiers, rays, mode);
}

std::vector<float> DirectLobeBackend::trace_sun_irradiance(
    const std::vector<std::string> &modifiers,
    const std::vector<DirectLobeSensor> &sensors, int samples,
    std::vector<float> *total_irradiance,
    std::vector<float> *visible_fraction)
{
    return impl_->trace_sun_irradiance(
        modifiers, sensors, samples, total_irradiance, visible_fraction);
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
