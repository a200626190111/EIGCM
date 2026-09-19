#ifndef DIRECT_LOBE_PROPOSALS_H
#define DIRECT_LOBE_PROPOSALS_H

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

enum DirectLobeProposalKind {
    DLP_TRANSMISSION = 1u << 0,
    DLP_REFRACTION = 1u << 1,
    DLP_PRISM = 1u << 2,
    DLP_ROUGH_SEED = 1u << 3
};

struct DirectLobeProposal {
    std::array<double, 3> direction;
    double solid_angle;
    unsigned int kinds;
};

struct DirectLobeProposalOptions {
    double normal_tolerance_degrees = 0.1;
};

class DirectLobeProposalGenerator {
public:
    DirectLobeProposalGenerator(
        const std::string &octree_path,
        const DirectLobeProposalOptions &options);
    ~DirectLobeProposalGenerator();

    std::vector<DirectLobeProposal> generate(
        const std::array<double, 3> &sun_direction,
        double sun_solid_angle) const;

    std::size_t rough_transmission_surface_count() const;
    std::size_t refractive_normal_count() const;
    std::size_t prism_surface_count() const;
    std::size_t skipped_instance_count() const;
    std::size_t skipped_mesh_count() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
