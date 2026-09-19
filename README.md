# EIGCM

EIGCM is an enhanced imageless annual daylight-glare workflow built on
Radiance. It retains a reusable MF1 Reinhart background and explicitly resolves
the finite solar disk, analytical transmission/refraction lobes, first- and
second-order specular reflections, and compact BSDF/aBSDF transmission peaks.
Matched coarse direct terms are removed before the explicit source terms are
added to the final Daylight Glare Probability (DGP).

This repository is a Radiance 6.1 source tree with the EIGCM utilities,
automatic scheduler, and the four validation cases used in the accompanying
manuscript. Experimental command-line routes have been removed from this
release; the exposed options correspond to the validated workflow.

## Included tools

- `eigcm`: inspects a scene, groups windows, selects calculation modules, and
  schedules the complete annual workflow.
- `dcglare2`: combines coarse window luminance, vertical eye illuminance, and
  explicitly resolved contrast terms into annual DGP.
- `directlobecontrast`: resolves the finite sun and analytical
  transmission/refraction lobes.
- `specularcontrast`: resolves first- and second-order ideal or rough specular
  solar paths.
- `ttsuncontrast`: resolves compact non-diffuse TensorTree/BSDF/aBSDF peaks.
- `bsdf2reinhart`: resamples BSDF transmission to a Reinhart basis.

## Platform

The supported release platform is 64-bit Linux. Windows users should use WSL2
with Ubuntu. The scheduler emits Bash pipelines and `directlobecontrast` uses
the in-process POSIX Radiance backend, so a native Windows build is not the
supported workflow.

## Quick start

```bash
sudo apt update
sudo apt install -y build-essential cmake git perl \
  libx11-dev libxext-dev libgl1-mesa-dev libtiff-dev

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_HEADLESS=ON
cmake --build build -j"$(nproc)"
export PATH="$PWD/build/bin:$PATH"

python3 eigcm_scheduler.py inspect cases/C1-specular-blinds/eigcm.cfg
python3 eigcm_scheduler.py plan cases/C1-specular-blinds/eigcm.cfg --summary
python3 eigcm_scheduler.py run cases/C1-specular-blinds/eigcm.cfg
```

The four cases contain 520 viewpoints and a full annual Shanghai weather file.
A complete Balanced run is computationally intensive. Use
`--set workflow.quality=draft` for a short functional check.

## Documentation

- [Build and installation](docs/BUILDING.md)
- [Command-line reference](docs/CLI_REFERENCE.md)
- [Scheduler configuration](docs/CONFIGURATION.md)
- [Validation cases](cases/README.md)

## Manuscript source data

The numerical source data associated with the manuscript figures and tables are
available in [`manuscript_data/`](manuscript_data/). The directory contains a
formatted Excel summary, machine-readable CSV files organized by figure or
table number, and a [downloadable ZIP archive](manuscript_data/EIGCM_manuscript_source_data.zip).

## License

This repository includes the complete Radiance 6.1 source tree, not only the
EIGCM additions. The upstream Radiance source and the publicly released EIGCM
enhancements are distributed under the
[Radiance Software License, Version 2.0](License.txt), except where an
individual file contains a more specific notice. See [NOTICE.md](NOTICE.md) for
upstream attribution, the EIGCM modification scope, redistribution conditions,
and the non-endorsement statement.
