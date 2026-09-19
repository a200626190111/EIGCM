# Scheduler configuration

EIGCM uses an INI file. Relative paths are resolved from the configuration
file's directory. `${section:option}` interpolation is supported. Blank numeric
sampling values inherit the selected quality profile.

```bash
eigcm init eigcm.cfg
eigcm inspect eigcm.cfg
eigcm plan eigcm.cfg --summary
eigcm run eigcm.cfg
```

## `[scene]`

| Key | Description |
| --- | --- |
| `name` | Case identifier used in reports. |
| `root` | Working directory used to execute Bash commands. |
| `out` | Output/cache directory. |
| `views` | Six-column viewpoint file: origin followed by direction. |
| `nview` | View count or `auto` to count non-comment rows. |
| `nproc` | Radiance worker count. |
| `weather` | Radiance WEA file. |
| `sky_receiver` | Reinhart sky/ground receiver, normally MF1. |
| `materials` | One or more material files, one path per line. |
| `black_materials` | Matching black-interior material files. |
| `scene` | One or more geometry files. |

## `[workflow]`

`quality` is `draft`, `balanced`, or `high`. The four bundled manuscript cases
use `balanced`.

`directlobecontrast`, `specularcontrast`, `bsdf_matrix`, and `ttsuncontrast`
accept `auto`, `true`, or `false`. In `auto` mode the scheduler uses parsed
material types and active scene modifiers.

`ev_strategy` accepts:

- `auto`: `bsdf_sun` for a BSDF matrix workflow, otherwise `path_components`.
- `path_components`: separate total and matched direct eye-illuminance paths.
- `bsdf_sun`: replace coarse BSDF direct-sun illuminance with finite-sun rays.
- `total`: retain total vertical eye illuminance without matched subtraction.

`specularity_threshold` and `roughness_threshold` control automatic material
classification. `material_include` and `material_exclude` restrict scene
inspection when a material file contains unused definitions.

### Quality profiles

| Parameter | Draft | Balanced | High |
| --- | ---: | ---: | ---: |
| `background_ad` | 10000 | 50000 | 100000 |
| `background_lw` | 1e-4 | 2e-5 | 1e-5 |
| `view_ad` | 10000 | 40000 | 80000 |
| `view_lw` | 1e-4 | 2.5e-5 | 1.25e-5 |
| `daylight_ad` | 500 | 1000 | 4000 |
| `daylight_lw` | 0.002 | 0.001 | 0.00025 |
| `daylight_c` | 64 | 250 | 1000 |
| `sun_samples` | 1000 | 10000 | 40000 |
| `lobe_samples` | 8 | 16 | 64 |
| `lobe_disk_resolution` | 4 | 8 | 16 |
| `bsdf_samples` | 64 | 256 | 1024 |
| `bsdf_matrix_samples` | 256 | 1024 | 4096 |
| `sun_disk_samples` | 16 | 64 | 256 |
| `secondary_sun_disk_samples` | 4 | 16 | 64 |
| `rough_samples` | 256 | 1024 | 4096 |
| `rough_secondary_samples` | 4 | 16 | 64 |
| `reflection_level` | 4 | 5 | 6 |

## `[auto_window_groups]`

| Key | Description |
| --- | --- |
| `enabled` | `auto`, `true`, or `false`; `auto` runs only when no explicit groups exist. |
| `orientation_tolerance_deg` | Maximum normal difference within one group. |
| `minimum_area` | Ignore smaller transmitting polygons. |
| `receiver_offset` | Shift generated glow polygons into the room. |
| `interior_point` | Optional `x y z` known to be indoors. |
| `material_include`, `material_exclude` | Restrict candidate fenestration materials. |
| `open_scene` | Window-free geometry used for BSDF daylight matrices. |
| `open_materials` | Materials for the BSDF open scene. |
| `open_black_materials` | Black materials for the BSDF open scene. |

The scheduler first uses `interior_point` to orient window normals. If it is
blank, each window uses the nearest position in `scene.views`; the geometry
center is the final fallback. Surfaces are grouped by optical material, BSDF
file, and inward normal. A separate MF receiver is generated for every group.

## `[runtime]`

| Key | Description |
| --- | --- |
| `shell` | Shell used for pipeline execution; supported value is `bash`. |
| `resume` | Reuse outputs newer than all inputs. |
| `skip_preflight` | Skip executable discovery checks. |
| `path_prepend` | Colon-separated directories prepended to `PATH`. |

## `[commands]`

Override executable names or absolute paths for `oconv`, `gendaymtx`,
`rfluxmtx`, `rcontrib`, `dctimestep`, `rmtxop`, `rcrop`, `getinfo`, `rlam`,
`rcollate`, `bsdf2reinhart`, `directlobecontrast`, `specularcontrast`,
`ttsuncontrast`, and `dcglare2`.

## `[radiance]`

| Key | Description |
| --- | --- |
| `mf` | Reinhart multiplier; the validated default is 1. |
| `photometric_weights` | RGB-to-illuminance weights; default `47.4 119.9 11.6`. |
| `background_ab`, `background_direct_ab` | Ambient bounces for total and matched direct eye illuminance. |
| `background_ad`, `background_lw` | Eye-illuminance sampling controls. |
| `view_ab`, `view_ad`, `view_lw` | View-matrix controls. |
| `daylight_ab`, `daylight_ad`, `daylight_lw`, `daylight_c` | Daylight-matrix controls. |
| `daylight_direct_ss` | Specular source subdivisions in the matched direct route. |

## Solver sections

### `[directlobecontrast]`

`include_illuminance` is `auto`, `true`, or `false`. `sun_samples`,
`lobe_samples`, and `lobe_disk_resolution` override the profile.
`lobe_disk_pilot_resolution`, `lobe_disk_pilot_guard_angle`, and `extra_args`
map directly to the command documented in `CLI_REFERENCE.md`.

### `[specularcontrast]`

`allowlist` optionally supplies a material list. Sampling keys are
`sun_disk_samples`, `secondary_sun_disk_samples`, `rough_samples`,
`rough_secondary_samples`, `reflection_level`, `max_specular_bounces`,
`block_size`, and `threshold`. `include_illuminance` is tri-state.
`extra_args` appends validated advanced command options.

### `[bsdf]`

| Key | Description |
| --- | --- |
| `xml` | Additional XML files not found through active materials. |
| `direction` | `tb` or `tf`. |
| `bsdf_matrix_samples` | Complete BSDF resampling count. |
| `diffuse_matrix_samples` | Diffuse-only resampling count. |
| `matrix_seed` | Deterministic resampling seed. |
| `sun_samples`, `sun_ad`, `sun_lw` | Finite-sun BSDF illuminance controls. |

### `[ttsuncontrast]`

`bsdf_samples`, `sun_disk_samples`, and `sample_seed` set the compact peak
solver. `extra_args` may adjust the validated adaptive-cell tolerances.

### `[dcglare2]`

`output` sets the final DGP matrix. `threshold` overrides the default source
luminance threshold. `extra_args` appends standard evaluation options such as
an occupancy schedule.

## Explicit window groups

Automatic groups can be replaced or selectively overridden:

```ini
[windowgroup:south]
receiver = ./receivers/south_mf1.rad
mode = geometry

[windowgroup:west_bsdf]
receiver = ./receivers/west_mf1.rad
mode = bsdf
xml = ./system.xml
open_scene = ./room_no_windows.rad
open_materials = ./materials.rad
open_black_materials = ./materials_black.rad
```

`mode` is `auto`, `geometry`, `bsdf`, or `precomputed`. A `precomputed` group
also accepts `tds_total` and `tds_direct`. Explicit group names replace
automatically generated groups with the same name.

