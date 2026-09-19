# Command-line reference

## `eigcm`

The installed scheduler command is `eigcm`. In a source checkout, use
`python3 eigcm_scheduler.py` in the examples below.

```text
eigcm init OUTPUT.cfg
eigcm inspect CONFIG [--json] [--set SECTION.OPTION=VALUE ...]
eigcm plan CONFIG [--json] [--summary] [--save workflow.sh]
eigcm run CONFIG [--force] [--from STEP] [--until STEP]
                 [--set SECTION.OPTION=VALUE ...]
```

- `init` copies the bundled configuration template without overwriting an
  existing file.
- `inspect` parses Radiance materials and geometry, reports the detected
  representation, automatically groups fenestration, and explains module
  selection.
- `plan` creates an ordered workflow without running Radiance. `--summary`
  hides full shell commands, `--json` emits machine-readable output, and
  `--save` writes a reproducible Bash script and generated receiver files.
- `run` executes the dependency-ordered plan. Completed steps are reused by
  default. `--force` reruns selected steps; `--from` and `--until` select an
  inclusive step range.
- `--set section.option=value` temporarily overrides any INI value and may be
  repeated.

## `dcglare2`

EIGCM normally uses the multi-window replacement form:

```text
dcglare2 [common options] \
  {-wgroup RECEIVER.rad \
   -wVTotal Vtotal.mtx -wTDSTotal TDStotal.mtx \
   -wVDirect Vdirect.mtx -wTDSDirect TDSdirect.mtx}... \
  [-wVisibleSun visible_sun.mtx -suns suns.rad] \
  [-wSpecularContrast explicit_contrast.mtx] Ev.mtx
```

Each `-wgroup` starts one independently oriented Reinhart window hemisphere.
`-wVTotal/-wTDSTotal` provide the smooth total field and
`-wVDirect/-wTDSDirect` identify the matched coarse direct contribution to be
removed. `-wSpecularContrast` adds the combined explicit contrast matrix once,
and `Ev.mtx` supplies annual vertical eye illuminance.

Common options:

| Option | Meaning |
| --- | --- |
| `-vf FILE` | Viewpoint directions; EIGCM cases use six values per row. |
| `-vd x y z` | One view direction instead of `-vf`. |
| `-vu x y z` | View-up vector. |
| `-n N` | Number of annual time steps when it cannot be inferred. |
| `-if`, `-id`, `-ih` | Input matrix format: float, double, or half. |
| `-of`, `-od` | Output format: float or double. |
| `-l VALUE` | DGP reporting limit. |
| `-b VALUE` | Glare-source luminance threshold. |
| `-sf FILE` | Occupancy schedule file. |
| `-ss START -se END` | Occupied-hour interval. |
| `-h` | Suppress the Radiance matrix header. |

The two original `dcglare` forms remain available for direct-coefficient or
BSDF-matrix input; run `dcglare2 --help` for their positional syntax.

## `bsdf2reinhart`

```text
bsdf2reinhart [-q] [-fa|-ff|-fd] [-mf N | -mi N -mo N]
  [-n SAMPLES] [-s SEED] [-tb|-tf] [-E]
  [-C all|diffuse|non-diffuse] BSDF.xml [OUTPUT.mtx]
```

| Option | Meaning |
| --- | --- |
| `-mf N` | Use the same Reinhart multiplier for incident and outgoing bases. |
| `-mi N`, `-mo N` | Set incident and outgoing multipliers separately. |
| `-n N` | Monte Carlo samples per output element. |
| `-s N` | Deterministic sample seed. |
| `-tb`, `-tf` | Transmission Back (default) or Transmission Front. |
| `-C all` | Complete transmission. |
| `-C diffuse` | Lambertian/diffuse component only. |
| `-C non-diffuse` | Complete transmission minus diffuse transmission. |
| `-E` | Disable column energy normalization. |
| `-fa`, `-ff`, `-fd` | ASCII, float, or double matrix output. |
| `-q` | Suppress progress output. |

## `ttsuncontrast`

```text
ttsuncontrast -vf VIEWS -S SUNS \
  [--auto-bsdf SCENE.oct | {-wgroup WINDOW.rad -wBSDF SYSTEM.xml}...] \
  [options]
```

The output has one row per viewpoint and one column per solar record. Only
compact non-diffuse direct-sun transmission is integrated.

| Option | Meaning |
| --- | --- |
| `--auto-bsdf OCTREE` | Discover BSDF/aBSDF polygons and local frames. |
| `-wgroup FILE` | Start an explicit finite-window group. |
| `-wBSDF XML` | Assign a TensorTree or matrix BSDF to the current group. |
| `-wABsdf RAD ID` | Assign an aBSDF material; its through peak is excluded by default. |
| `--include-aBSDF-through` | Include the aBSDF through peak. |
| `--bsdf-samples N` | Outgoing importance samples per sun and group. |
| `--adaptive-peak-cells` | Use the validated adaptive equal-solid-angle reconstruction. |
| `--peak-cluster-factor D` | Fallback importance-sample adjacency scale. |
| `--peak-max-link-angle D` | Maximum fallback cluster link angle in degrees. |
| `--adaptive-base-level N` | Initial Shirley-Chiu grid level. |
| `--adaptive-max-level N` | Maximum refinement level. |
| `--adaptive-contrast-tolerance D` | Local contrast convergence tolerance. |
| `--adaptive-gradient-tolerance D` | Normalized luminance-gradient tolerance. |
| `--sun-disk-samples N` | Perfect-square incident solar-disk samples. |
| `--sample-seed N` | Deterministic BSDF sampling seed. |
| `-i OCTREE` | Reject rays blocked before the window plane. |
| `--rtrace CMD` | Visibility-query executable. |
| `--visibility-batch N` | Rays per visibility query. |
| `--visibility-tolerance D` | Window hit-distance tolerance. |
| `-n N` | Visibility worker count. |
| `-b D` | Glare-source threshold in cd/m2. |
| `-vu x y z` | Up vector used for windows and Guth position index. |
| `-o FILE` | Output matrix; default is standard output. |
| `-h`, `-q` | Suppress matrix header or progress messages. |

## `directlobecontrast`

```text
directlobecontrast -vf VIEWS -S SUNS [options] SCENE.oct
```

The solver scans the octree for analytical transmission/refraction materials,
constructs candidate directions automatically, and returns direct-sun plus
non-diffuse lobe contrast. Mirror paths and XML BSDF peaks are excluded.

| Option | Meaning |
| --- | --- |
| `-r N` | Base Shirley-Chiu grid width. |
| `--adaptive-levels N` | Maximum lobe quadtree levels. |
| `--adaptive-trigger D` | Luminance refinement trigger. |
| `--adaptive-gradient D` | Neighbor-gradient trigger. |
| `--adaptive-variance D` | Sibling-cell deviation trigger. |
| `--adaptive-uncertainty D` | Unresolved relative DGP-contrast budget. |
| `--adaptive-guard N` | Neighboring refinement rings. |
| `--contrast-tolerance D` | Relative contrast convergence tolerance. |
| `--convergence-passes N` | Stable levels required before convergence. |
| `--min-adaptive-levels N` | Mandatory adaptive levels. |
| `--normal-tolerance D` | Surface-normal deduplication angle. |
| `-c N` | Repeated source samples. |
| `-t D` | Glare threshold in cd/m2. |
| `--min-cluster-cells N` | Minimum connected lobe size. |
| `--max-lobe-bounces N` | Radiance recursion limit for lobes. |
| `--lobe-samples N` | Radiance rough-lobe samples. |
| `--sun-disk-resolution N` | Base adaptive solar-disk grid width. |
| `--sun-disk-levels N` | Maximum solar-disk refinement levels. |
| `--sun-disk-min-levels N` | Mandatory solar-disk levels. |
| `--sun-disk-gradient D` | Solar luminance-gradient trigger. |
| `--sun-disk-guard N` | Solar boundary guard rings. |
| `--sun-disk-tolerance D` | Relative solar contrast convergence. |
| `--lobe-sun-disk-resolution N` | Transmission seed grid width. |
| `--lobe-sun-disk-pilot-resolution N` | Pilot width; zero traces the full grid. |
| `--lobe-sun-disk-pilot-guard-angle D` | Complete neighboring solar positions. |
| `--sun-mode batch|adaptive` | Direct-sun calculation route. |
| `--sun-samples N` | Batch irradiance samples. |
| `--view-batch-size N` | Viewpoints per lobe trace batch. |
| `--sun-only`, `--lobe-only` | Restrict calculation to one component. |
| `--sun-output FILE` | Direct-sun contrast component. |
| `--lobe-output FILE` | Transmission/refraction contrast component. |
| `--sun-irradiance-output FILE` | Direct-sun RGB irradiance. |
| `--sun-total-irradiance-output FILE` | Include mirror virtual-source irradiance. |
| `--lobe-view-mask FILE` | One `0/1` value per view; zero skips lobe sampling. |
| `--render-options "..."` | Additional built-in Radiance options. |
| `-vu x y z`, `-n N` | View-up vector and worker count. |
| `-o FILE`, `-h`, `-q` | Output, header, and progress controls. |

## `specularcontrast`

```text
specularcontrast -vf VIEWS -S SUNS \
  [-N NORMALS | --normal-rad GEOMETRY.rad] [-M MATERIALS.mod] \
  [options] SCENE.oct
```

The result has one row per viewpoint and one column per solar record.

### Path discovery

| Option | Meaning |
| --- | --- |
| `--auto-materials` | Classify ideal and rough Radiance reflectors automatically. |
| `--auto-material-allowlist FILE` | Restrict automatic classification to listed identifiers. |
| `--ideal-only` | Ignore rough materials during automatic classification. |
| `-N FILE` | Load known normals. |
| `--normal-rad FILE` | Extract normals from Radiance geometry. |
| `-M FILE` | Material filter for hit checking. |
| `--proposal-modifiers FILE` | Separate material filter for path proposals. |
| `--reflection-level N` | Path presearch level. |
| `--reflection-seed N` | Deterministic presearch seed. |
| `--reflection-octree FILE` | Prebuilt scene containing a skyglow sky. |
| `--max-specular-bounces 1|2` | Include first- or second-order reflections. |
| `--all-normal-pairs` | Test all ordered normal pairs for second-order paths. |
| `--normal-tolerance D` | Normal deduplication angle. |
| `--save-normals FILE` | Save normals with source metadata. |
| `--save-reflection-paths FILE` | Save ordered paths and originating views. |

### Finite sun and rough paths

| Option | Meaning |
| --- | --- |
| `--sun-disk-samples N` | Equal-solid-angle samples for direct/first-order paths. |
| `--secondary-sun-disk-samples N` | Samples for second-order-only paths. |
| `--adaptive-sun-disk` | Pilot-test each path before full disk integration. |
| `--sun-disk-pilot-samples N` | Pilot samples for direct/first-order paths. |
| `--secondary-sun-disk-pilot-samples N` | Pilot samples for second-order paths. |
| `--sun-disk-pilot-guard-angle D` | Protect nearby matching paths. |
| `--sun-disk-seed N` | Deterministic solar-disk rotation. |
| `--roughness D` | Override effective Radiance roughness. |
| `--rough-samples N` | Final square Shirley-Chiu sample count. |
| `--adaptive-rough-sampling` | Refine only retained center/pilot paths. |
| `--rough-pilot-samples N` | Rescue samples after a rejected center ray. |
| `--rough-pilot-guard-angle D` | Guard angle for retained paths. |
| `--rough-pilot-anchor-angle D` | Annual rescue-anchor spacing. |
| `--adaptive-rough-cells` | Refine bright rough-cap cells only. |
| `--rough-coarse-samples N` | Initial rough integration level. |
| `--rough-medium-samples N` | Accepted intermediate level. |
| `--rough-convergence D` | Relative contrast/illuminance tolerance. |
| `--rough-illuminance-tolerance D` | Absolute illuminance tolerance in lux. |
| `--rough-cell-trigger D` | Cell trigger relative to glare threshold. |
| `--rough-secondary-samples N` | Samples at the second rough bounce. |
| `--rough-extent D` | Lobe radius in standard deviations. |
| `--rough-pilot-threshold D` | Pilot threshold relative to glare threshold. |
| `--rough-seed N` | Deterministic rough-cap rotation. |

### Evaluation and output

| Option | Meaning |
| --- | --- |
| `--include-direct-sun` | Add deterministic zero-reflection contrast. |
| `--integrated-path-check` | Validate paths during final in-process tracing. |
| `--no-hit-check` | Ignore `-M` during hit checks. |
| `--direct-specular-only` | Suppress direct diffuse terms. |
| `--nonspec-octree FILE` | Scene with target specularity removed. |
| `--mirror-illuminance-output FILE` | Write first/second-order eye illuminance. |
| `--no-origin-reuse` | Trace duplicate colocated view origins separately. |
| `-n N` | Worker count. |
| `-b N` | Solar modifiers per batch. |
| `--view-batch-size N` | Viewpoints retained per pass. |
| `-t D` | Glare threshold in cd/m2. |
| `--visible-fraction D` | Visible reflected solar-disk fraction. |
| `-u x y z` | View-up vector. |
| `--oconv`, `--rcontrib`, `--rtrace` | Override Radiance executables. |
| `--rcontrib-options "..."` | Additional rendering options. |
| `-o FILE`, `-h`, `-q` | Output, header, and progress controls. |

## Radiance commands scheduled by `eigcm`

| Command | Role in the workflow |
| --- | --- |
| `oconv` | Compile full, black, open-aperture, and explicit-sun octrees. |
| `gendaymtx` | Create annual smooth-sky, direct-sun, and explicit source files. |
| `rfluxmtx` | Compute view, daylight, and eye-illuminance coefficient matrices. |
| `rcontrib` | Compute finite-sun BSDF illuminance when that route is selected. |
| `dctimestep` | Multiply coefficient matrices by annual sky/sun vectors. |
| `rmtxop` | Convert RGB to illuminance and add/subtract matched components. |
| `rcrop`, `getinfo`, `rlam`, `rcollate` | Split and reassemble path-component matrices. |

