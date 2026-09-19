# Manuscript validation cases

The four cases use the same 520 indoor viewpoints and Shanghai annual weather
file under `common/`. Geometry and optical inputs are self-contained in each
case directory. No precomputed EIGCM result matrices are included.

| Case | Fenestration | Automatically selected explicit solvers |
| --- | --- | --- |
| C1 | Clear glazing with specular venetian blinds | Direct lobe and first/second-order specular paths |
| C2 | TensorTree prismatic glazing | BSDF matrix, TensorTree solar peaks, and non-BSDF specular paths |
| C3 | Electrochromic glazing | Direct finite sun/transmission and specular paths |
| C4 | Specular blinds with rough transmitting glazing | Direct/rough transmission and specular paths |

Run scene inspection and print the planned stages:

```bash
for cfg in cases/C*/eigcm.cfg; do
  python3 eigcm_scheduler.py inspect "$cfg"
  python3 eigcm_scheduler.py plan "$cfg" --summary
done
```

Run one case:

```bash
python3 eigcm_scheduler.py run cases/C2-tensortree-prism/eigcm.cfg
```

Use the Draft profile for a functional smoke test without editing the file:

```bash
python3 eigcm_scheduler.py run cases/C2-tensortree-prism/eigcm.cfg \
  --set workflow.quality=draft
```

The final matrix is written to `output/dgp_annual.mtx` below the case
directory. Intermediate files are resumable and remain in the same output
directory. The C2 TensorTree XML is approximately 84 MiB and is intentionally
included because it is required to reproduce the published fenestration case.

