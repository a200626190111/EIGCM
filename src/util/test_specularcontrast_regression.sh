#!/usr/bin/env bash
set -euo pipefail

binary=${1:-"$(cd "$(dirname "$0")/../build_linux/util" && pwd)/specularcontrast"}
oconv_bin=${OCONV:-oconv}

if [[ ! -x "$binary" ]]; then
    echo "specularcontrast executable not found: $binary" >&2
    exit 2
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/specularcontrast-regression.XXXXXX")
trap 'rm -rf "$work"' EXIT

cat > "$work/ideal.rad" <<'EOF'
void light solar0
0
0
3 100 100 100

solar0 source sun0
0
0
4 0 0 1 0.533

void light solar1
0
0
3 50 50 50

solar1 source sun1
0
0
4 0.2 0 0.979795897 0.533

void mirror test_mirror
0
0
3 1 1 1

test_mirror polygon mirror_plane
0
0
12
-10 -10 0
 10 -10 0
 10  10 0
-10  10 0
EOF

cat > "$work/ideal_views.pts" <<'EOF'
0 0 1 0 0 -1
0 0 1 0 0 -1
EOF
printf '0 0 1\n' > "$work/ideal_normals.txt"
printf 'test_mirror\n' > "$work/ideal.mod"
"$oconv_bin" -f "$work/ideal.rad" > "$work/ideal.oct"

ideal_args=(
    -h -q -vf "$work/ideal_views.pts" -N "$work/ideal_normals.txt"
    -S "$work/ideal.rad" -M "$work/ideal.mod"
    --integrated-path-check --direct-specular-only
    -n 2 -b 2 -t 0 "$work/ideal.oct"
)
"$binary" "${ideal_args[@]}" > "$work/ideal_reuse.mtx"
"$binary" "${ideal_args[@]:0:${#ideal_args[@]}-1}" \
    --no-origin-reuse "$work/ideal.oct" > "$work/ideal_no_reuse.mtx"

cat > "$work/ideal_expected.mtx" <<'EOF'
2.177731402e+04 4.431302982e+03
2.177731402e+04 4.431302982e+03
EOF

cat > "$work/rough_material.rad" <<'EOF'
void metal rough_metal
0
0
5 1 1 1 1 0.05
EOF

cat > "$work/rough_scene.rad" <<'EOF'
rough_metal polygon reflector
0
0
12
1 -1 0
1 -1 2
1 1 2
1 1 0
EOF

cat > "$work/rough_suns.rad" <<'EOF'
void light solar_rough
0
0
3 1e6 1e6 1e6

solar_rough source sun_rough
0
0
4 -1 0 0 0.533
EOF

printf '0 0 1 1 0 0\n' > "$work/rough_views.pts"
printf '1 0 0\n' > "$work/rough_normals.txt"
printf 'rough_metal\n' > "$work/rough.mod"
"$oconv_bin" -f "$work/rough_material.rad" "$work/rough_scene.rad" \
    "$work/rough_suns.rad" > "$work/rough.oct"

rough_common=(
    -h -q -vf "$work/rough_views.pts" -N "$work/rough_normals.txt"
    -S "$work/rough_suns.rad" -M "$work/rough.mod"
    --roughness 0.05 --rough-seed 0 -t 0 "$work/rough.oct"
)
"$binary" "${rough_common[@]:0:${#rough_common[@]}-1}" \
    --rough-samples 64 "$work/rough.oct" > "$work/rough_fixed.mtx"

"$binary" -h -q -vf "$work/rough_views.pts" \
    -N "$work/rough_normals.txt" -S "$work/rough_suns.rad" \
    -M "$work/rough.mod" --roughness 0.05 \
    --rough-samples 1024 --rough-secondary-samples 16 --rough-seed 0 \
    --rough-pilot-samples 64 \
    --rough-pilot-threshold 1e-6 --rough-pilot-guard-angle 5 \
    --rough-pilot-anchor-angle 5 --adaptive-rough-cells \
    --rough-medium-samples 64 --rough-cell-trigger 0.5 \
    --normal-tolerance 0.25 -t 2000 "$work/rough.oct" \
    > "$work/rough_adaptive.mtx"

"$binary" -h -q -vf "$work/rough_views.pts" \
    -S "$work/rough_suns.rad" --auto-materials --reflection-level 3 \
    --max-specular-bounces 1 --rough-samples 64 \
    --rough-pilot-samples 16 --rough-secondary-samples 4 \
    -t 0 "$work/rough.oct" > "$work/rough_auto.mtx" \
    2> "$work/rough_auto.stderr"

printf '2.422678947e+09\n' > "$work/rough_fixed_expected.mtx"
printf '9.113496635e+08\n' > "$work/rough_adaptive_expected.mtx"
printf '5.713211058e+08\n' > "$work/rough_auto_expected.mtx"

compare_matrix() {
    local expected=$1 actual=$2 label=$3
    paste <(tr ' \t' '\n\n' < "$expected" | sed '/^$/d') \
          <(tr ' \t' '\n\n' < "$actual" | sed '/^$/d') |
        awk -v label="$label" '
            BEGIN { n=0; failed=0 }
            NF != 2 { failed=1; next }
            {
                e=$1+0; a=$2+0; d=e-a; if (d<0) d=-d
                scale=(e<0 ? -e : e); if (scale<1) scale=1
                if (d > 1e-7*scale) {
                    printf "%s mismatch at %d: expected %.12g, got %.12g\n", \
                        label, n+1, e, a > "/dev/stderr"
                    failed=1
                }
                n++
            }
            END { if (!n || failed) exit 1 }
        '
}

compare_matrix "$work/ideal_expected.mtx" "$work/ideal_reuse.mtx" \
    "ideal reflection"
compare_matrix "$work/ideal_reuse.mtx" "$work/ideal_no_reuse.mtx" \
    "origin reuse"
compare_matrix "$work/rough_fixed_expected.mtx" "$work/rough_fixed.mtx" \
    "fixed rough sampling"
compare_matrix "$work/rough_adaptive_expected.mtx" \
    "$work/rough_adaptive.mtx" "adaptive rough cells"
compare_matrix "$work/rough_auto_expected.mtx" "$work/rough_auto.mtx" \
    "automatic material classification"
if [[ -s "$work/rough_auto.stderr" ]]; then
    echo "quiet automatic-material run wrote to stderr" >&2
    cat "$work/rough_auto.stderr" >&2
    exit 1
fi

echo "specularcontrast regression checks passed"
