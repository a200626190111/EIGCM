/*
 *  cmglare.c - routines for calculating glare autonomy.
 *
 *  N. Jones
 */

/*
 * Copyright (c) 2017-2019 Nathaniel Jones
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "rtmath.h"
#include "cmglare2.h"
#include <ctype.h>

#define LUMINOUS_EFFICACY	179	/* lumens per Watt */
#define LUMINANCE_THRESHOLD	100 /* minimum threshold that will be interpreted as luminance rather than ratio between Ev and glare source */

#define ANGLE(u,v)	acos(DOT((u),(v)))	// TODO need to normalize?

typedef struct reinhart_sky {
	//int mf;					/* Linear divisions per Tregenza patch. */
	int rings;					/* Number of rings of sky patches. */
	//int patches;				/* Number of sky patches. */
	double ringElevationAngle;	/* Angle in radians between rings. */
	int *patchesPerRow;			/* Sky patches per row. */
	int *firstPatchIndex;		/* Index of first sky patch in each row. */
	double *solidAngle;			/* Solid angle of each patch in each row. */
} ReinhartSky;

typedef struct reinhart_div {
	int rings;
	double ringElevationAngle;
	int* patchesPerRow;
	int* firstPatchIndex;
	double* solidAngle;
}ReinhartWin;

typedef struct cm_window_group_state_s {
	const CM_WINDOW_GROUP	*group;
	ReinhartWin		*low;
	ReinhartWin		*high;
	FVECT			*low_patch_dir;
	FVECT			*high_patch_dir;
	int			*upsample_offset;
	int			*upsample_low_patch;
	double			*upsample_weight;
	double			*low_flux;
} CM_WINDOW_GROUP_STATE;

static const int tnaz[] = { 30, 30, 24, 24, 18, 12, 6 };	/* Number of bins per row */

extern char* progname;

static int
win_mf_from_nrows(const int nrows)
{
	switch (nrows) {
	case 1: return 0;
	case 145: return 1;
	case 577: return 2;
	case 1297: return 3;
	case 2305: return 4;
	case 3601: return 5;
	case 5185: return 6;
	case 7057: return 7;
	case 9217: return 8;
	case 11665: return 9;
	case 14401: return 10;
	case 17425: return 11;
	case 20737: return 12;
	default: return -1;
	}
}

static ReinhartWin* make_win(const CMATRIX* cmtx)
{
	int mf, count, i, j;
	double remaining = PI / 2;

	/* Check Reinhart subdivision for window outgoing direction bins */
	mf = win_mf_from_nrows(cmtx->nrows);
	if (mf < 0) {
		fprintf(stderr,
			"%s: unknown number of window direction bins %d\n",
			progname, cmtx->nrows);
		return NULL;
	}

	/* Allocate win */
	ReinhartWin* win = (ReinhartWin*)malloc(sizeof(ReinhartWin));
	if (!win) goto memerr;

	/* Calculate direction bins per row */
	win->rings = 7 * mf + 1;
	win->patchesPerRow = (int*)malloc(win->rings * sizeof(int));
	win->firstPatchIndex = (int*)malloc(win->rings * sizeof(int));
	if (!win->patchesPerRow || !win->firstPatchIndex) goto memerr;
	count = 0;
	for (i = 0; i < 7; i++)
		for (j = 0; j < mf; j++) {
			win->firstPatchIndex[i * mf + j] = count;
			count += win->patchesPerRow[i * mf + j] = tnaz[i] * mf;
	}
	win->firstPatchIndex[7 * mf] = count;
	win->patchesPerRow[7 * mf] = 1;

	/*
	 * Calculate solid angle of Reinhart outgoing direction bins.  These
	 * are angular bins on the exterior/window hemisphere, not physical
	 * window surface patches.
	 */
	win->solidAngle = (double*)malloc(win->rings * sizeof(double));
	if (!win->solidAngle) goto memerr;
	win->ringElevationAngle = PI / (2 * win->rings - 1);
	win->solidAngle[0] = 2 * PI;
	for (i = 1; i < win->rings; i++) {
		remaining -= win->ringElevationAngle;
		win->solidAngle[i] = 2 * PI * (1 - cos(remaining)); // solid angle of cap
		win->solidAngle[i - 1] -= win->solidAngle[i];
		win->solidAngle[i - 1] /= win->patchesPerRow[i - 1];
	}

	return win;

memerr:
	fprintf(stderr,
		"%s: out of memory\n",
		progname);
	return NULL;

}



static ReinhartSky* make_sky(const CMATRIX *smx)
{
	int mf, count, i, j;
	double remaining = PI / 2;

	/* Check sky partitionss */
	switch (smx->nrows) { // nrows is Reinhart sky subdivisoins plus one for miss
	case 2: mf = 0; break;
	case 146: mf = 1; break;
	case 578: mf = 2; break;
	case 1298: mf = 3; break;
	case 2306: mf = 4; break;
	case 3602: mf = 5; break;
	case 5186: mf = 6; break;
	case 7058: mf = 7; break;
	case 9218: mf = 8; break;
	case 11666: mf = 9; break;
	case 14402: mf = 10; break;
	case 17426: mf = 11; break;
	case 20738: mf = 12; break;
	default:
		fprintf(stderr,
			"%s: unknown number of sky patches %d\n",
			progname, smx->nrows);
		return NULL;
	}

	/* Allocate sky */
	ReinhartSky *sky = (ReinhartSky*)malloc(sizeof(ReinhartSky));
	if (!sky) goto memerr;

	/* Calculate patches per row */
	sky->rings = 7 * mf + 1;
	sky->patchesPerRow = (int*)malloc(sky->rings * sizeof(int));
	sky->firstPatchIndex = (int*)malloc(sky->rings * sizeof(int));
	if (!sky->patchesPerRow || !sky->firstPatchIndex) goto memerr;
	count = 1; // The below horizon patch
	for (i = 0; i < 7; i++)
		for (j = 0; j < mf; j++) {
			sky->firstPatchIndex[i * mf + j] = count;
			count += sky->patchesPerRow[i * mf + j] = tnaz[i] * mf;
		}
	sky->firstPatchIndex[7 * mf] = count;
	sky->patchesPerRow[7 * mf] = 1;
	//sky->patches = count + 1;

	/* Calculate solid angle of patches in each row */
	sky->solidAngle = (double*)malloc(sky->rings * sizeof(double));
	if (!sky->solidAngle) goto memerr;
	sky->ringElevationAngle = PI / (2 * sky->rings - 1);
	sky->solidAngle[0] = 2 * PI;
	for (i = 1; i < sky->rings; i++) {
		remaining -= sky->ringElevationAngle;
		sky->solidAngle[i] = 2 * PI * (1 - cos(remaining)); // solid angle of cap
		sky->solidAngle[i - 1] -= sky->solidAngle[i];
		sky->solidAngle[i - 1] /= sky->patchesPerRow[i - 1];
	}

	return sky;

memerr:
	fprintf(stderr,
		"%s: out of memory\n",
		progname);
	return NULL;
}

static void free_win(ReinhartWin* win)
{
	if (win) {
		free(win->patchesPerRow);
		free(win->firstPatchIndex);
		free(win->solidAngle);
		free(win);
	}
}

static int
get_win_patch_row(const ReinhartWin *win, const int patch)
{
	int row = win->rings - 1;
	while (row > 0 && patch < win->firstPatchIndex[row])
		row--;
	return row;
}

static void
get_win_patch_direction(const ReinhartWin *win, const int patch, FVECT target)
{
	int row = get_win_patch_row(win, patch);
	/* Reinhart coordinates before orientation: +Z is the local hemisphere apex. */
	const double alt = (row == win->rings-1) ?
			PI/2 :
			(row + .5)*win->ringElevationAngle;
	const double azi = 2 * PI * (patch - win->firstPatchIndex[row]) / win->patchesPerRow[row];
	const double cos_alt = cos(alt);

	target[0] = cos_alt * sin(azi);
	target[1] = cos_alt * cos(azi);
	target[2] = sin(alt);
}

static double
get_win_patch_angular_radius(const ReinhartWin *win, const int patch)
{
	const int row = get_win_patch_row(win, patch);
	FVECT center, corner;
	double alt0, alt1, azi, azi_width, maxang = 0.0;
	int i, j;

	if (row == win->rings-1)
		return(PI/2 - row*win->ringElevationAngle);
	get_win_patch_direction(win, patch, center);
	alt0 = row * win->ringElevationAngle;
	alt1 = (row + 1) * win->ringElevationAngle;
	azi = 2 * PI * (patch - win->firstPatchIndex[row]) /
			win->patchesPerRow[row];
	azi_width = 2 * PI / win->patchesPerRow[row];
	for (i = 0; i < 2; i++) {
		const double alt = i ? alt1 : alt0;
		const double cos_alt = cos(alt);
		for (j = 0; j < 2; j++) {
			const double az = azi + (j ? .5 : -.5)*azi_width;
			double ang;

			corner[0] = cos_alt * sin(az);
			corner[1] = cos_alt * cos(az);
			corner[2] = sin(alt);
			ang = ANGLE(center, corner);
			if (ang > maxang)
				maxang = ang;
		}
	}
	return(maxang);
}

static double
get_win_patch_solid_angle(const ReinhartWin *win, const int patch)
{
	const int row = get_win_patch_row(win, patch);
	return(win->solidAngle[row]);
}

static double
get_win_patch_projected_solid_angle(const ReinhartWin *win, const int patch)
{
	const int row = get_win_patch_row(win, patch);
	const double alt0 = row * win->ringElevationAngle;
	const double alt1 = (row == win->rings-1) ? PI/2 :
			(row + 1) * win->ringElevationAngle;
	const double azi_width = 2 * PI / win->patchesPerRow[row];
	const double s0 = sin(alt0);
	const double s1 = sin(alt1);

	return(.5 * azi_width * (s1*s1 - s0*s0));
}

static int
split_azimuth_interval(const double center, const double width,
		double segment[2][2])
{
	double lo, hi;

	if (width >= 2*PI-FTINY) {
		segment[0][0] = 0;
		segment[0][1] = 2*PI;
		return(1);
	}
	lo = center - .5*width;
	hi = center + .5*width;
	if (lo < 0) {
		segment[0][0] = 0;
		segment[0][1] = hi;
		segment[1][0] = lo + 2*PI;
		segment[1][1] = 2*PI;
		return(2);
	}
	if (hi > 2*PI) {
		segment[0][0] = lo;
		segment[0][1] = 2*PI;
		segment[1][0] = 0;
		segment[1][1] = hi - 2*PI;
		return(2);
	}
	segment[0][0] = lo;
	segment[0][1] = hi;
	return(1);
}

static double
get_win_patch_projected_overlap(const ReinhartWin *a, const int apatch,
		const ReinhartWin *b, const int bpatch)
{
	const int arow = get_win_patch_row(a, apatch);
	const int brow = get_win_patch_row(b, bpatch);
	const double a0 = arow*a->ringElevationAngle;
	const double a1 = (arow == a->rings-1) ? PI/2 :
			(arow + 1)*a->ringElevationAngle;
	const double b0 = brow*b->ringElevationAngle;
	const double b1 = (brow == b->rings-1) ? PI/2 :
			(brow + 1)*b->ringElevationAngle;
	const double alt0 = a0 > b0 ? a0 : b0;
	const double alt1 = a1 < b1 ? a1 : b1;
	const int acol = apatch - a->firstPatchIndex[arow];
	const int bcol = bpatch - b->firstPatchIndex[brow];
	const double awidth = 2*PI/a->patchesPerRow[arow];
	const double bwidth = 2*PI/b->patchesPerRow[brow];
	double aseg[2][2], bseg[2][2], az_overlap = 0;
	int na, nb, i, j;

	if (alt1 <= alt0)
		return(0);
	na = split_azimuth_interval(acol*awidth, awidth, aseg);
	nb = split_azimuth_interval(bcol*bwidth, bwidth, bseg);
	for (i = 0; i < na; i++)
		for (j = 0; j < nb; j++) {
			const double lo = aseg[i][0] > bseg[j][0] ?
					aseg[i][0] : bseg[j][0];
			const double hi = aseg[i][1] < bseg[j][1] ?
					aseg[i][1] : bseg[j][1];
			if (hi > lo)
				az_overlap += hi - lo;
		}
	if (az_overlap <= FTINY)
		return(0);
	{
		const double s0 = sin(alt0);
		const double s1 = sin(alt1);
		return(.5*az_overlap*(s1*s1 - s0*s0));
	}
}

static int
make_energy_conserving_upsampler(CM_WINDOW_GROUP_STATE *state,
		const int nlow, const int nhigh)
{
	double *weight_sum = NULL;
	int h, l, count = 0;

	state->upsample_offset = (int *)malloc((nhigh + 1)*sizeof(int));
	if (!state->upsample_offset)
		goto memerr;
	for (h = 0; h < nhigh; h++) {
		state->upsample_offset[h] = count;
		for (l = 0; l < nlow; l++)
			if (get_win_patch_projected_overlap(state->low, l,
					state->high, h) > FTINY)
				count++;
	}
	state->upsample_offset[nhigh] = count;
	state->upsample_low_patch = (int *)malloc(count*sizeof(int));
	state->upsample_weight = (double *)malloc(count*sizeof(double));
	weight_sum = (double *)calloc(nlow, sizeof(double));
	if (!state->upsample_low_patch || !state->upsample_weight || !weight_sum)
		goto memerr;
	count = 0;
	for (h = 0; h < nhigh; h++)
		for (l = 0; l < nlow; l++) {
			const double overlap = get_win_patch_projected_overlap(
					state->low, l, state->high, h);
			const double low_omega = get_win_patch_projected_solid_angle(
					state->low, l);

			if (overlap <= FTINY)
				continue;
			state->upsample_low_patch[count] = l;
			state->upsample_weight[count] = overlap/low_omega;
			weight_sum[l] += state->upsample_weight[count++];
		}
	for (l = 0; l < nlow; l++)
		if (fabs(weight_sum[l] - 1) > 1e-8) {
			fprintf(stderr,
				"%s: non-conserving Reinhart upsample weight for low bin %d: %.12g\n",
				progname, l, weight_sum[l]);
			free(weight_sum);
			return(0);
		}
	free(weight_sum);
	return(1);
memerr:
	free(weight_sum);
	fprintf(stderr, "%s: out of memory building Reinhart upsampler\n",
			progname);
	return(0);
}

static int
get_reinhart_basis(const FVECT apex, FVECT u, FVECT v, FVECT w)
{
	FVECT	ref;

	/*
	 * Rotate local Reinhart +Z to the window exterior normal direction.
	 * Match rfluxmtx's right-handed h=rN basis: udir = up x normal and
	 * vdir = normal x udir.  Since we use the glow polygon normal's
	 * reverse as the exterior top, local +Y is azimuth zero and local +X
	 * is positive azimuth for h=rN.
	 */
	VCOPY(w, apex);
	if (normalize(w) == 0.0)
		return(0);
	ref[0] = ref[1] = 0.0; ref[2] = 1.0;
	if (fabs(DOT(ref, w)) > .98) {
		ref[0] = ref[2] = 0.0; ref[1] = 1.0;
	}
	VSUM(v, ref, w, -DOT(ref, w));
	normalize(v);
	VCROSS(u, v, w);
	normalize(u);
	return(1);
}

static void
orient_reinhart_dir(FVECT target, const FVECT local, const FVECT apex)
{
	FVECT	w, u, v;

	if (!get_reinhart_basis(apex, u, v, w)) {
		target[0] = local[0];
		target[1] = local[1];
		target[2] = local[2];
		return;
	}
	target[0] = local[0]*u[0] + local[1]*v[0] + local[2]*w[0];
	target[1] = local[0]*u[1] + local[1]*v[1] + local[2]*w[1];
	target[2] = local[0]*u[2] + local[1]*v[2] + local[2]*w[2];
}

static int
get_win_direction_patch(const ReinhartWin *win, const FVECT dir,
	const FVECT apex)
{
	FVECT	u, v, w, wd;
	double	local_z, alt, azi, azi_width, azi_adj;
	int	row, col;

	VCOPY(wd, dir);
	if (normalize(wd) == 0.0)
		return(-1);
	if (!get_reinhart_basis(apex, u, v, w)) {
		u[0] = 1.0; u[1] = u[2] = 0.0;
		v[1] = 1.0; v[0] = v[2] = 0.0;
		w[2] = 1.0; w[0] = w[1] = 0.0;
	}
	local_z = DOT(wd, w);
	if (local_z <= 0.0)
		return(-1);
	if (local_z > 1.0)
		local_z = 1.0;
	alt = asin(local_z);
	row = (int)floor(alt / win->ringElevationAngle);
	if (row >= win->rings-1)
		return(win->firstPatchIndex[win->rings-1]);
	if (row < 0)
		row = 0;
	azi = atan2(DOT(wd, u), DOT(wd, v));
	azi_width = 2 * PI / win->patchesPerRow[row];
	azi_adj = fmod(azi + .5*azi_width, 2 * PI);
	if (azi_adj < 0)
		azi_adj += 2 * PI;
	col = (int)floor(azi_adj / azi_width);
	if (col >= win->patchesPerRow[row])
		col = win->patchesPerRow[row] - 1;
	return(win->firstPatchIndex[row] + col);
}

static void
get_oriented_win_patch_direction(const ReinhartWin *win, const int patch,
		const FVECT apex, FVECT target)
{
	FVECT	local;

	get_win_patch_direction(win, patch, local);
	orient_reinhart_dir(target, local, apex);
}

static double
get_sun_half_angle(const CM_SUN *sun)
{
	double	cos_half;

	if (!sun || sun->omega <= FTINY)
		return(0.0);
	cos_half = 1 - sun->omega/(2*PI);
	if (cos_half > 1.0)
		cos_half = 1.0;
	else if (cos_half < -1.0)
		cos_half = -1.0;
	return(acos(cos_half));
}

static int
win_patch_overlaps_sun(const FVECT patch_dir, const double patch_radius,
	const CM_SUN *sun)
{
	const double	limit = patch_radius + get_sun_half_angle(sun) + 1e-7;

	if (!sun)
		return(0);
	if (limit >= PI)
		return(1);
	return(DOT(patch_dir, sun->dir) >= cos(limit));
}

static char *
cm_get_radword(char *s, int n, FILE *fp)
{
	int	quote = '\0';
	int	c;
	char	*cp;

	do {
		c = getc(fp);
		if (c == '#')
			while ((c = getc(fp)) != EOF && c != '\n')
				;
	} while (isspace(c));
	if (c == EOF)
		return(NULL);
	if ((c == '"') | (c == '\'')) {
		quote = c;
		c = getc(fp);
	}
	cp = s;
	while (c != EOF) {
		if (quote) {
			if (c == quote)
				break;
		} else if (isspace(c)) {
			break;
		} else if (c == '#') {
			while ((c = getc(fp)) != EOF && c != '\n')
				;
			break;
		}
		if (--n <= 0)
			break;
		*cp++ = c;
		c = getc(fp);
	}
	*cp = '\0';
	return(s);
}

static int
cm_polygon_normal(const double *va, int nv, FVECT norm)
{
	FVECT	v1, v2, vc;
	int	i;

	if (nv < 3)
		return(0);
	norm[0] = norm[1] = norm[2] = 0.0;
	v1[0] = va[3] - va[0];
	v1[1] = va[4] - va[1];
	v1[2] = va[5] - va[2];
	for (i = 2; i < nv; i++) {
		v2[0] = va[3*i] - va[0];
		v2[1] = va[3*i+1] - va[1];
		v2[2] = va[3*i+2] - va[2];
		VCROSS(vc, v1, v2);
		norm[0] += vc[0];
		norm[1] += vc[1];
		norm[2] += vc[2];
		VCOPY(v1, v2);
	}
	return(normalize(norm) != 0.0);
}

int
cm_load_window_dir(FVECT wdir, const char *fspec)
{
	FILE	*fp;
	char	mod[256], typ[256], nam[256], tok[256];
	char	glow_name[128][256];
	int	nglow = 0;
	int	found = 0;
	FVECT	first_norm, glow_norm;

	if (!fspec)
		return(0);
	if ((fp = fopen(fspec, "r")) == NULL) {
		fprintf(stderr, "%s: cannot open window file '%s'\n",
				progname, fspec);
		return(-1);
	}
	while (cm_get_radword(mod, sizeof(mod), fp) != NULL) {
		int	ns, ni, nf, i;
		int	is_glow_poly = 0;
		double	*fa = NULL;

		if (cm_get_radword(typ, sizeof(typ), fp) == NULL ||
				cm_get_radword(nam, sizeof(nam), fp) == NULL)
			goto badfmt;
		if (cm_get_radword(tok, sizeof(tok), fp) == NULL)
			goto badfmt;
		ns = atoi(tok);
		for (i = 0; i < ns; i++)
			if (cm_get_radword(tok, sizeof(tok), fp) == NULL)
				goto badfmt;
		if (cm_get_radword(tok, sizeof(tok), fp) == NULL)
			goto badfmt;
		ni = atoi(tok);
		for (i = 0; i < ni; i++)
			if (cm_get_radword(tok, sizeof(tok), fp) == NULL)
				goto badfmt;
		if (cm_get_radword(tok, sizeof(tok), fp) == NULL)
			goto badfmt;
		nf = atoi(tok);
		if (!strcmp(typ, "polygon") && nf >= 9 && nf % 3 == 0) {
			int j;
			FVECT this_norm;
			fa = (double *)malloc(sizeof(double)*nf);
			if (fa == NULL)
				goto memerr;
			for (i = 0; i < nf; i++) {
				if (cm_get_radword(tok, sizeof(tok), fp) == NULL) {
					free(fa);
					goto badfmt;
				}
				fa[i] = atof(tok);
			}
			for (j = 0; j < nglow; j++)
				if (!strcmp(mod, glow_name[j])) {
					is_glow_poly = 1;
					break;
				}
			if (cm_polygon_normal(fa, nf/3, this_norm) == 0) {
				free(fa);
				continue;
			}
			free(fa);
			if (is_glow_poly) {
				VCOPY(glow_norm, this_norm);
				found = 2;
				break;
			}
			if (!found) {
				VCOPY(first_norm, this_norm);
				found = 1;
			}
		} else {
			for (i = 0; i < nf; i++)
				if (cm_get_radword(tok, sizeof(tok), fp) == NULL)
					goto badfmt;
		}
		if (!strcmp(typ, "glow") && nglow < 128)
			strcpy(glow_name[nglow++], nam);
	}
	fclose(fp);
	if (!found) {
		fprintf(stderr, "%s: no polygon found in window file '%s'\n",
				progname, fspec);
		return(-1);
	}
	if (found == 2)
		VCOPY(wdir, glow_norm);
	else
		VCOPY(wdir, first_norm);
	wdir[0] = -wdir[0];
	wdir[1] = -wdir[1];
	wdir[2] = -wdir[2];
	return(normalize(wdir) == 0.0 ? -1 : 0);
badfmt:
	fclose(fp);
	fprintf(stderr, "%s: bad Radiance window file '%s'\n", progname, fspec);
	return(-1);
memerr:
	fclose(fp);
	fprintf(stderr, "%s: out of memory reading window file '%s'\n",
			progname, fspec);
	return(-1);
}

CM_SUN*
cm_load_suns(const char *fspec, int *nsuns)
{
	FILE	*fp;
	char	mod[256], typ[256], nam[256], tok[256];
	CM_SUN	*suns = NULL;
	int	nalloc = 0;
	int	nfound = 0;

	if (nsuns)
		*nsuns = 0;
	if (!fspec)
		return(NULL);
	if ((fp = fopen(fspec, "r")) == NULL) {
		fprintf(stderr, "%s: cannot open suns file '%s'\n",
				progname, fspec);
		return(NULL);
	}
	while (cm_get_radword(mod, sizeof(mod), fp) != NULL) {
		int	ns, ni, nf, i;
		FVECT	sdir;
		double	angle = 0.0;

		if (cm_get_radword(typ, sizeof(typ), fp) == NULL ||
				cm_get_radword(nam, sizeof(nam), fp) == NULL)
			goto badfmt;
		if (cm_get_radword(tok, sizeof(tok), fp) == NULL)
			goto badfmt;
		ns = atoi(tok);
		for (i = 0; i < ns; i++)
			if (cm_get_radword(tok, sizeof(tok), fp) == NULL)
				goto badfmt;
		if (cm_get_radword(tok, sizeof(tok), fp) == NULL)
			goto badfmt;
		ni = atoi(tok);
		for (i = 0; i < ni; i++)
			if (cm_get_radword(tok, sizeof(tok), fp) == NULL)
				goto badfmt;
		if (cm_get_radword(tok, sizeof(tok), fp) == NULL)
			goto badfmt;
		nf = atoi(tok);
		sdir[0] = sdir[1] = sdir[2] = 0.0;
		for (i = 0; i < nf; i++) {
			double	v;
			if (cm_get_radword(tok, sizeof(tok), fp) == NULL)
				goto badfmt;
			v = atof(tok);
			if (!strcmp(typ, "source")) {
				if (i < 3)
					sdir[i] = v;
				else if (i == 3)
					angle = v;
			}
		}
		if (strcmp(typ, "source") || nf < 4)
			continue;
		if (normalize(sdir) == 0.0)
			continue;
		if (nfound >= nalloc) {
			CM_SUN	*new_suns;
			nalloc = nalloc ? 2*nalloc : 128;
			new_suns = (CM_SUN *)realloc(suns,
					sizeof(CM_SUN)*nalloc);
			if (!new_suns)
				goto memerr;
			suns = new_suns;
		}
		VCOPY(suns[nfound].dir, sdir);
		angle *= PI/180.0;
		suns[nfound].omega = 2*PI*(1 - cos(.5*angle));
		nfound++;
	}
	fclose(fp);
	if (!nfound) {
		fprintf(stderr, "%s: no solar source found in suns file '%s'\n",
				progname, fspec);
		free(suns);
		return(NULL);
	}
	if (nsuns)
		*nsuns = nfound;
	return(suns);
badfmt:
	fclose(fp);
	fprintf(stderr, "%s: bad Radiance suns file '%s'\n", progname, fspec);
	free(suns);
	return(NULL);
memerr:
	fclose(fp);
	fprintf(stderr, "%s: out of memory reading suns file '%s'\n",
			progname, fspec);
	free(suns);
	return(NULL);
}



static void free_sky(ReinhartSky *sky)
{
	if (sky) {
		free(sky->patchesPerRow);
		free(sky->firstPatchIndex);
		free(sky->solidAngle);
		free(sky);
	}
}

static void get_patch_direction(const ReinhartSky *sky, const int patch, FVECT target)
{
	//if (patch >= sky->patches || patch < 0) throw new RuntimeException("Illegal patch " + patch);
	if (!patch) {
		target[0] = target[1] = 0;
		target[2] = -1;
		return; // Ignore below horizon?
	}
	int row = sky->rings - 1;
	while (patch < sky->firstPatchIndex[row]) row--;
	const double alt = PI / 2 - sky->ringElevationAngle * (row - (sky->rings - 1));
	const double azi = 2 * PI * (patch - sky->firstPatchIndex[row]) / sky->patchesPerRow[row];
	const double cos_alt = cos(alt);
	target[0] = cos_alt * -sin(azi);
	target[1] = cos_alt * -cos(azi);
	target[2] = sin(alt);
}

static double get_patch_solid_angle(const ReinhartSky *sky, const int patch, const double cos_theta)
{
	//if (patch >= sky->patches || patch < 0) throw new RuntimeException("Illegal patch " + patch);
	if (!patch) return 2 * (PI - 2 * acos(cos_theta)); // Solid angle overlap between visible hemisphere and ground hemisphere
	int row = sky->rings - 1;
	while (patch < sky->firstPatchIndex[row]) row--;
	return sky->solidAngle[row];
}



static double get_guth(const FVECT dir, const FVECT forward, const FVECT up)
{
	double posindex;
	FVECT hv, temp, vup;
	double sigma, tau;

	VCOPY(vup, up);
	VSUM(vup, vup, forward, -DOT(vup, forward));
	if (normalize(vup) == 0.0) {
		vup[0] = vup[1] = 0.0; vup[2] = 1.0;
		VSUM(vup, vup, forward, -DOT(vup, forward));
		if (normalize(vup) == 0.0) {
			vup[0] = 0.0; vup[1] = 1.0; vup[2] = 0.0;
			VSUM(vup, vup, forward, -DOT(vup, forward));
			normalize(vup);
		}
	}
	VCROSS(hv, forward, vup);
	normalize(hv);
	VCROSS(temp, forward, hv);
	double phi = ANGLE(dir, temp) - PI / 2;

	/* Match evalglare's eccentricity and azimuth construction. */
	sigma = ANGLE(dir, forward);
	VSUM(hv, dir, forward, -DOT(dir, forward));
	if (normalize(hv) == 0.0)
		tau = 0.0;
	else
		tau = ANGLE(hv, vup);

	/* Guth model, equation from IES lighting handbook */
	if (phi >= 0) {
		tau *= 180.0 / PI;
		sigma *= 180.0 / PI;

		if (sigma <= 0)
			sigma = -sigma;

		posindex = exp((35.2 - 0.31889 * tau - 1.22 * exp(-2.0 * tau / 9.0)) / 1000.0 * sigma + (21.0 + 0.26667 * tau - 0.002963 * tau * tau) / 100000.0 * sigma * sigma);
	}
	/* Below line of sight: modified Iwata model used by evalglare. */
	else {
		const double beta = atan(tan(sigma) *
				sqrt(1.0 + 0.3225*cos(tau)*cos(tau))) * 180.0 / PI;
		posindex = exp(6.49/1000.0*beta +
				21.0/100000.0*beta*beta);
	}
	if (posindex > 16)
		posindex = 16.0;

	return posindex;
}

static double
get_vertical_illuminance(const CMATRIX *evmx, const int p, const int t)
{
	const COLORV	*ev = cm_lval(evmx, p, t);

	if (evmx->ncomp == 1)
		return(ev[0]);
	return(LUMINOUS_EFFICACY * bright(ev));
}

static double
get_matrix_illuminance(const CMATRIX *mx, const int p, const int t)
{
	const COLORV	*cv = cm_lval(mx, p, t);

	if (mx->ncomp == 1)
		return(cv[0]);
	return(LUMINOUS_EFFICACY * bright(cv));
}

static double
get_luminous_product(const CMATRIX *m1, const int r1, const int c1,
	const CMATRIX *m2, const int r2, const int c2)
{
	const COLORV	*a = cm_lval(m1, r1, c1);
	const COLORV	*b = cm_lval(m2, r2, c2);
	COLOR		prod;

	prod[0] = a[0] * b[0];
	prod[1] = a[1] * b[1];
	prod[2] = a[2] * b[2];
	return(LUMINOUS_EFFICACY * bright(prod));
}

float* cm_glare(const CMATRIX *dcmx, const CMATRIX *evmx, const CMATRIX *cmtx, const int *occupied, const double dgp_limit, const double dgp_threshold, const FVECT *views, const FVECT dir, const FVECT up, const FVECT wdir)
{
	int p, t, c;
	int hourly_output = dgp_limit < 0;
	float *dgp_list;
	//ReinhartSky *sky;
	ReinhartWin* win;
	FVECT vdir;

	/* Check consistancy */
	if ((dcmx->nrows != evmx->nrows) | (dcmx->ncols != cmtx->nrows) | (evmx->ncols != cmtx->ncols)) {
		fprintf(stderr,
			"%s: inconsistant matrix dimensions: dc(%d, %d) ev(%d, %d) s(%d, %d)\n",
			progname, dcmx->nrows, dcmx->ncols, evmx->nrows, evmx->ncols, cmtx->nrows, cmtx->ncols);
		return NULL;
	}

	/* Create output buffer */
	dgp_list = (float*)malloc(evmx->nrows * (hourly_output ? evmx->ncols : 1) * sizeof(float));
	if (!dgp_list) {
		fprintf(stderr,
			"%s: out of memory in cm_glare()\n",
			progname);
		return NULL;
	}

	/* Create sky */
	//sky = make_sky(smx);
	win = make_win(cmtx);

	if (!win) return NULL;

	/* Calculate glare limit */
	double ev_max = -1;
	if (!hourly_output) {
		ev_max = (dgp_limit - 0.159) / 5.87e-5;
		if (ev_max < 0) ev_max = 0;
	}

	/* For each position and direction */
	if (!views) VCOPY(vdir, dir);
	for (p = 0; p < evmx->nrows; p++) {
		/* For each time step */
		int occupied_hours = 0;
		int glare_hours = 0;
		if (views) VCOPY(vdir, views[p]);
		for (t = 0; t < evmx->ncols; t++) {
			if (!occupied[t]) {
				/* Not occupied */
				if (hourly_output) dgp_list[p * evmx->ncols + t] = 0.0f;
			}
			else {
				/* Occupied */
				double illum = get_vertical_illuminance(evmx, p, t);
				double illum_safe = (illum > 1e-6) ? illum : 1e-6;
				occupied_hours++;

				if ((illum >= ev_max) & (!hourly_output)) {
					/* Guarangeed glare */
					glare_hours++;
				}
				else {
					/* Calculate enhanced simplified daylight glare probability */
					double sum = 0.0;
					FVECT patch_normal;
					for (c = 0; c < cmtx->nrows; c++) {
						const double dc = bright(cm_lval(dcmx, p, c));
						if (dc > 0) {


							get_oriented_win_patch_direction(win, c, wdir, patch_normal);

							const double cos_theta = DOT(vdir, patch_normal);
							if (cos_theta <= FTINY) continue;
							const double omega = get_win_patch_solid_angle(win, c);
							const double patch_luminance =
									get_luminous_product(dcmx, p, c,
											cmtx, c, t) / (omega * cos_theta);

							double min_patch_luminance = dgp_threshold;
							if (dgp_threshold < LUMINANCE_THRESHOLD)
								min_patch_luminance *= illum / PI; // TODO should use average luminance, not illuminance
							if (patch_luminance < min_patch_luminance) continue;
							const double P = get_guth(patch_normal, vdir, up);
							sum += (patch_luminance * patch_luminance * omega) / (P * P);
						}
					}

					double eDGPs = 5.87e-5 * illum_safe + 0.092 * log10(1 + sum / pow(illum_safe, 1.87)) + 0.159;
					//eDGPs /= 1.1 - 0.5 * age / 100.0; /* age correction */
					if (eDGPs > 1.0) eDGPs = 1.0;

					if (hourly_output)
						dgp_list[p * evmx->ncols + t] = (float)eDGPs;
					else if (eDGPs >= dgp_limit)
						glare_hours++;
				}
			}
		}
		if (!hourly_output) {
			/* Save glare autonomy */
			dgp_list[p] = (float)(occupied_hours - glare_hours) / occupied_hours;
		}
	}

	free_win(win);

	return dgp_list;
}

static void
free_window_group_state(CM_WINDOW_GROUP_STATE *state)
{
	if (!state)
		return;
	free_win(state->low);
	free_win(state->high);
	free(state->low_patch_dir);
	free(state->high_patch_dir);
	free(state->upsample_offset);
	free(state->upsample_low_patch);
	free(state->upsample_weight);
	free(state->low_flux);
}

float*
cm_glare_reinhart_replace(const CMATRIX *v1, const CMATRIX *tds1,
	const CMATRIX *v2, const CMATRIX *tds2,
	const CMATRIX *v_residual, const CMATRIX *tds_residual,
	const CMATRIX *evmx,
	const CMATRIX *visible_sun, const CMATRIX *specular_contrast,
	const CM_SUN *suns, const int nsuns,
	const int *occupied, const double dgp_limit,
	const double dgp_threshold, const FVECT *views, const FVECT dir,
	const FVECT up, const FVECT wdir)
{
	CM_WINDOW_GROUP group;

	group.v1 = v1;
	group.tds1 = tds1;
	group.v2 = v2;
	group.tds2 = tds2;
	group.v_residual = v_residual;
	group.tds_residual = tds_residual;
	VCOPY(group.wdir, wdir);
	return(cm_glare_reinhart_groups(&group, 1, evmx, visible_sun,
			specular_contrast, suns, nsuns, occupied, dgp_limit,
			dgp_threshold, views, dir, up));
}

float*
cm_glare_reinhart_groups(const CM_WINDOW_GROUP *groups, const int ngroups,
	const CMATRIX *evmx, const CMATRIX *visible_sun,
	const CMATRIX *specular_contrast, const CM_SUN *suns,
	const int nsuns, const int *occupied, const double dgp_limit,
	const double dgp_threshold, const FVECT *views, const FVECT dir,
	const FVECT up)
{
	int		p, t, c, g;
	const int	hourly_output = dgp_limit < 0;
	const int	have_suns = (suns != NULL) && (nsuns > 0);
	float		*dgp_list = NULL;
	CM_WINDOW_GROUP_STATE *states = NULL;
	double		ev_max = -1;
	FVECT		vdir;

	if (!groups || ngroups <= 0 || !evmx || !occupied)
		return(NULL);
	if ((suns != NULL || nsuns > 0) && !have_suns) {
		fprintf(stderr, "%s: invalid annual sun-direction input\n", progname);
		return(NULL);
	}
	if (visible_sun && !have_suns) {
		fprintf(stderr, "%s: visible-sun matrix requires suns.rad\n", progname);
		return(NULL);
	}
	if (visible_sun && nsuns < evmx->ncols) {
		fprintf(stderr,
			"%s: inconsistent sun directions: suns(%d) Ev(%d,%d)\n",
			progname, nsuns, evmx->nrows, evmx->ncols);
		return(NULL);
	}
	if (visible_sun && ((visible_sun->nrows != evmx->nrows) |
			(visible_sun->ncols != evmx->ncols))) {
		fprintf(stderr,
			"%s: inconsistent visible-sun dimensions: sun(%d,%d) Ev(%d,%d)\n",
			progname, visible_sun->nrows, visible_sun->ncols,
			evmx->nrows, evmx->ncols);
		return(NULL);
	}
	if (specular_contrast && ((specular_contrast->nrows != evmx->nrows) |
			(specular_contrast->ncols != evmx->ncols) |
			(specular_contrast->ncomp != 1))) {
		fprintf(stderr,
			"%s: specular-contrast matrix must be scalar and match Ev: contrast(%d,%d,%d) Ev(%d,%d)\n",
			progname, specular_contrast->nrows, specular_contrast->ncols,
			specular_contrast->ncomp, evmx->nrows, evmx->ncols);
		return(NULL);
	}
	states = (CM_WINDOW_GROUP_STATE *)calloc(ngroups,
			sizeof(CM_WINDOW_GROUP_STATE));
	if (!states)
		goto memerr;
	for (g = 0; g < ngroups; g++) {
		const CM_WINDOW_GROUP *wg = &groups[g];
		CM_WINDOW_GROUP_STATE *state = &states[g];
		const int have_residual = (wg->v_residual != NULL) |
				(wg->tds_residual != NULL);

		state->group = wg;
		if (!wg->v1 || !wg->tds1 || !wg->v2 || !wg->tds2) {
			fprintf(stderr, "%s: window group %d is incomplete\n",
					progname, g + 1);
			goto fail;
		}
		if (have_residual && (!wg->v_residual || !wg->tds_residual)) {
			fprintf(stderr,
				"%s: window group %d requires both fine residual matrices\n",
				progname, g + 1);
			goto fail;
		}
		if ((wg->v1->nrows != evmx->nrows) |
				(wg->v2->nrows != evmx->nrows) |
				(wg->v1->ncols != wg->tds1->nrows) |
				(wg->v2->ncols != wg->tds2->nrows) |
				(wg->tds1->ncols != evmx->ncols) |
				(wg->tds2->ncols != evmx->ncols) |
				(wg->tds1->nrows != wg->tds2->nrows)) {
			fprintf(stderr,
				"%s: inconsistent dimensions in window group %d: Vtotal(%d,%d) TDStotal(%d,%d) Vdirect(%d,%d) TDSdirect(%d,%d) Ev(%d,%d)\n",
				progname, g + 1,
				wg->v1->nrows, wg->v1->ncols,
				wg->tds1->nrows, wg->tds1->ncols,
				wg->v2->nrows, wg->v2->ncols,
				wg->tds2->nrows, wg->tds2->ncols,
				evmx->nrows, evmx->ncols);
			goto fail;
		}
		if (have_residual &&
				((wg->v_residual->nrows != evmx->nrows) |
				(wg->v_residual->ncols != wg->tds_residual->nrows) |
				(wg->tds_residual->ncols != evmx->ncols) |
				(wg->tds_residual->nrows <= wg->tds1->nrows))) {
			fprintf(stderr,
				"%s: inconsistent fine residual dimensions in window group %d: Vresidual(%d,%d) TDSresidual(%d,%d) coarse bins=%d Ev(%d,%d)\n",
				progname, g + 1,
				wg->v_residual->nrows, wg->v_residual->ncols,
				wg->tds_residual->nrows, wg->tds_residual->ncols,
				wg->tds1->nrows, evmx->nrows, evmx->ncols);
			goto fail;
		}
		state->low = make_win(wg->tds1);
		if (!state->low)
			goto memerr;
		state->low_patch_dir = (FVECT *)malloc(
				sizeof(FVECT)*wg->tds1->nrows);
		if (!state->low_patch_dir)
			goto memerr;
		for (c = 0; c < wg->tds1->nrows; c++)
			get_oriented_win_patch_direction(state->low, c,
					wg->wdir, state->low_patch_dir[c]);
		if (have_residual) {
			state->high = make_win(wg->tds_residual);
			state->high_patch_dir = (FVECT *)malloc(
					sizeof(FVECT)*wg->tds_residual->nrows);
			state->low_flux = (double *)malloc(
					wg->tds1->nrows*sizeof(double));
			if (!state->high || !state->high_patch_dir || !state->low_flux)
				goto memerr;
			for (c = 0; c < wg->tds_residual->nrows; c++)
				get_oriented_win_patch_direction(state->high, c,
						wg->wdir, state->high_patch_dir[c]);
			if (!make_energy_conserving_upsampler(state,
					wg->tds1->nrows, wg->tds_residual->nrows))
				goto fail;
		}
	}
	dgp_list = (float*)malloc(evmx->nrows *
			(hourly_output ? evmx->ncols : 1) * sizeof(float));
	if (!dgp_list)
		goto memerr;
	if (!hourly_output) {
		ev_max = (dgp_limit - 0.159) / 5.87e-5;
		if (ev_max < 0)
			ev_max = 0;
	}
	if (!views)
		VCOPY(vdir, dir);
	for (p = 0; p < evmx->nrows; p++) {
		int occupied_hours = 0;
		int glare_hours = 0;
		if (views)
			VCOPY(vdir, views[p]);
		for (t = 0; t < evmx->ncols; t++) {
			if (!occupied[t]) {
				if (hourly_output)
					dgp_list[p * evmx->ncols + t] = 0.0f;
				continue;
			}
			occupied_hours++;
			{
				const double illum = get_vertical_illuminance(evmx, p, t);
				const double illum_safe = (illum > 1e-6) ? illum : 1e-6;
				double sum = 0.0;
				double Esun = 0.0, cos_sun = 0.0;
				int use_sun_term = 0;

				if ((illum >= ev_max) & (!hourly_output)) {
					glare_hours++;
					continue;
				}
				if (specular_contrast) {
					const double spec_sum = cm_lval(specular_contrast, p, t)[0];

					if (!isfinite(spec_sum)) {
						fprintf(stderr,
							"%s: non-finite specular contrast at viewpoint %d, time %d\n",
							progname, p, t);
						goto fail;
					}
					if (spec_sum > 0.0)
						sum += spec_sum;
				}
				if (visible_sun) {
					Esun = get_matrix_illuminance(visible_sun, p, t);
					cos_sun = DOT(vdir, suns[t].dir);
					use_sun_term = (Esun > FTINY) &
							(cos_sun > FTINY) &
							(suns[t].omega > FTINY);
				}
				for (g = 0; g < ngroups; g++) {
					const CM_WINDOW_GROUP *wg = &groups[g];
					CM_WINDOW_GROUP_STATE *state = &states[g];
					FVECT patch_normal;

					if (state->high) {
						int h;

						for (c = 0; c < wg->tds1->nrows; c++)
							state->low_flux[c] =
								get_luminous_product(wg->v1, p, c,
									wg->tds1, c, t) -
								get_luminous_product(wg->v2, p, c,
									wg->tds2, c, t);
						for (h = 0; h < wg->tds_residual->nrows; h++) {
							double cos_theta, omega, denom, phi, L, P;
							int k;

							VCOPY(patch_normal, state->high_patch_dir[h]);
							cos_theta = DOT(vdir, patch_normal);
							if (cos_theta <= FTINY)
								continue;
							omega = get_win_patch_solid_angle(state->high, h);
							/* Project the source bin onto the observer's view plane. */
							denom = omega*cos_theta;
							if (denom <= FTINY)
								continue;
							phi = get_luminous_product(wg->v_residual, p, h,
									wg->tds_residual, h, t);
							for (k = state->upsample_offset[h];
									k < state->upsample_offset[h+1]; k++)
								phi += state->low_flux[
									state->upsample_low_patch[k]] *
									state->upsample_weight[k];
							L = phi/denom;
							if (L <= dgp_threshold)
								continue;
							P = get_guth(patch_normal, vdir, up);
							sum += (L*L*omega)/(P*P);
						}
					} else
						for (c = 0; c < wg->tds1->nrows; c++) {
							double cos_theta, omega, denom;
							double phi1, phi2, L, P;

							VCOPY(patch_normal, state->low_patch_dir[c]);
							cos_theta = DOT(vdir, patch_normal);
							if (cos_theta <= FTINY)
								continue;
							omega = get_win_patch_solid_angle(state->low, c);
							denom = omega*cos_theta;
							if (denom <= FTINY)
								continue;
							phi1 = get_luminous_product(wg->v1, p, c,
									wg->tds1, c, t);
							phi2 = get_luminous_product(wg->v2, p, c,
									wg->tds2, c, t);
							L = (phi1 - phi2)/denom;
							if (L <= dgp_threshold)
								continue;
							P = get_guth(patch_normal, vdir, up);
							sum += (L*L*omega)/(P*P);
						}
				}
				if (use_sun_term) {
					const double Lsun = Esun /
							(suns[t].omega * cos_sun);
					if (Lsun > dgp_threshold) {
						const double P = get_guth(suns[t].dir, vdir, up);
						sum += (Lsun * Lsun * suns[t].omega) / (P * P);
					}
				}
				{
					double eDGPs = 5.87e-5 * illum_safe +
						0.092 * log10(1 + sum /
						pow(illum_safe, 1.87)) + 0.159;
					if (eDGPs > 1.0)
						eDGPs = 1.0;
					if (hourly_output)
						dgp_list[p * evmx->ncols + t] = (float)eDGPs;
					else if (eDGPs >= dgp_limit)
						glare_hours++;
				}
			}
		}
		if (!hourly_output)
			dgp_list[p] = occupied_hours ?
				(float)(occupied_hours - glare_hours) / occupied_hours : 0.0f;
	}
	for (g = 0; g < ngroups; g++)
		free_window_group_state(&states[g]);
	free(states);
	return(dgp_list);
memerr:
	fprintf(stderr, "%s: out of memory in cm_glare_reinhart_groups()\n",
			progname);
fail:
	if (states) {
		for (g = 0; g < ngroups; g++)
			free_window_group_state(&states[g]);
	}
	free(states);
	free(dgp_list);
	return(NULL);
}

static int getvec(FVECT vec, const int dtype, FILE *fp)		/* get a vector from fp */
{
	static float  vf[3];
	static double  vd[3];
	char  buf[32];
	int  i;

	switch (dtype) {
	case DTascii:
		for (i = 0; i < 3; i++) {
			if (fgetword(buf, sizeof(buf), fp) == NULL ||
				!isflt(buf))
				return(-1);
			vec[i] = atof(buf);
		}
		break;
	case DTfloat:
		if (getbinary(vf, sizeof(float), 3, fp) != 3)
			return(-1);
		VCOPY(vec, vf);
		break;
	case DTdouble:
		if (getbinary(vd, sizeof(double), 3, fp) != 3)
			return(-1);
		VCOPY(vec, vd);
		break;
	default:
		fprintf(stderr,
			"%s: botched input format\n",
			progname);
		return(-1);
	}
	return(0);
}

int cm_load_schedule(const int count, int* schedule, FILE *fp)
{
	char buf[512];
	char *cp;
	char *comma;
	double val;
	int i = 0;

	while (fgetline(buf, sizeof(buf), fp) != NULL) {
		if (buf[0] == '#') continue; // Comment line
		comma = NULL;
		for (cp = buf; *cp; cp++) {
			/* If there are multiple commas, assume the value is after the last comma */
			if (*cp == ',') {
				comma = cp; /* Record position of last comma */
			}
		}
		if (comma)
			val = atof(comma + 1);
		else
			val = atof(buf);

		if (i < count) {
			/* Add the value to the schedule */
			schedule[i++] = (val > 0);
		}
		else {
			fprintf(stderr,
				"%s: too many schedule entries\n",
				progname);
			return(1);
		}
	}
	fclose(fp);

	if (i < count) {
		fprintf(stderr,
			"%s: too few schedule entries\n",
			progname);
		return(-1);
	}

	return 0;
}

static int
cm_load_rview_dirs(FVECT *views, const int nrows, FILE *fp)
{
	FVECT	*dirs;
	char	word[256];
	int	ndir = 0, cap = 0;
	int	i;

	if (nrows <= 0)
		return(0);
	if (fseek(fp, 0L, SEEK_SET) < 0)
		return(-1);
	dirs = (FVECT *)malloc(sizeof(FVECT)*nrows);
	if (!dirs)
		return(-1);
	while (cm_get_radword(word, sizeof(word), fp) != NULL) {
		if (strcmp(word, "-vd"))
			continue;
		if (ndir >= nrows) {
			free(dirs);
			return(-1);
		}
		for (i = 0; i < 3; i++) {
			if (cm_get_radword(word, sizeof(word), fp) == NULL) {
				free(dirs);
				return(-1);
			}
			dirs[ndir][i] = atof(word);
		}
		if (normalize(dirs[ndir]) == 0.0) {
			free(dirs);
			return(-1);
		}
		ndir++;
		cap = ndir;
	}
	if (!cap) {
		free(dirs);
		return(-1);
	}
	if (cap == 1) {
		for (i = 0; i < nrows; i++)
			VCOPY(views[i], dirs[0]);
	} else if (cap == nrows) {
		for (i = 0; i < nrows; i++)
			VCOPY(views[i], dirs[i]);
	} else {
		fprintf(stderr,
			"%s: view file has %d -vd entries for %d view rows\n",
			progname, cap, nrows);
		free(dirs);
		return(-1);
	}
	free(dirs);
	return(0);
}

FVECT* cm_load_views(const int nrows, const int dtype, FILE *fp)
{
	int i;
	double d;
	FVECT orig;

	FVECT *views = (FVECT*)malloc(nrows * sizeof(FVECT));
	if (!views) {
		fprintf(stderr,
			"%s: out of memory in cm_load_views()\n",
			progname);
		return NULL;
	}

	for (i = 0; i < nrows; i++) {
		if (getvec(orig, dtype, fp) | getvec(views[i], dtype, fp)) {
			if ((dtype == DTascii) & (i == 0) &&
					(cm_load_rview_dirs(views, nrows, fp) == 0))
				return views;
			fprintf(stderr,
				"%s: unexpected end of input, missing %d entries\n",
				progname, i);
			free(views);
			return NULL;
		}

		d = normalize(views[i]);
		if (d == 0.0) {				/* zero ==> flush */
			fprintf(stderr,
				"%s: zero length direction detected\n",
				progname);
			free(views);
			return NULL;
		}
	}

	return views;
}

int cm_write_glare(const float *mp, const int nrows, const int ncols, const int dtype, FILE *fp)
{
	static const char	tabEOL[2] = { '\t', '\n' };
	int			r, c;
	double	dc[1];

	switch (dtype) {
	case DTascii:
		for (r = 0; r < nrows; r++)
			for (c = 0; c < ncols; c++, mp++)
				fprintf(fp, "%.6e%c",
				mp[0],
				tabEOL[c >= ncols - 1]);
		break;
	case DTfloat:
		r = ncols*nrows;
		while (r > 0) {
			c = putbinary(mp, sizeof(float), r, fp);
			if (c <= 0)
				return(0);
			mp += c;
			r -= c;
		}
		break;
	case DTdouble:
		r = ncols*nrows;
		while (r--) {
			dc[0] = mp[0];
			if (putbinary(dc, sizeof(double), 1, fp) != 1)
				return(0);
			mp++;
		}
		break;
	default:
		fputs("Unsupported data type in cm_write_glare()!\n", stderr);
		return(0);
	}
	return(fflush(fp) == 0);
}
