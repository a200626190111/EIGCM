#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>

#include "platform.h"
#include "fvect.h"
#include "bsdf.h"
#include "bsdf_m.h"
#include "color.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void
vec_from_deg(FVECT v, double theta, double phi)
{
	theta *= M_PI/180.0;
	phi *= M_PI/180.0;
	v[0] = sin(theta)*cos(phi);
	v[1] = sin(theta)*sin(phi);
	v[2] = cos(theta);
}

static unsigned int
hash_u32(unsigned int x)
{
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

static double
rand_host(unsigned int seed)
{
	return (double)((hash_u32(seed) >> 8) + 0.5) *
		(1.0 / 16777216.0);
}

static void
jitter_in_vec(FVECT out, const FVECT in, double sr_psa, unsigned int seed)
{
	VCOPY(out, in);
	if (sr_psa <= FTINY)
		return;
	out[0] += sr_psa * (0.5 - rand_host(seed));
	out[1] += sr_psa * (0.5 - rand_host(seed ^ 0x68bc21ebu));
	normalize(out);
}

static SDSpectralDF *
spectral_df_for_sample(const SDData *sd, const FVECT in, int xmit)
{
	if (in[2] > 0)
		return xmit ? (sd->tf != NULL ? sd->tf : sd->tb) : sd->rf;
	return xmit ? (sd->tb != NULL ? sd->tb : sd->tf) : sd->rb;
}

static SDMat *
matrix_for_sample(const SDData *sd, const FVECT in, int xmit)
{
	SDSpectralDF *df = spectral_df_for_sample(sd, in, xmit);
	int c;

	if (df == NULL)
		return NULL;
	for (c = 0; c < df->ncomp; c++)
		if (df->comp[c].func == &SDhandleMtx)
			return (SDMat *)df->comp[c].dist;
	return NULL;
}

static const char *
ndx_name(b_ndxf *fn)
{
	if (fn == &fo_getndx)
		return "fo";
	if (fn == &fi_getndx)
		return "fi";
	if (fn == &bi_getndx)
		return "bi";
	if (fn == &bo_getndx)
		return "bo";
	return "?";
}

static int
sample_component_dir_weight(FVECT out, double *weight, const FVECT in,
	double randx, int xmit, const SDData *sd)
{
	SDSpectralDF *df = spectral_df_for_sample(sd, in, xmit);
	const SDCDst *cdists[64];
	double total = 0.0, target;
	int c;

	if (weight != NULL)
		*weight = 0.0;
	if (df == NULL || df->ncomp <= 0 || df->ncomp > 64)
		return 0;
	for (c = 0; c < df->ncomp; c++) {
		cdists[c] = (*df->comp[c].func->getCDist)(in, &df->comp[c]);
		if (cdists[c] != NULL && cdists[c]->cTotal > 0.0)
			total += cdists[c]->cTotal;
	}
	if (total <= 0.0)
		return 0;
	target = randx * total;
	for (c = 0; c < df->ncomp; c++) {
		const SDCDst *cd = cdists[c];
		if (cd == NULL || cd->cTotal <= 0.0)
			continue;
		if (target <= cd->cTotal) {
			SDValue sv;
			double local_rand = target / cd->cTotal;
			VCOPY(out, in);
			if (SDsampComponent(&sv, out, local_rand, &df->comp[c]) !=
					SDEnone || sv.cieY <= FTINY)
				return 0;
			if (weight != NULL)
				*weight = sv.cieY;
			return 1;
		}
		target -= cd->cTotal;
	}
	return 0;
}

static int
sample_component_dir(FVECT out, const FVECT in, double randx, int xmit,
	const SDData *sd)
{
	return sample_component_dir_weight(out, NULL, in, randx, xmit, sd);
}

static int
sample_klems_matrix_dir(FVECT out, double *weight, const FVECT in,
	double randx, int xmit, const SDData *sd)
{
	SDMat *dp = matrix_for_sample(sd, in, xmit);
	double total = 0.0, target;
	int inc, rev = 0, n, i;

	if (weight != NULL)
		*weight = 0.0;
	if (dp == NULL)
		return 0;
	if (randx < 0.0)
		randx = 0.0;
	else if (randx >= 1.0)
		randx = 0.999999999999999;
	inc = mBSDF_incndx(dp, in);
	if (inc < 0) {
		inc = mBSDF_outndx(dp, in);
		rev = 1;
	}
	if (inc < 0)
		return 0;
	n = rev ? dp->ninc : dp->nout;
	for (i = 0; i < n; i++) {
		const double ohm = rev ? mBSDF_incohm(dp, i) :
			mBSDF_outohm(dp, i);
		const double val = rev ? mBSDF_value(dp, inc, i) :
			mBSDF_value(dp, i, inc);
		if (ohm > 0.0 && val > 0.0)
			total += val * ohm;
	}
	if (total <= 1.0e-6)
		return 0;
	target = randx * total;
	for (i = 0; i < n; i++) {
		const double ohm = rev ? mBSDF_incohm(dp, i) :
			mBSDF_outohm(dp, i);
		const double val = rev ? mBSDF_value(dp, inc, i) :
			mBSDF_value(dp, i, inc);
		const double seg = (ohm > 0.0 && val > 0.0) ? val * ohm : 0.0;
		if (target <= seg || i + 1 == n) {
			const double local_rand = seg > 0.0 ? target / seg : 0.5;
			if (rev) {
				if (!mBSDF_incvec(out, dp, (double)i + local_rand))
					return 0;
			} else {
				if (!mBSDF_outvec(out, dp, (double)i + local_rand))
					return 0;
			}
			if (weight != NULL)
				*weight = total;
			return 1;
		}
		target -= seg;
	}
	return 0;
}

static int
dir_hits_square(const FVECT v, double z, double half)
{
	double t, x, y;

	if ((z > 0 && v[2] <= FTINY) || (z < 0 && v[2] >= -FTINY))
		return 0;
	t = z / v[2];
	if (t <= 0.0)
		return 0;
	x = t * v[0];
	y = t * v[1];
	return fabs(x) <= half && fabs(y) <= half;
}

static int
dir_hits_rect_from(const FVECT v, double z, double x0, double x1,
	double y0, double y1, double ox, double oy, double oz)
{
	double t, x, y;

	if (((z - oz) > 0 && v[2] <= FTINY) ||
			((z - oz) < 0 && v[2] >= -FTINY))
		return 0;
	t = (z - oz) / v[2];
	if (t <= 0.0)
		return 0;
	x = ox + t * v[0];
	y = oy + t * v[1];
	return x >= x0 && x <= x1 && y >= y0 && y <= y1;
}

static int
dir_hits_rect(const FVECT v, double z, double x0, double x1,
	double y0, double y1)
{
	return dir_hits_rect_from(v, z, x0, x1, y0, y1, 0.0, 0.0, 0.0);
}


static void
crossv(FVECT r, const FVECT a, const FVECT b)
{
	r[0] = a[1]*b[2] - a[2]*b[1];
	r[1] = a[2]*b[0] - a[0]*b[2];
	r[2] = a[0]*b[1] - a[1]*b[0];
}

static void
gpu_like_world_to_local(FVECT out, const FVECT world, const FVECT n,
	const FVECT up)
{
	FVECT x, y, nn, uu;

	VCOPY(nn, n);
	normalize(nn);
	VCOPY(uu, up);
	crossv(x, uu, nn);
	if (DOT(x, x) <= 1.0e-8) {
		uu[0] = 1.0; uu[1] = uu[2] = 0.0;
		crossv(x, uu, nn);
	}
	normalize(x);
	crossv(y, nn, x);
	out[0] = DOT(world, x);
	out[1] = DOT(world, y);
	out[2] = DOT(world, nn);
	normalize(out);
}

typedef struct {
	FVECT tdir;
	double vy;
} PEAKY;

static int
cmp_peaky(const void *a, const void *b)
{
	double d = ((const PEAKY *)b)->vy - ((const PEAKY *)a)->vy;

	return (d > 0) - (d < 0);
}

static SDSpectralDF *
select_trans_df(const SDData *sd, double rod)
{
	if (rod > 0)
		return sd->tf != NULL ? sd->tf : sd->tb;
	return sd->tb != NULL ? sd->tb : sd->tf;
}

static SDSpectralDF *
select_spec_df(const SDData *sd, const FVECT vin, const FVECT vsrc)
{
	switch ((vsrc[2] > 0)<<1 | (vin[2] > 0)) {
	case 3: return sd->rf;
	case 0: return sd->rb;
	case 1: return sd->tf != NULL ? sd->tf : sd->tb;
	default: return sd->tb != NULL ? sd->tb : sd->tf;
	}
}

static double
select_lamb_y(const SDData *sd, const FVECT vin, const FVECT vsrc)
{
	switch ((vsrc[2] > 0)<<1 | (vin[2] > 0)) {
	case 3: return sd->rLambFront.cieY;
	case 0: return sd->rLambBack.cieY;
	case 1: return sd->tLambFront.cieY;
	default: return sd->tLambBack.cieY;
	}
}

static void
compute_through_y(double *peakY, double *surrY, const SDData *sd,
	const FVECT vin, double rod)
{
#define NDIR2CHECK 29
	static const float dir2check[NDIR2CHECK][2] = {
		{0, 0}, {-0.6f, 0}, {0, 0.6f}, {0, -0.6f}, {0.6f, 0},
		{-0.6f, 0.6f}, {-0.6f, -0.6f}, {0.6f, 0.6f},
		{0.6f, -0.6f}, {-1.2f, 0}, {0, 1.2f}, {0, -1.2f},
		{1.2f, 0}, {-1.2f, 1.2f}, {-1.2f, -1.2f},
		{1.2f, 1.2f}, {1.2f, -1.2f}, {-1.8f, 0}, {0, 1.8f},
		{0, -1.8f}, {1.8f, 0}, {-1.8f, 1.8f},
		{-1.8f, -1.8f}, {1.8f, 1.8f}, {1.8f, -1.8f},
		{-2.4f, 0}, {0, 2.4f}, {0, -2.4f}, {2.4f, 0},
	};
	SDSpectralDF *dfp = select_trans_df(sd, rod);
	PEAKY psamp[NDIR2CHECK];
	double srchrad, tomsum = 0.0, tomsurr = 0.0, vypeak = 0.0;
	int i, ns = 0;

	*peakY = *surrY = 0.0;
	if (dfp == NULL)
		return;
	srchrad = sqrt(dfp->minProjSA);
	for (i = 0; i < NDIR2CHECK; i++) {
		SDValue sv;
		psamp[i].tdir[0] = -vin[0] + dir2check[i][0]*srchrad;
		psamp[i].tdir[1] = -vin[1] + dir2check[i][1]*srchrad;
		psamp[i].tdir[2] = -vin[2];
		normalize(psamp[i].tdir);
		if (SDevalBSDF(&sv, vin, psamp[i].tdir, sd) != SDEnone)
			return;
		psamp[i].vy = sv.cieY;
	}
	qsort(psamp, NDIR2CHECK, sizeof(PEAKY), cmp_peaky);
	if (psamp[0].vy <= FTINY)
		return;
	for (i = 0; i < NDIR2CHECK; i++) {
		double tomega;
		if (i && psamp[i].vy == psamp[i-1].vy)
			continue;
		if (SDsizeBSDF(&tomega, vin, psamp[i].tdir, SDqueryMin, sd) !=
				SDEnone)
			return;
		if (tomega > 1.5*dfp->minProjSA || vypeak > 8.*psamp[i].vy*ns) {
			if (!i)
				return;
			*surrY += psamp[i].vy*tomega;
			tomsurr += tomega;
			continue;
		}
		*peakY += psamp[i].vy*tomega;
		tomsum += tomega;
		vypeak += psamp[i].vy;
		++ns;
	}
	if (tomsurr < 0.2*tomsum)
		return;
	*surrY /= tomsurr;
	if (vin[2] > 0) {
		*peakY -= tomsum*sd->tLambFront.cieY/M_PI;
		*surrY -= sd->tLambFront.cieY/M_PI;
	} else {
		*peakY -= tomsum*sd->tLambBack.cieY/M_PI;
		*surrY -= sd->tLambBack.cieY/M_PI;
	}
	if (*peakY < 0.0) *peakY = 0.0;
	if (*surrY < 0.0) *surrY = 0.0;
	if (*peakY < .0005)
		*peakY = *surrY = 0.0;
#undef NDIR2CHECK
}

int
main(int argc, char **argv)
{
	SDData *sd;
	FVECT vin, vout;
	SDValue val;
	int sflags = SDsampAll;
	int n, i;

	if (argc >= 8 && toupper((unsigned char)argv[2][0]) == 'E') {
		FVECT vsrc;
		double omega, ldot, diffY = 0.0, specY, directY;
		SDValue lamb;

		sd = loadBSDF(argv[1]);
		if (sd == NULL)
			return 1;
		vec_from_deg(vin, atof(argv[3]), atof(argv[4]));
		vec_from_deg(vsrc, atof(argv[5]), atof(argv[6]));
		omega = atof(argv[7]);
		if (SDevalBSDF(&val, vin, vsrc, sd) != SDEnone) {
			fprintf(stderr, "eval failed\n");
			return 1;
		}
		switch ((vsrc[2] > 0)<<1 | (vin[2] > 0)) {
		case 3: lamb = sd->rLambFront; break;
		case 0: lamb = sd->rLambBack; break;
		case 1: lamb = sd->tLambFront; break;
		default: lamb = sd->tLambBack; break;
		}
		if (lamb.cieY > 0.0)
			diffY = lamb.cieY / M_PI;
		specY = val.cieY - diffY;
		if (specY < 0.0)
			specY = 0.0;
		ldot = vsrc[2];
		directY = specY * fabs(ldot) * omega;
		printf("evalY %.12g diffuseY_per_sr %.12g specY %.12g cos %.12g omega %.12g directY %.12g\n",
			val.cieY, diffY, specY, fabs(ldot), omega, directY);
		return 0;
	}
	if (argc >= 8 && toupper((unsigned char)argv[2][0]) == 'O') {
		FVECT vsrc;
		double omega, rod, tomega = 0.0, minProjSA = 0.0;
		double dxdy2, overlap_limit = 0.0, diffY, specY = 0.0;
		double peakY, surrY;
		SDSpectralDF *dfp;

		sd = loadBSDF(argv[1]);
		if (sd == NULL)
			return 1;
		vec_from_deg(vin, atof(argv[3]), atof(argv[4]));
		vec_from_deg(vsrc, atof(argv[5]), atof(argv[6]));
		omega = atof(argv[7]);
		rod = argc > 8 ? atof(argv[8]) : vin[2];
		dfp = select_spec_df(sd, vin, vsrc);
		if (dfp != NULL)
			minProjSA = dfp->minProjSA;
		if (SDevalBSDF(&val, vin, vsrc, sd) == SDEnone) {
			diffY = select_lamb_y(sd, vin, vsrc) / M_PI;
			specY = val.cieY - diffY;
			if (specY < 0.0)
				specY = 0.0;
		} else {
			diffY = 0.0;
		}
		(void)SDsizeBSDF(&tomega, vin, vsrc, SDqueryMin, sd);
		compute_through_y(&peakY, &surrY, sd, vin, rod);
		dxdy2 = (vsrc[0] + vin[0])*(vsrc[0] + vin[0]) +
			(vsrc[1] + vin[1])*(vsrc[1] + vin[1]);
		if (dfp != NULL) {
			double somega = omega * fabs(vsrc[2]);
			overlap_limit = (2.5*4.0/M_PI) *
				(somega + dfp->minProjSA +
				2.0*sqrt(somega*dfp->minProjSA));
		}
		printf("evalY %.12g diffuseY_per_sr %.12g specY %.12g "
			"sdMin %.12g dfMinProjSA %.12g throughPeakY %.12g "
			"throughSurrY %.12g dxdy2 %.12g overlapLimit %.12g "
			"overlap %d directY %.12g\n",
			val.cieY, diffY, specY, tomega, minProjSA, peakY, surrY,
			dxdy2, overlap_limit,
			(peakY > FTINY && dfp != NULL && dxdy2 <= overlap_limit),
			specY*fabs(vsrc[2])*omega);
		return 0;
	}
	if (argc >= 5 && toupper((unsigned char)argv[2][0]) == 'H') {
		SDSpectralDF *rfp, *tfp;
		double rLamb, tLamb, rSpec, tSpec;
		double psa[2] = {0.0, 0.0};

		sd = loadBSDF(argv[1]);
		if (sd == NULL)
			return 1;
		vec_from_deg(vin, atof(argv[3]), atof(argv[4]));
		(void)SDsizeBSDF(psa, vin, NULL, SDqueryMin+SDqueryMax, sd);
		if (vin[2] > 0) {
			rfp = sd->rf;
			tfp = sd->tf != NULL ? sd->tf : sd->tb;
			rLamb = sd->rLambFront.cieY;
			tLamb = sd->tLambFront.cieY;
		} else {
			rfp = sd->rb;
			tfp = sd->tb != NULL ? sd->tb : sd->tf;
			rLamb = sd->rLambBack.cieY;
			tLamb = sd->tLambBack.cieY;
		}
		rSpec = SDdirectHemi(vin, SDsampSpR, sd);
		tSpec = SDdirectHemi(vin, SDsampSpT, sd);
		printf("rLamb %.12g rSpecHemi %.12g rTotalGPU %.12g "
			"rfMaxHemi %.12g rfMinProjSA %.12g "
			"tLamb %.12g tSpecHemi %.12g tTotalGPU %.12g "
			"tfMaxHemi %.12g tfMinProjSA %.12g "
			"sdSizeMin %.12g sdSizeMax %.12g srMin %.12g srMax %.12g\n",
			rLamb, rSpec, rLamb + rSpec,
			rfp != NULL ? rfp->maxHemi : 0.0,
			rfp != NULL ? rfp->minProjSA : 0.0,
			tLamb, tSpec, tLamb + tSpec,
			tfp != NULL ? tfp->maxHemi : 0.0,
			tfp != NULL ? tfp->minProjSA : 0.0,
			psa[0], psa[1], sqrt(psa[0]), sqrt(psa[1]));
		return 0;
	}
	if (argc >= 8 && toupper((unsigned char)argv[2][0]) == 'Q') {
		int xmit, n, i, high = 0, low = 0;
		double threshold, psa[2] = {0.0, 0.0};
		double sr_max = 0.0, sum = 0.0, high_sum = 0.0, low_sum = 0.0;
		unsigned int seed_base;

		sd = loadBSDF(argv[1]);
		if (sd == NULL)
			return 1;
		xmit = toupper((unsigned char)argv[3][0]) == 'T';
		n = atoi(argv[4]);
		vec_from_deg(vin, atof(argv[5]), atof(argv[6]));
		threshold = atof(argv[7]);
		seed_base = argc > 8 ? (unsigned int)strtoul(argv[8], NULL, 0) :
			131071u;
		if (SDsizeBSDF(psa, vin, NULL, SDqueryMin+SDqueryMax, sd) ==
				SDEnone)
			sr_max = sqrt(psa[1]);
		for (i = 0; i < n; i++) {
			FVECT vjit;
			double hv;
			jitter_in_vec(vjit, vin, sr_max, seed_base + (unsigned int)i);
			hv = SDdirectHemi(vjit, xmit ? SDsampSpT : SDsampSpR, sd);
			sum += hv;
			if (hv <= threshold + FTINY) {
				low++;
				low_sum += hv;
			} else {
				high++;
				high_sum += hv;
			}
		}
		printf("gateSamples %d comp %c threshold %.12g sdSizeMin %.12g "
			"sdSizeMax %.12g srMax %.12g raw %.12g avg %.12g "
			"low %d high %d lowAvg %.12g highAvg %.12g\n",
			n, xmit ? 'T' : 'R', threshold, psa[0], psa[1], sr_max,
			SDdirectHemi(vin, xmit ? SDsampSpT : SDsampSpR, sd),
			n > 0 ? sum/(double)n : 0.0, low, high,
			low > 0 ? low_sum/(double)low : 0.0,
			high > 0 ? high_sum/(double)high : 0.0);
		return 0;
	}
	if (argc >= 13 && toupper((unsigned char)argv[2][0]) == 'C') {
		int xmit, n, i, high = 0, low = 0;
		unsigned int seed_base;
		double threshold, z, x0, x1, y0, y1;
		double ox = 0.0, oy = 0.0, oz = 0.0;
		double psa[2] = {0.0, 0.0}, sr_min = 0.0, sr_max = 0.0;
		double cpu_sum = 0.0, gpu_sum = 0.0, cpu_hit_sum = 0.0;
		double gpu_hit_sum = 0.0, gate_sum = 0.0;
		int cpu_ok = 0, gpu_ok = 0, cpu_hit = 0, gpu_hit = 0;

		sd = loadBSDF(argv[1]);
		if (sd == NULL)
			return 1;
		xmit = toupper((unsigned char)argv[3][0]) == 'T';
		n = atoi(argv[4]);
		vec_from_deg(vin, atof(argv[5]), atof(argv[6]));
		threshold = atof(argv[7]);
		z = atof(argv[8]);
		x0 = atof(argv[9]);
		x1 = atof(argv[10]);
		y0 = atof(argv[11]);
		y1 = atof(argv[12]);
		seed_base = argc > 13 ? (unsigned int)strtoul(argv[13], NULL, 0) :
			131071u;
		if (argc > 16) {
			ox = atof(argv[14]);
			oy = atof(argv[15]);
			oz = atof(argv[16]);
		}
		if (x1 < x0) {
			double tmp = x0; x0 = x1; x1 = tmp;
		}
		if (y1 < y0) {
			double tmp = y0; y0 = y1; y1 = tmp;
		}
		if (SDsizeBSDF(psa, vin, NULL, SDqueryMin+SDqueryMax, sd) ==
				SDEnone) {
			sr_min = sqrt(psa[0]);
			sr_max = sqrt(psa[1]);
		}
		for (i = 0; i < n; i++) {
			FVECT vgate, vsmp, cpu_dir, gpu_dir;
			double hv, rv, cpu_weight = 0.0, gpu_weight = 0.0;
			const unsigned int si = seed_base ^ ((unsigned int)i * 0x9e3779b9u);

			jitter_in_vec(vgate, vin, sr_max, si ^ 0x94d049bbu);
			hv = SDdirectHemi(vgate, xmit ? SDsampSpT : SDsampSpR, sd);
			gate_sum += hv;
			if (hv <= threshold + FTINY) {
				low++;
				continue;
			}
			high++;
			jitter_in_vec(vsmp, vin, sr_min, si ^ 0xd1b54a35u);
			rv = rand_host(si ^ 0x4cf5ad43u);
			if (sample_component_dir_weight(cpu_dir, &cpu_weight, vsmp, rv,
					xmit, sd)) {
				cpu_ok++;
				cpu_sum += cpu_weight;
				if (dir_hits_rect_from(cpu_dir, z, x0, x1, y0, y1,
						ox, oy, oz)) {
					cpu_hit++;
					cpu_hit_sum += cpu_weight;
				}
			}
			if (sample_klems_matrix_dir(gpu_dir, &gpu_weight, vsmp, rv,
					xmit, sd)) {
				gpu_ok++;
				gpu_sum += gpu_weight;
				if (dir_hits_rect_from(gpu_dir, z, x0, x1, y0, y1,
						ox, oy, oz)) {
					gpu_hit++;
					gpu_hit_sum += gpu_weight;
				}
			}
		}
		printf("cpuPathSamples %d comp %c threshold %.12g z %.12g "
			"origin %.12g %.12g %.12g rect %.12g %.12g %.12g %.12g "
			"srMin %.12g srMax %.12g low %d high %d gateAvg %.12g "
			"cpuOk %d cpuHits %d cpuAvgWeight %.12g cpuFiniteAvg %.12g "
			"gpuOk %d gpuHits %d gpuAvgWeight %.12g gpuFiniteAvg %.12g\n",
			n, xmit ? 'T' : 'R', threshold, z, ox, oy, oz, x0, x1, y0, y1,
			sr_min, sr_max, low, high, n > 0 ? gate_sum/(double)n : 0.0,
			cpu_ok, cpu_hit, n > 0 ? cpu_sum/(double)n : 0.0,
			n > 0 ? cpu_hit_sum/(double)n : 0.0,
			gpu_ok, gpu_hit, n > 0 ? gpu_sum/(double)n : 0.0,
			n > 0 ? gpu_hit_sum/(double)n : 0.0);
		return 0;
	}
	if (argc >= 9 && toupper((unsigned char)argv[2][0]) == 'P') {
		int xmit, n, i, ok = 0, hit = 0;
		unsigned int seed_base;
		double z, half, sr_vpsa = 0.0;
		double psa[2];

		sd = loadBSDF(argv[1]);
		if (sd == NULL)
			return 1;
		xmit = toupper((unsigned char)argv[3][0]) == 'T';
		n = atoi(argv[4]);
		vec_from_deg(vin, atof(argv[5]), atof(argv[6]));
		z = atof(argv[7]);
		half = atof(argv[8]);
		seed_base = argc > 9 ? (unsigned int)strtoul(argv[9], NULL, 0) :
			131071u;
		if (SDsizeBSDF(psa, vin, NULL, SDqueryMin+SDqueryMax, sd) ==
				SDEnone)
			sr_vpsa = sqrt(psa[0]);
		for (i = 0; i < n; i++) {
			FVECT vjit, vsmp;
			double rv = ((double)i + 0.5) / (double)n;
			jitter_in_vec(vjit, vin, sr_vpsa, seed_base + (unsigned int)i);
			if (!sample_component_dir(vsmp, vjit, rv, xmit, sd))
				continue;
			ok++;
			if (dir_hits_square(vsmp, z, half))
				hit++;
		}
		printf("componentSamples %d ok %d hits %d hitFraction %.12g "
			"sdDirectHemi %.12g finiteEstimate %.12g\n",
			n, ok, hit, ok > 0 ? (double)hit/(double)ok : 0.0,
			SDdirectHemi(vin, xmit ? SDsampSpT : SDsampSpR, sd),
			SDdirectHemi(vin, xmit ? SDsampSpT : SDsampSpR, sd) *
			(ok > 0 ? (double)hit/(double)ok : 0.0));
		return 0;
	}
	if (argc >= 12 && toupper((unsigned char)argv[2][0]) == 'K') {
		int xmit, n, i, cpu_ok = 0, cpu_hit = 0;
		int gpu_ok = 0, gpu_hit = 0;
		unsigned int seed_base;
		double z, x0, x1, y0, y1, sr_vpsa = 0.0;
		double ox = 0.0, oy = 0.0, oz = 0.0;
		double psa[2], gpu_wsum = 0.0, gpu_whit = 0.0;

		sd = loadBSDF(argv[1]);
		if (sd == NULL)
			return 1;
		xmit = toupper((unsigned char)argv[3][0]) == 'T';
		n = atoi(argv[4]);
		vec_from_deg(vin, atof(argv[5]), atof(argv[6]));
		z = atof(argv[7]);
		x0 = atof(argv[8]);
		x1 = atof(argv[9]);
		y0 = atof(argv[10]);
		y1 = atof(argv[11]);
		seed_base = argc > 12 ? (unsigned int)strtoul(argv[12], NULL, 0) :
			131071u;
		if (argc > 15) {
			ox = atof(argv[13]);
			oy = atof(argv[14]);
			oz = atof(argv[15]);
		}
		if (x1 < x0) {
			double tmp = x0; x0 = x1; x1 = tmp;
		}
		if (y1 < y0) {
			double tmp = y0; y0 = y1; y1 = tmp;
		}
		if (SDsizeBSDF(psa, vin, NULL, SDqueryMin+SDqueryMax, sd) ==
				SDEnone)
			sr_vpsa = sqrt(psa[0]);
		for (i = 0; i < n; i++) {
			FVECT vjit, cpu_dir, gpu_dir;
			double rv = ((double)i + 0.5) / (double)n;
			double gpu_weight = 0.0;
			jitter_in_vec(vjit, vin, sr_vpsa, seed_base + (unsigned int)i);
			if (sample_component_dir(cpu_dir, vjit, rv, xmit, sd)) {
				cpu_ok++;
				if (dir_hits_rect_from(cpu_dir, z, x0, x1, y0, y1,
						ox, oy, oz))
					cpu_hit++;
			}
			if (sample_klems_matrix_dir(gpu_dir, &gpu_weight, vjit, rv,
					xmit, sd)) {
				gpu_ok++;
				gpu_wsum += gpu_weight;
				if (dir_hits_rect_from(gpu_dir, z, x0, x1, y0, y1,
						ox, oy, oz)) {
					gpu_hit++;
					gpu_whit += gpu_weight;
				}
			}
		}
		printf("rectSamples %d comp %c z %.12g origin %.12g %.12g %.12g "
			"rect %.12g %.12g %.12g %.12g "
			"cpuOk %d cpuHits %d cpuHitFraction %.12g "
			"gpuOk %d gpuHits %d gpuHitFraction %.12g "
			"sdDirectHemi %.12g cpuFiniteEstimate %.12g "
			"gpuAvgWeight %.12g gpuFiniteAvg %.12g\n",
			n, xmit ? 'T' : 'R', z, ox, oy, oz, x0, x1, y0, y1,
			cpu_ok, cpu_hit,
			cpu_ok > 0 ? (double)cpu_hit/(double)cpu_ok : 0.0,
			gpu_ok, gpu_hit,
			gpu_ok > 0 ? (double)gpu_hit/(double)gpu_ok : 0.0,
			SDdirectHemi(vin, xmit ? SDsampSpT : SDsampSpR, sd),
			SDdirectHemi(vin, xmit ? SDsampSpT : SDsampSpR, sd) *
			(cpu_ok > 0 ? (double)cpu_hit/(double)cpu_ok : 0.0),
			gpu_ok > 0 ? gpu_wsum/(double)gpu_ok : 0.0,
			n > 0 ? gpu_whit/(double)n : 0.0);
		return 0;
	}
	if (argc >= 8 && toupper((unsigned char)argv[2][0]) == 'M') {
		int xmit, inc, o, nout, hits = 0;
		double hemi = 0.0, finite = 0.0, z, half;
		SDMat *dp;

		sd = loadBSDF(argv[1]);
		if (sd == NULL)
			return 1;
		xmit = toupper((unsigned char)argv[3][0]) == 'T';
		vec_from_deg(vin, atof(argv[4]), atof(argv[5]));
		z = atof(argv[6]);
		half = atof(argv[7]);
		dp = matrix_for_sample(sd, vin, xmit);
		if (dp == NULL) {
			fprintf(stderr, "no matrix component\n");
			return 1;
		}
		inc = mBSDF_incndx(dp, vin);
		if (inc < 0)
			inc = mBSDF_outndx(dp, vin);
		if (inc < 0) {
			fprintf(stderr, "incident index failed\n");
			return 1;
		}
		nout = dp->nout;
		for (o = 0; o < nout; o++) {
			FVECT vo;
			double contrib = mBSDF_value(dp, o, inc) *
				mBSDF_outohm(dp, o);
			(void)mBSDF_outvec(vo, dp, o + 0.5);
			hemi += contrib;
			if (dir_hits_square(vo, z, half)) {
				finite += contrib;
				hits++;
			}
		}
		printf("inc %d nout %d ib %s ob %s hits %d matrixHemi %.12g finiteSum %.12g "
			"finiteFraction %.12g sdDirectHemi %.12g\n",
			inc, nout, ndx_name(dp->ib_ndx), ndx_name(dp->ob_ndx),
			hits, hemi, finite,
			hemi > 0.0 ? finite/hemi : 0.0,
			SDdirectHemi(vin, xmit ? SDsampSpT : SDsampSpR, sd));
		return 0;
	}
	if (argc >= 11 && toupper((unsigned char)argv[2][0]) == 'X') {
		FVECT world, nrm, upv, gpuv, radv;
		RREAL mtx[3][3];

		vec_from_deg(world, atof(argv[3]), atof(argv[4]));
		nrm[0] = atof(argv[5]); nrm[1] = atof(argv[6]); nrm[2] = atof(argv[7]);
		upv[0] = atof(argv[8]); upv[1] = atof(argv[9]); upv[2] = atof(argv[10]);
		gpu_like_world_to_local(gpuv, world, nrm, upv);
		if (SDcompXform(mtx, nrm, upv) != SDEnone ||
				SDmapDir(radv, mtx, world) != SDEnone) {
			fprintf(stderr, "xform failed\n");
			return 1;
		}
		printf("gpuLocal %.12g %.12g %.12g radLocal %.12g %.12g %.12g "
			"delta %.12g %.12g %.12g\n",
			gpuv[0], gpuv[1], gpuv[2], radv[0], radv[1], radv[2],
			gpuv[0]-radv[0], gpuv[1]-radv[1], gpuv[2]-radv[2]);
		return 0;
	}
	if (argc >= 8 && toupper((unsigned char)argv[2][0]) == 'G') {
		double z, half, cell_area, sumY = 0.0;
		int ns, ix, iy, used = 0;

		sd = loadBSDF(argv[1]);
		if (sd == NULL)
			return 1;
		vec_from_deg(vin, atof(argv[3]), atof(argv[4]));
		z = atof(argv[5]);
		half = atof(argv[6]);
		ns = atoi(argv[7]);
		if (ns <= 0 || half <= 0.0 || z == 0.0)
			return 2;
		cell_area = (2.0*half/ns) * (2.0*half/ns);
		for (iy = 0; iy < ns; iy++) {
			for (ix = 0; ix < ns; ix++) {
				double x = -half + ((double)ix + 0.5) * 2.0*half/ns;
				double y = -half + ((double)iy + 0.5) * 2.0*half/ns;
				double r2 = x*x + y*y + z*z;
				double invr = 1.0 / sqrt(r2);
				double mu_panel, mu_emit, omega, diffY = 0.0, specY;
				SDValue lamb;

				vout[0] = x*invr;
				vout[1] = y*invr;
				vout[2] = z*invr;
				if (SDevalBSDF(&val, vin, vout, sd) != SDEnone)
					continue;
				switch ((vout[2] > 0)<<1 | (vin[2] > 0)) {
				case 3: lamb = sd->rLambFront; break;
				case 0: lamb = sd->rLambBack; break;
				case 1: lamb = sd->tLambFront; break;
				default: lamb = sd->tLambBack; break;
				}
				if (lamb.cieY > 0.0)
					diffY = lamb.cieY / M_PI;
				specY = val.cieY - diffY;
				if (specY < 0.0)
					specY = 0.0;
				mu_panel = fabs(vout[2]);
				mu_emit = fabs(z) * invr;
				omega = mu_emit * cell_area / r2;
				sumY += specY * mu_panel * omega;
				used++;
			}
		}
		printf("gridY %.12g samples %d z %.12g half %.12g n %d\n",
			sumY, used, z, half, ns);
		return 0;
	}
	if (argc < 7) {
		fprintf(stderr,
			"usage: %s bsdf.xml R|T|A D|S|A n theta phi\n"
			"       %s bsdf.xml E theta_in phi_in theta_src phi_src omega\n"
			"       %s bsdf.xml O theta_in phi_in theta_src phi_src omega [rod]\n"
			"       %s bsdf.xml H theta_in phi_in\n"
			"       %s bsdf.xml C R|T n theta_in phi_in threshold z x0 x1 y0 y1 [seed_base [ox oy oz]]\n"
			"       %s bsdf.xml P R|T n theta_in phi_in emitter_z emitter_half [seed_base]\n"
			"       %s bsdf.xml K R|T n theta_in phi_in z x0 x1 y0 y1 [seed_base [ox oy oz]]\n"
			"       %s bsdf.xml M R|T theta_in phi_in emitter_z emitter_half\n"
			"       %s bsdf.xml X world_theta world_phi nx ny nz ux uy uz\n"
			"       %s bsdf.xml G theta_in phi_in emitter_z emitter_half grid_n\n",
			argv[0], argv[0], argv[0], argv[0], argv[0], argv[0],
			argv[0], argv[0], argv[0], argv[0]);
		return 2;
	}
	sd = loadBSDF(argv[1]);
	if (sd == NULL)
		return 1;
	switch (toupper((unsigned char)argv[2][0])) {
	case 'R': sflags &= ~SDsampT; break;
	case 'T': sflags &= ~SDsampR; break;
	}
	switch (toupper((unsigned char)argv[3][0])) {
	case 'D': sflags &= ~SDsampSp; break;
	case 'S': sflags &= ~SDsampDf; break;
	}
	n = atoi(argv[4]);
	vec_from_deg(vin, atof(argv[5]), atof(argv[6]));
	for (i = 0; i < n; i++) {
		double rx = ((double)i + 0.5) / (double)n;
		VCOPY(vout, vin);
		if (SDsampBSDF(&val, vout, rx, sflags, sd) != SDEnone) {
			fprintf(stderr, "sample failed at %d\n", i);
			return 1;
		}
		printf("%.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
			vout[0], vout[1], vout[2],
			val.cieY, val.spec.cx, val.spec.cy, rx);
	}
	return 0;
}
