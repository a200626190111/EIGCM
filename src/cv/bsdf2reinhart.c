#ifndef lint
static const char RCSid[] = "$Id$";
#endif
/*
 * Resample an XML BSDF transmission component into a Reinhart matrix.
 */

#define _USE_MATH_DEFINES
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"
#include "paths.h"
#include "random.h"
#include "rtio.h"
#include "color.h"
#include "bsdf.h"

#define T_NALT 7

enum OutputType {
	OUT_ASCII,
	OUT_FLOAT,
	OUT_DOUBLE
};

enum TransmissionSide {
	TRANS_BACK,
	TRANS_FRONT
};

enum ComponentMode {
	COMPONENT_ALL,
	COMPONENT_DIFFUSE,
	COMPONENT_NON_DIFFUSE
};

typedef struct {
	int	row;
	int	col;
	int	nazi;
	double	alt0;
	double	alt1;
	double	azi0;
	double	azi1;
	double	proj_sa;
} RBin;

static const int treg_nazi[T_NALT] = {30, 30, 24, 24, 18, 12, 6};

static int	mf_in = 1;
static int	mf_out = 1;
static int	nsamples = 16;
static unsigned long sample_seed = 1;
static int	quiet = 0;
static int	renormalize = 1;
static enum OutputType output_type = OUT_FLOAT;
static enum TransmissionSide trans_side = TRANS_BACK;
static enum ComponentMode component_mode = COMPONENT_ALL;

static void
usage(void)
{
	fprintf(stderr,
		"Usage: %s [-q] [-fa|-ff|-fd] [-mf N | -mi N -mo N] "
		"[-n samples] [-s seed] [-tb|-tf] [-E] "
		"[-C all|diffuse|non-diffuse] bsdf.xml [output.mtx]\n",
		progname);
	fputs("  -tb  exterior-to-interior (Transmission Back, default)\n", stderr);
	fputs("  -tf  interior-to-exterior (Transmission Front)\n", stderr);
	fputs("  -C   select complete, Lambertian diffuse, or non-diffuse transmission\n",
			stderr);
	fputs("  -E   disable column energy normalization\n", stderr);
}

static const char *
component_name(void)
{
	switch (component_mode) {
	case COMPONENT_DIFFUSE:
		return "diffuse";
	case COMPONENT_NON_DIFFUSE:
		return "non-diffuse";
	default:
		return "all";
	}
}

static int
set_component_mode(const char *name)
{
	if (!strcmp(name, "all"))
		component_mode = COMPONENT_ALL;
	else if (!strcmp(name, "diffuse"))
		component_mode = COMPONENT_DIFFUSE;
	else if (!strcmp(name, "non-diffuse") || !strcmp(name, "nondiffuse"))
		component_mode = COMPONENT_NON_DIFFUSE;
	else
		return 0;
	return 1;
}

static int
reinhart_nbins(int mf)
{
	return 144*mf*mf + 1;
}

static int
row_nazi(int mf, int row)
{
	if (row >= T_NALT*mf)
		return 1;
	return mf*treg_nazi[row/mf];
}

static int
get_rbin(RBin *rb, int mf, int bin)
{
	const int top_row = T_NALT*mf;
	const double row_height = (.5*M_PI)/(top_row + .5);
	int row = 0;
	int col = bin;
	int nazi;
	double z0, z1;

	if ((rb == NULL) | (mf <= 0) | (bin < 0) |
			(bin >= reinhart_nbins(mf)))
		return 0;
	while (col >= (nazi = row_nazi(mf, row))) {
		col -= nazi;
		++row;
	}
	rb->row = row;
	rb->col = col;
	rb->nazi = nazi;
	rb->alt0 = row*row_height;
	rb->alt1 = (row >= top_row) ? .5*M_PI : (row+1)*row_height;
	rb->azi0 = 2.*M_PI*(col-.5)/nazi;
	rb->azi1 = 2.*M_PI*(col+.5)/nazi;
	z0 = sin(rb->alt0);
	z1 = sin(rb->alt1);
	rb->proj_sa = (rb->azi1-rb->azi0)*.5*(z1*z1-z0*z0);
	return 1;
}

/* Sample uniformly with respect to projected solid angle. */
static void
sample_rbin(FVECT v, const RBin *rb, double u0, double u1)
{
	const double z0 = sin(rb->alt0);
	const double z1 = sin(rb->alt1);
	const double z = sqrt(z0*z0 + u0*(z1*z1-z0*z0));
	const double azi = rb->azi0 + u1*(rb->azi1-rb->azi0);
	const double r = sqrt(fmax(.0, 1.-z*z));

	v[0] = sin(azi)*r;
	v[1] = -cos(azi)*r;
	v[2] = z;
}

static void
orient_vectors(FVECT vin, FVECT vout)
{
	if (trans_side == TRANS_BACK) {
		vin[0] = -vin[0];
		vin[1] = -vin[1];
		vin[2] = -vin[2];
	} else {
		vin[0] = -vin[0];
		vin[1] = -vin[1];
		vout[2] = -vout[2];
	}
}

static uint32_t
mix32(uint32_t x)
{
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	return x ^ (x >> 16);
}

static double
radical_inverse(unsigned n, unsigned base)
{
	double value = .0;
	double scale = 1./base;

	while (n) {
		value += (n%base)*scale;
		n /= base;
		scale /= base;
	}
	return value;
}

static double
qsample(int sample, unsigned base, int bin, int dimension)
{
	uint32_t h = mix32((uint32_t)sample_seed ^
			(uint32_t)(bin+1)*0x9e3779b9U ^
			(uint32_t)(dimension+1)*0x85ebca6bU);
	double value = radical_inverse((unsigned)sample+1, base) +
			((double)h+.5)*(1./4294967296.);

	return value - floor(value);
}

static const SDValue *
diffuse_transmission(const FVECT vin, const SDData *bsdf)
{
	return vin[2] > .0 ? &bsdf->tLambFront : &bsdf->tLambBack;
}

static const SDValue *
selected_diffuse_transmission(const SDData *bsdf)
{
	return trans_side == TRANS_BACK ?
			&bsdf->tLambBack : &bsdf->tLambFront;
}

static int
eval_component(COLOR cval, double *ycoef, const FVECT vin, const FVECT vout,
		const SDData *bsdf)
{
	SDValue full, diffuse;
	COLOR diffuse_rgb;
	SDError ec;

	diffuse = *diffuse_transmission(vin, bsdf);
	diffuse.cieY *= 1./M_PI;
	if (component_mode == COMPONENT_DIFFUSE) {
		*ycoef = diffuse.cieY;
		if (diffuse.cieY <= .0) {
			cval[0] = cval[1] = cval[2] = .0;
			return 1;
		}
		ccy2rgb(&diffuse.spec, diffuse.cieY, cval);
		return 1;
	}
	ec = SDevalBSDF(&full, vin, vout, bsdf);
	if (ec != SDEnone) {
		SDreportError(ec, stderr);
		return 0;
	}
	*ycoef = full.cieY;
	if (full.cieY <= .0) {
		cval[0] = cval[1] = cval[2] = .0;
		return 1;
	}
	ccy2rgb(&full.spec, full.cieY, cval);
	if (component_mode == COMPONENT_ALL)
		return 1;
	*ycoef -= diffuse.cieY;
	if (*ycoef <= 1e-12) {
		*ycoef = .0;
		cval[0] = cval[1] = cval[2] = .0;
		return 1;
	}
	if (diffuse.cieY > .0) {
		ccy2rgb(&diffuse.spec, diffuse.cieY, diffuse_rgb);
		cval[0] -= diffuse_rgb[0];
		cval[1] -= diffuse_rgb[1];
		cval[2] -= diffuse_rgb[2];
	}
	return 1;
}

static int
eval_cell(float rgb[3], double *ycoef, int ibin, int obin,
		const RBin *ib, const RBin *ob, const SDData *bsdf)
{
	double sum_rgb[3] = {.0, .0, .0};
	double sum_y = .0;
	int s;

	if (component_mode == COMPONENT_DIFFUSE) {
		SDValue diffuse = *selected_diffuse_transmission(bsdf);
		COLOR cval;

		diffuse.cieY *= 1./M_PI;
		if (diffuse.cieY <= .0) {
			rgb[0] = rgb[1] = rgb[2] = .0f;
			*ycoef = .0;
			return 1;
		}
		ccy2rgb(&diffuse.spec, diffuse.cieY, cval);
		rgb[0] = (float)(cval[0]*ib->proj_sa);
		rgb[1] = (float)(cval[1]*ib->proj_sa);
		rgb[2] = (float)(cval[2]*ib->proj_sa);
		*ycoef = diffuse.cieY*ib->proj_sa;
		return 1;
	}

	for (s = 0; s < nsamples; ++s) {
		FVECT vin, vout;
		COLOR cval;
		double yval;

		sample_rbin(vin, ib, qsample(s, 2, ibin, 0),
				qsample(s, 3, ibin, 1));
		sample_rbin(vout, ob, qsample(s, 5, obin, 2),
				qsample(s, 7, obin, 3));
		orient_vectors(vin, vout);
		if (!eval_component(cval, &yval, vin, vout, bsdf))
			return 0;
		if (yval <= .0)
			continue;
		sum_rgb[0] += cval[0];
		sum_rgb[1] += cval[1];
		sum_rgb[2] += cval[2];
		sum_y += yval;
	}
	rgb[0] = (float)(sum_rgb[0]/nsamples*ib->proj_sa);
	rgb[1] = (float)(sum_rgb[1]/nsamples*ib->proj_sa);
	rgb[2] = (float)(sum_rgb[2]/nsamples*ib->proj_sa);
	*ycoef = sum_y/nsamples*ib->proj_sa;
	return 1;
}

static double
reference_transmittance(int ibin, const RBin *ib, const SDData *bsdf)
{
	double sum = .0;
	int s;

	if (component_mode == COMPONENT_DIFFUSE)
		return selected_diffuse_transmission(bsdf)->cieY;

	for (s = 0; s < nsamples; ++s) {
		FVECT vin, dummy = {.0, .0, 1.};

		sample_rbin(vin, ib, qsample(s, 2, ibin, 0),
				qsample(s, 3, ibin, 1));
		orient_vectors(vin, dummy);
		switch (component_mode) {
		case COMPONENT_NON_DIFFUSE:
			sum += SDdirectHemi(vin, SDsampSpT, bsdf);
			break;
		default:
			sum += SDdirectHemi(vin,
					SDsampDf | SDsampSp | SDsampT, bsdf);
			break;
		}
	}
	return sum/nsamples;
}

static int
write_matrix(FILE *fp, const float *mtx, int nrows, int ncols,
		int argc, char *argv[])
{
	size_t nvals = (size_t)nrows*ncols*3;
	size_t i;

	newheader("RADIANCE", fp);
	printargs(argc, argv, fp);
	fprintf(fp, "NROWS=%d\nNCOLS=%d\nNCOMP=3\n", nrows, ncols);
	switch (output_type) {
	case OUT_ASCII:
		fputformat("ascii", fp);
		fputc('\n', fp);
		for (i = 0; i < nvals; i += 3)
			fprintf(fp, "%.7e %.7e %.7e%c", mtx[i], mtx[i+1],
					mtx[i+2], ((i/3+1)%ncols) ? '\t' : '\n');
		break;
	case OUT_DOUBLE:
		fputformat("double", fp);
		fputc('\n', fp);
		for (i = 0; i < nvals; ++i) {
			double v = mtx[i];
			if (putbinary(&v, sizeof(v), 1, fp) != 1)
				return 0;
		}
		break;
	default:
		fputformat("float", fp);
		fputc('\n', fp);
		if (putbinary(mtx, sizeof(float), nvals, fp) != nvals)
			return 0;
		break;
	}
	return fflush(fp) == 0;
}

int
main(int argc, char *argv[])
{
	const char *xml_path;
	const char *out_path = NULL;
	FILE *out = stdout;
	SDData bsdf;
	SDError ec;
	RBin *ibins = NULL, *obins = NULL;
	float *mtx = NULL;
	double *raw_tau = NULL, *ref_tau = NULL;
	int ninc, nout;
	int a, i, o;
	int rc = 1;

	fixargv0(argv[0]);
	for (a = 1; a < argc && argv[a][0] == '-'; ++a) {
		if (!strcmp(argv[a], "-q"))
			quiet = 1;
		else if (!strcmp(argv[a], "-fa"))
			output_type = OUT_ASCII;
		else if (!strcmp(argv[a], "-ff"))
			output_type = OUT_FLOAT;
		else if (!strcmp(argv[a], "-fd"))
			output_type = OUT_DOUBLE;
		else if (!strcmp(argv[a], "-tb"))
			trans_side = TRANS_BACK;
		else if (!strcmp(argv[a], "-tf"))
			trans_side = TRANS_FRONT;
		else if (!strcmp(argv[a], "-E"))
			renormalize = 0;
		else if ((!strcmp(argv[a], "-C") ||
				!strcmp(argv[a], "--component")) && a+1 < argc) {
			if (!set_component_mode(argv[++a])) {
				fprintf(stderr, "%s: unknown BSDF component '%s'\n",
						progname, argv[a]);
				return 1;
			}
		}
		else if (!strcmp(argv[a], "-mf") && a+1 < argc)
			mf_in = mf_out = atoi(argv[++a]);
		else if (!strcmp(argv[a], "-mi") && a+1 < argc)
			mf_in = atoi(argv[++a]);
		else if (!strcmp(argv[a], "-mo") && a+1 < argc)
			mf_out = atoi(argv[++a]);
		else if (!strcmp(argv[a], "-n") && a+1 < argc)
			nsamples = atoi(argv[++a]);
		else if (!strcmp(argv[a], "-s") && a+1 < argc)
			sample_seed = strtoul(argv[++a], NULL, 10);
		else if (!strcmp(argv[a], "-h") || !strcmp(argv[a], "--help")) {
			usage();
			return 0;
		} else {
			usage();
			return 1;
		}
	}
	if ((mf_in <= 0) | (mf_out <= 0) | (nsamples <= 0) |
			(argc-a < 1) | (argc-a > 2)) {
		usage();
		return 1;
	}
	xml_path = argv[a++];
	if (a < argc)
		out_path = argv[a];
	ninc = reinhart_nbins(mf_in);
	nout = reinhart_nbins(mf_out);
	if ((size_t)nout*ninc > ((size_t)-1)/(3*sizeof(float))) {
		fprintf(stderr, "%s: requested matrix is too large\n", progname);
		return 1;
	}

	SDclearBSDF(&bsdf, xml_path);
	ec = SDloadFile(&bsdf, xml_path);
	if (ec != SDEnone) {
		SDreportError(ec, stderr);
		return 1;
	}
	if (trans_side == TRANS_BACK) {
		if ((bsdf.tb == NULL) && (bsdf.tf == NULL) &&
				(bsdf.tLambBack.cieY <= .0)) {
			fprintf(stderr, "%s: BSDF has no transmission component\n", progname);
			goto cleanup;
		}
	} else if ((bsdf.tf == NULL) && (bsdf.tb == NULL) &&
			(bsdf.tLambFront.cieY <= .0)) {
		fprintf(stderr, "%s: BSDF has no transmission component\n", progname);
		goto cleanup;
	}

	ibins = (RBin *)malloc((size_t)ninc*sizeof(RBin));
	obins = (RBin *)malloc((size_t)nout*sizeof(RBin));
	mtx = (float *)calloc((size_t)nout*ninc*3, sizeof(float));
	raw_tau = (double *)calloc(ninc, sizeof(double));
	ref_tau = (double *)calloc(ninc, sizeof(double));
	if ((ibins == NULL) | (obins == NULL) | (mtx == NULL) |
			(raw_tau == NULL) | (ref_tau == NULL)) {
		fprintf(stderr, "%s: out of memory\n", progname);
		goto cleanup;
	}
	for (i = 0; i < ninc; ++i)
		get_rbin(&ibins[i], mf_in, i);
	for (o = 0; o < nout; ++o)
		get_rbin(&obins[o], mf_out, o);

	for (o = 0; o < nout; ++o) {
		if (!quiet && (!(o % ((nout+19)/20)) || o == nout-1)) {
			fprintf(stderr, "\r%s: resampling %d/%d rows", progname, o+1, nout);
			fflush(stderr);
		}
		for (i = 0; i < ninc; ++i) {
			float *dst = mtx + 3*((size_t)o*ninc+i);
			double ycoef;
			if (!eval_cell(dst, &ycoef, i, o, &ibins[i], &obins[o], &bsdf))
				goto cleanup;
			raw_tau[i] += ycoef*obins[o].proj_sa/ibins[i].proj_sa;
		}
	}
	if (!quiet)
		fputc('\n', stderr);

	for (i = 0; i < ninc; ++i)
		ref_tau[i] = reference_transmittance(i, &ibins[i], &bsdf);
	if (renormalize) {
		for (i = 0; i < ninc; ++i) {
			double scale;
			if (raw_tau[i] <= 1e-12) {
				if (ref_tau[i] > 1e-8 && !quiet)
					fprintf(stderr,
						"%s: warning - incident bin %d missed nonzero transmission %.6g\n",
						progname, i, ref_tau[i]);
				continue;
			}
			scale = ref_tau[i]/raw_tau[i];
			for (o = 0; o < nout; ++o) {
				float *dst = mtx + 3*((size_t)o*ninc+i);
				dst[0] = (float)(dst[0]*scale);
				dst[1] = (float)(dst[1]*scale);
				dst[2] = (float)(dst[2]*scale);
			}
		}
	}
	if (!quiet) {
		double max_raw_err = .0;
		double min_ref = FHUGE, max_ref = -FHUGE;
		int over_unity = 0;
		for (i = 0; i < ninc; ++i) {
			double err = fabs(raw_tau[i]-ref_tau[i]);
			if (err > max_raw_err) max_raw_err = err;
			if (ref_tau[i] < min_ref) min_ref = ref_tau[i];
			if (ref_tau[i] > max_ref) max_ref = ref_tau[i];
			over_unity += ref_tau[i] > 1.0001;
		}
		fprintf(stderr,
			"%s: MF%d -> MF%d, %d x %d, %d samples/cell, component %s\n"
			"%s: source hemispherical transmittance %.6g..%.6g; "
			"max raw energy error %.6g; %d bins above unity\n",
			progname, mf_in, mf_out, nout, ninc, nsamples,
			component_name(),
			progname, min_ref, max_ref, max_raw_err, over_unity);
	}

	if (out_path != NULL) {
		out = fopen(out_path, output_type == OUT_ASCII ? "w" : "wb");
		if (out == NULL) {
			fprintf(stderr, "%s: cannot open '%s': %s\n", progname,
					out_path, strerror(errno));
			goto cleanup;
		}
	} else if (output_type != OUT_ASCII)
		SET_FILE_BINARY(stdout);
	if (!write_matrix(out, mtx, nout, ninc, argc, argv)) {
		fprintf(stderr, "%s: error writing matrix\n", progname);
		goto cleanup;
	}
	rc = 0;

cleanup:
	if (out != stdout && out != NULL)
		fclose(out);
	free(ref_tau);
	free(raw_tau);
	free(mtx);
	free(obins);
	free(ibins);
	SDfreeBSDF(&bsdf);
	return rc;
}
