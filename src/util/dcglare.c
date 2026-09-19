#ifndef lint
static const char RCSid[] = "$Id: dcglare.c,v 2.10 2025/06/07 05:09:46 greg Exp $";
#endif
/*
 * Compute time-step glare using imageless DGP calculation method.
 *
 *	N. Jones
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

#include <ctype.h>
#include "platform.h"
#include "standard.h"
#include "cmatrix.h"
#include "resolu.h"
#include "cmglare.h"

/* Sum together a set of images and write result to fout */
static int
sum_images(const char *fspec, const CMATRIX *cv, FILE *fout)
{
	int	myDT = DTfromHeader;
	COLR	*scanline = NULL;
	CMATRIX	*pmat = NULL;
	int	myXR=0, myYR=0;
	int	i, y;

	if (cv->ncols != 1)
		error(INTERNAL, "expected vector in sum_images()");
	for (i = 0; i < cv->nrows; i++) {
		const COLORV	*scv = cv_lval(cv,i);
		int		flat_file = 0;
		char		fname[1024];
		FILE		*fp;
		long		data_start;
		int		dt, xr, yr;
		COLORV		*psp;
		char		*err;
							/* check for zero */
		if ((scv[RED] == 0) & (scv[GRN] == 0) & (scv[BLU] == 0) &&
				(myDT != DTfromHeader) | (i < cv->nrows-1))
			continue;
							/* open next picture */
		sprintf(fname, fspec, i);
		if ((fp = fopen(fname, "rb")) == NULL) {
			sprintf(errmsg, "cannot open picture '%s'", fname);
			error(SYSTEM, errmsg);
		}
		dt = DTfromHeader;
		if ((err = cm_getheader(&dt, NULL, NULL, NULL, NULL, fp)) != NULL)
			error(USER, err);
		if ((dt != DTrgbe) & (dt != DTxyze) ||
				!fscnresolu(&xr, &yr, fp)) {
			sprintf(errmsg, "file '%s' not a picture", fname);
			error(USER, errmsg);
		}
		if (myDT == DTfromHeader) {		/* on first one */
			myDT = dt;
			myXR = xr; myYR = yr;
			scanline = (COLR *)malloc(sizeof(COLR)*myXR);
			if (scanline == NULL)
				error(SYSTEM, "out of memory in sum_images()");
			pmat = cm_alloc(myYR, myXR);
			memset(pmat->cmem, 0, sizeof(COLOR)*myXR*myYR);
							/* finish header */
			fputformat(cm_fmt_id[myDT], fout);
			fputc('\n', fout);
			fflush(fout);
		} else if ((dt != myDT) | (xr != myXR) | (yr != myYR)) {
			sprintf(errmsg, "picture '%s' format/size mismatch",
					fname);
			error(USER, errmsg);
		}
							/* flat file check */
		if ((data_start = ftell(fp)) > 0 && fseek(fp, 0L, SEEK_END) == 0) {
			flat_file = (ftell(fp) >= data_start + sizeof(COLR)*xr*yr);
			if (fseek(fp, data_start, SEEK_SET) < 0) {
				sprintf(errmsg, "cannot seek on picture '%s'", fname);
				error(SYSTEM, errmsg);
			}
		}
		psp = pmat->cmem;
		for (y = 0; y < yr; y++) {		/* read it in */
			COLOR	col;
			int	x;
			if (flat_file ? getbinary(scanline, sizeof(COLR), xr, fp) != xr :
					freadcolrs(scanline, xr, fp) < 0) {
				sprintf(errmsg, "error reading picture '%s'",
						fname);
				error(SYSTEM, errmsg);
			}
							/* sum in scanline */
			for (x = 0; x < xr; x++, psp += 3) {
				if (!scanline[x][EXP])
					continue;	/* skip zeroes */
				colr_color(col, scanline[x]);
				multcolor(col, scv);
				addcolor(psp, col);
			}
		}
		fclose(fp);				/* done this picture */
	}
	free(scanline);
	i = cm_write(pmat, myDT, fout);			/* write picture */
	cm_free(pmat);					/* free data */
	return(i);
}

/* check to see if a string contains a %d or %o specification */
static int
hasNumberFormat(const char *s)
{
	if (s == NULL)
		return(0);

	while (*s) {
		while (*s != '%')
			if (!*s++)
				return(0);
		if (*++s == '%') {		/* ignore "%%" */
			++s;
			continue;
		}
		while (isdigit(*s))		/* field length */
			++s;
						/* field we'll use? */
		if ((*s == 'd') | (*s == 'i') | (*s == 'o') |
					(*s == 'x') | (*s == 'X'))
			return(1);
	}
	return(0);				/* didn't find one */
}

#ifdef DC_GLARE
typedef struct window_group_spec_s {
	char	*window_rad_path;
	char	*view_low_total_path;
	char	*tds_low_total_path;
	char	*direct_low_view_path;
	char	*direct_low_path;
	char	*direct_high_view_path;
	char	*direct_high_path;
	char	*direct_high_view_sub_path;
	char	*direct_high_sub_path;
	char	*visible_sun_path;
} WINDOW_GROUP_SPEC;

static WINDOW_GROUP_SPEC *
append_window_group(WINDOW_GROUP_SPEC **groups, int *ngroups)
{
	WINDOW_GROUP_SPEC *res = (WINDOW_GROUP_SPEC *)realloc(*groups,
			(*ngroups + 1) * sizeof(WINDOW_GROUP_SPEC));
	WINDOW_GROUP_SPEC *group;

	if (!res)
		return(NULL);
	*groups = res;
	group = &res[(*ngroups)++];
	memset(group, 0, sizeof(WINDOW_GROUP_SPEC));
	return(group);
}
#endif

int
main(int argc, char *argv[])
{
	int		skyfmt = DTfromHeader;
	int		outfmt = DTascii;
	int		headout = 1;
	int		nsteps = 0;
	char		*ofspec = NULL;
	FILE		*ofp = stdout;
	CMATRIX		*cmtx;		/* component vector/matrix result */
	char		fnbuf[256];
	int		a, i;
#ifdef DC_GLARE
	char	*direct_path = NULL;
	char	*view_low_total_path = NULL;
	char	*tds_low_total_path = NULL;
	char	*direct_low_view_path = NULL;
	char	*direct_low_path = NULL;
	char	*direct_high_view_path = NULL;
	char	*direct_high_path = NULL;
	char	*direct_high_view_sub_path = NULL;
	char	*direct_high_sub_path = NULL;
	char	*window_rad_path = NULL;
	char	*visible_sun_path = NULL;
	char	*suns_rad_path = NULL;
	WINDOW_GROUP_SPEC *window_groups = NULL;
	WINDOW_GROUP_SPEC *current_group = NULL;
	int		nwindow_groups = 0;
	char	*schedule_path = NULL;
	int		*occupancy = NULL;
	int		replacement_mode = 0;
	int		start_hour = 0;
	int		end_hour = 24;
	double	dgp_limit = -1;
	double	dgp_threshold = 2000;
	char	*view_path = NULL;
	FILE	*fp;
	float	*dgp_values = NULL;
	FVECT	vdir, vup;
	FVECT	wdir;
	FVECT	*views = NULL;
	int		viewfmt = DTascii;

	vdir[0] = vdir[1] = vdir[2] = vup[0] = vup[1] = 0;
	vup[2] = 1;
	wdir[0] = 0; wdir[1] = -1; wdir[2] = 0;

	clock_t timer = clock();
#endif /* DC_GLARE */

	fixargv0(argv[0]);
					/* get options */
	for (a = 1; a < argc && argv[a][0] == '-'; a++)
		switch (argv[a][1]) {
		case 'n':
			nsteps = atoi(argv[++a]);
			if (nsteps < 0)
				goto userr;
			skyfmt = nsteps ? DTascii : DTfromHeader;
			break;
		case 'h':
			headout = !headout;
			break;
		case 'i':
			switch (argv[a][2]) {
			case 'f':
				skyfmt = DTfloat;
				break;
			case 'd':
				skyfmt = DTdouble;
				break;
			case 'a':
				skyfmt = DTascii;
				break;
			default:
				goto userr;
			}
			break;
		case 'o':
			switch (argv[a][2]) {
#ifndef DC_GLARE
			case '\0':	/* output specification (not format) */
				ofspec = argv[++a];
				break;
#endif /* DC_GLARE */
			case 'f':
				outfmt = DTfloat;
				break;
			case 'd':
				outfmt = DTdouble;
				break;
			case 'a':
				outfmt = DTascii;
				break;
			default:
				goto userr;
			}
			break;
#ifdef DC_GLARE
		case 's':
			if (!strcmp(argv[a], "-sunm")) {
				visible_sun_path = argv[++a];
			} else if (!strcmp(argv[a], "-suns")) {
				suns_rad_path = argv[++a];
			} else {
				switch (argv[a][2]) {
				case 'f':	/* occupancy schedule file */
					schedule_path = argv[++a];
					break;
				case 's':	/* occupancy start hour */
					start_hour = atoi(argv[++a]);
					break;
				case 'e':	/* occupancy end hour */
					end_hour = atoi(argv[++a]);
					break;
				default:
					goto userr;
				}
			}
			break;
		case 'l':	/* perceptible glare threshold */
			dgp_limit = atof(argv[++a]);
			break;
		case 'b':	/* luminance threshold */
			dgp_threshold = atof(argv[++a]);
			break;
		case 'v':
			switch (argv[a][2]) {
			case 'd':	/* forward */
				//check(3, "fff");
				vdir[0] = atof(argv[++a]);
				vdir[1] = atof(argv[++a]);
				vdir[2] = atof(argv[++a]);
				if (normalize(vdir) == 0.0) goto userr;
				break;
			case 'u':	/* up */
				//check(3, "fff");
				vup[0] = atof(argv[++a]);
				vup[1] = atof(argv[++a]);
				vup[2] = atof(argv[++a]);
				if (normalize(vup) == 0.0) goto userr;
				break;
			case 'f':	/* view directions file */
				//check(2, "s");
				view_path = argv[++a];
				break;
			case 'i':
				switch (argv[a][3]) {
				case 'f':
					viewfmt = DTfloat;
					break;
				case 'd':
					viewfmt = DTdouble;
					break;
				case 'a':
					viewfmt = DTascii;
					break;
				default:
					goto userr;
				}
				break;
			default:
				goto userr;
			}
			break;
		case 'w':
			if (!strcmp(argv[a], "-wgroup")) {
				current_group = append_window_group(&window_groups,
						&nwindow_groups);
				if (!current_group) {
					fprintf(stderr, "%s: out of memory for window groups\n",
							progname);
					return(1);
				}
				current_group->window_rad_path = argv[++a];
			} else if (!strcmp(argv[a], "-wVtotalMF1") ||
					!strcmp(argv[a], "-wV1")) {
				if (current_group)
					current_group->view_low_total_path = argv[++a];
				else
					view_low_total_path = argv[++a];
			} else if (!strcmp(argv[a], "-wTDSallMF1")) {
				if (current_group)
					current_group->tds_low_total_path = argv[++a];
				else
					tds_low_total_path = argv[++a];
			} else if (!strcmp(argv[a], "-wVsunMF1") ||
					!strcmp(argv[a], "-w1v")) {
				if (current_group)
					current_group->direct_low_view_path = argv[++a];
				else
					direct_low_view_path = argv[++a];
			} else if (!strcmp(argv[a], "-wTDSsunMF1") ||
					!strcmp(argv[a], "-w1d")) {
				if (current_group)
					current_group->direct_low_path = argv[++a];
				else
					direct_low_path = argv[++a];
			} else if (!strcmp(argv[a], "-wVsunAllMF10") ||
					!strcmp(argv[a], "-wVMF10")) {
				if (current_group)
					current_group->direct_high_view_path = argv[++a];
				else
					direct_high_view_path = argv[++a];
			} else if (!strcmp(argv[a], "-wTDSsunAllMF10") ||
					!strcmp(argv[a], "-w10d")) {
				if (current_group)
					current_group->direct_high_path = argv[++a];
				else
					direct_high_path = argv[++a];
			} else if (!strcmp(argv[a], "-wVsunOnlyMF10") ||
					!strcmp(argv[a], "-wVMF10dir")) {
				if (current_group)
					current_group->direct_high_view_sub_path = argv[++a];
				else
					direct_high_view_sub_path = argv[++a];
			} else if (!strcmp(argv[a], "-wTDSsunOnlyMF10") ||
					!strcmp(argv[a], "-w10ddir")) {
				if (current_group)
					current_group->direct_high_sub_path = argv[++a];
				else
					direct_high_sub_path = argv[++a];
			} else if (!strcmp(argv[a], "-wVisibleSun")) {
				if (!current_group)
					goto userr;
				current_group->visible_sun_path = argv[++a];
			} else if (!strcmp(argv[a], "-wr")) {
				if (current_group)
					goto userr;
				window_rad_path = argv[++a];
			} else
				goto userr;
			break;
		case 'T':
			if (!strcmp(argv[a], "-TDS"))
				tds_low_total_path = argv[++a];
			else
				goto userr;
			break;
#endif /* DC_GLARE */
		default:
			goto userr;
		}
#ifdef DC_GLARE
	replacement_mode = view_low_total_path || tds_low_total_path ||
			direct_low_view_path || direct_low_path ||
			direct_high_view_path || direct_high_path ||
			direct_high_view_sub_path || direct_high_sub_path ||
			nwindow_groups;
	if (replacement_mode) {
		if (nwindow_groups) {
			if (view_low_total_path || tds_low_total_path ||
					direct_low_view_path || direct_low_path ||
					direct_high_view_path || direct_high_path ||
					direct_high_view_sub_path || direct_high_sub_path ||
					window_rad_path || visible_sun_path)
				goto userr;
			for (i = 0; i < nwindow_groups; i++) {
				WINDOW_GROUP_SPEC *wg = &window_groups[i];
				if (!wg->view_low_total_path || !wg->tds_low_total_path ||
						!wg->direct_low_view_path || !wg->direct_low_path ||
						!wg->direct_high_view_path || !wg->direct_high_path)
					goto userr;
				if ((wg->direct_high_view_sub_path == NULL) !=
						(wg->direct_high_sub_path == NULL))
					goto userr;
				if ((wg->visible_sun_path ||
						wg->direct_high_view_sub_path) && !suns_rad_path)
					goto userr;
			}
			/* Reuse the first group for initial dimensions and time count. */
			view_low_total_path = window_groups[0].view_low_total_path;
			tds_low_total_path = window_groups[0].tds_low_total_path;
		} else {
			if (!view_low_total_path || !tds_low_total_path ||
					!direct_low_view_path || !direct_low_path ||
					!direct_high_view_path || !direct_high_path)
				goto userr;
			if ((visible_sun_path == NULL) != (suns_rad_path == NULL))
				goto userr;
			if ((direct_high_view_sub_path == NULL) !=
					(direct_high_sub_path == NULL))
				goto userr;
			if ((direct_high_view_sub_path || direct_high_sub_path) &&
					(!visible_sun_path || !suns_rad_path))
				goto userr;
		}
		if (argc-a != 1)
			goto userr;
	} else if (visible_sun_path || suns_rad_path)
		goto userr;
	else if ((argc-a < 2) | (argc-a > 5))
		goto userr;
	if (!replacement_mode) {
		/* direct view matrix / daylight coefficients file */
		direct_path = argv[a++];
	}
#else
	if ((argc-a < 1) | (argc-a > 4))
		goto userr;
#endif /* DC_GLARE */

	if (replacement_mode) {			/* TDS1 matrix; Ev matrix is loaded later */
		cmtx = cm_load(tds_low_total_path, 0, nsteps, skyfmt);
		nsteps = cmtx->ncols;
	} else if (argc-a > 2) {			/* VTDs expression */
		CMATRIX		*smtx, *Dmat, *Tmat, *imtx;
		const char	*ccp;
						/* get sky vector/matrix */
		smtx = cm_load(argv[a+3], 0, nsteps, skyfmt);
		nsteps = smtx->ncols;
						/* load BSDF */
		if (argv[a+1][0] != '!' &&
				(ccp = strrchr(argv[a+1], '.')) != NULL &&
				!strcasecmp(ccp+1, "XML"))
			Tmat = cm_loadBTDF(argv[a+1]);
		else
			Tmat = cm_load(argv[a+1], 0, 0, DTfromHeader);
						/* load Daylight matrix */
		Dmat = cm_load(argv[a+2], Tmat->ncols,
					smtx->nrows, DTfromHeader);
						/* multiply vector through */
		imtx = cm_multiply(Dmat, smtx);
		cm_free(Dmat); cm_free(smtx);
		cmtx = cm_multiply(Tmat, imtx);
		cm_free(Tmat); 
		cm_free(imtx);
	} else {				/* sky vector/matrix only */
		//TIMER(timer, "read args");
		cmtx = cm_load(argv[a+1], 0, nsteps, skyfmt);
		nsteps = cmtx->ncols;
		//TIMER(timer, "load sky matrix");
	}
						/* prepare output stream */
	if ((ofspec != NULL) & (nsteps == 1) && hasNumberFormat(ofspec)) {
		sprintf(fnbuf, ofspec, 1);
		ofspec = fnbuf;
	}
	if (ofspec != NULL && !hasNumberFormat(ofspec)) {
		if ((ofp = fopen(ofspec, "w")) == NULL) {
			fprintf(stderr, "%s: cannot open '%s' for output\n",
					progname, ofspec);
			return(1);
		}
		ofspec = NULL;			/* only need to open once */
	}
	if (hasNumberFormat(argv[a])) {		/* generating image(s) */
		if (ofspec == NULL) {
			SET_FILE_BINARY(ofp);
			newheader("RADIANCE", ofp);
			printargs(argc, argv, ofp);
			fputnow(ofp);
		}
		if (nsteps > 1)			/* multiple output frames? */
			for (i = 0; i < nsteps; i++) {
				CMATRIX	*cvec = cm_column(cmtx, i);
				if (ofspec != NULL) {
					sprintf(fnbuf, ofspec, i);
					if ((ofp = fopen(fnbuf, "wb")) == NULL) {
						fprintf(stderr,
							"%s: cannot open '%s' for output\n",
							progname, fnbuf);
						return(1);
					}
					newheader("RADIANCE", ofp);
					printargs(argc, argv, ofp);
					fputnow(ofp);
				}
				fprintf(ofp, "FRAME=%d\n", i);
				if (!sum_images(argv[a], cvec, ofp))
					return(1);
				if (ofspec != NULL) {
					if (fclose(ofp) == EOF) {
						fprintf(stderr,
							"%s: error writing to '%s'\n",
							progname, fnbuf);
						return(1);
					}
					ofp = stdout;
				}
				cm_free(cvec);
			}
		else if (!sum_images(argv[a], cmtx, ofp))
			return(1);
	} else {				/* generating vector/matrix */
		CMATRIX	*Vmat = cm_load(replacement_mode ? view_low_total_path : argv[a],
					0, cmtx->nrows, DTfromHeader);
		CMATRIX	*rmtx;
		//TIMER(timer, "load view matrix");
		if (replacement_mode)
			rmtx = cm_load(argv[a], Vmat->nrows, nsteps, DTfromHeader);
		else
			rmtx = cm_multiply(Vmat, cmtx);
		//TIMER(timer, "multiply");
#ifdef DC_GLARE
		if (direct_path || replacement_mode) { /* Do glare autonomy calculation */
			/* Load occupancy schedule */
			occupancy = (int*)malloc(nsteps * sizeof(int));
			if (!occupancy) {
				fprintf(stderr,
					"%s: out of memory for schedule\n",
					progname);
				return(1);
			}
			if (schedule_path) {
				if ((fp = fopen(schedule_path, "r")) == NULL) {
					fprintf(stderr,
						"%s: cannot open input file \"%s\"\n",
						progname, schedule_path);
					return(1);
				}
				if (cm_load_schedule(nsteps, occupancy, fp) < 0) return(1);
			}
			else {
				for (i = 0; i < nsteps; i++) {
					/* Assume hourly spacing */
					int hour = i % 24;
					occupancy[i] = ((hour >= start_hour) & (hour < end_hour));
				}
			}
			//TIMER(timer, "load occupancy schedule");

			if (!nwindow_groups &&
					cm_load_window_dir(wdir, window_rad_path) < 0)
				return(1);

			/* Load view directions */
			if (view_path == NULL) {
				if (vdir[0] == 0.0f && vdir[1] == 0.0f && vdir[2] == 0.0f) {
					fprintf(stderr,
						"%s: missing view direction\n",
						progname);
					return(1);
				}
			}
			else {
				if ((fp = fopen(view_path, "r")) == NULL) {
					fprintf(stderr,
						"%s: cannot open input file \"%s\"\n",
						progname, view_path);
					return(1);
				}
				if (viewfmt != DTascii)
					SET_FILE_BINARY(fp);
				views = cm_load_views(rmtx->nrows, viewfmt, fp);
				if (!views) return(1);
				//TIMER(timer, "load views");
			}

			/* Calculate glare values */
			{
				if (replacement_mode) {
					if (nwindow_groups) {
						CM_WINDOW_GROUP *loaded = (CM_WINDOW_GROUP *)calloc(
								nwindow_groups, sizeof(CM_WINDOW_GROUP));
						CM_SUN *suns = NULL;
						int nsuns = 0;
						int load_failed = (loaded == NULL);

						if (!load_failed && suns_rad_path) {
							suns = cm_load_suns(suns_rad_path, &nsuns);
							load_failed = (suns == NULL);
						}
						for (i = 0; !load_failed && i < nwindow_groups; i++) {
							WINDOW_GROUP_SPEC *spec = &window_groups[i];
							CM_WINDOW_GROUP *wg = &loaded[i];
							const char *radpath = spec->window_rad_path;

							wg->wdir[0] = 0.0;
							wg->wdir[1] = -1.0;
							wg->wdir[2] = 0.0;
							if (radpath && !strcmp(radpath, "-"))
								radpath = NULL;
							if (cm_load_window_dir(wg->wdir, radpath) < 0) {
								load_failed = 1;
								break;
							}
							if (i == 0) {
								wg->v1 = Vmat;
								wg->tds1 = cmtx;
							} else {
								wg->tds1 = cm_load(spec->tds_low_total_path,
										0, nsteps, DTfromHeader);
								if (wg->tds1)
									wg->v1 = cm_load(spec->view_low_total_path,
										rmtx->nrows, wg->tds1->nrows,
										DTfromHeader);
							}
							wg->tds2 = cm_load(spec->direct_low_path,
									wg->tds1 ? wg->tds1->nrows : 0,
									nsteps, DTfromHeader);
							wg->tds3 = cm_load(spec->direct_high_path,
									0, nsteps, DTfromHeader);
							if (wg->tds2)
								wg->v2 = cm_load(spec->direct_low_view_path,
										rmtx->nrows, wg->tds2->nrows,
										DTfromHeader);
							if (wg->tds3)
								wg->v3 = cm_load(spec->direct_high_view_path,
										rmtx->nrows, wg->tds3->nrows,
										DTfromHeader);
							if (spec->direct_high_view_sub_path && wg->tds3) {
								wg->tds3direct = cm_load(spec->direct_high_sub_path,
										wg->tds3->nrows, nsteps, DTfromHeader);
								if (wg->tds3direct)
									wg->v3direct = cm_load(
											spec->direct_high_view_sub_path,
											rmtx->nrows, wg->tds3direct->nrows,
											DTfromHeader);
							}
							if (spec->visible_sun_path)
								wg->sunm = cm_load(spec->visible_sun_path,
										rmtx->nrows, nsteps, DTfromHeader);
							load_failed = !wg->v1 || !wg->tds1 || !wg->v2 ||
									!wg->tds2 || !wg->v3 || !wg->tds3 ||
									(spec->direct_high_view_sub_path &&
										(!wg->v3direct || !wg->tds3direct)) ||
									(spec->visible_sun_path && !wg->sunm);
						}
						if (!load_failed)
							dgp_values = cm_glare_reinhart_groups(loaded,
									nwindow_groups, rmtx, suns, nsuns,
									occupancy, dgp_limit, dgp_threshold,
									views, vdir, vup);
						if (loaded) {
							for (i = 0; i < nwindow_groups; i++) {
								if (i > 0) {
									cm_free((CMATRIX *)loaded[i].v1);
									cm_free((CMATRIX *)loaded[i].tds1);
								}
								cm_free((CMATRIX *)loaded[i].v2);
								cm_free((CMATRIX *)loaded[i].tds2);
								cm_free((CMATRIX *)loaded[i].v3);
								cm_free((CMATRIX *)loaded[i].tds3);
								cm_free((CMATRIX *)loaded[i].v3direct);
								cm_free((CMATRIX *)loaded[i].tds3direct);
								cm_free((CMATRIX *)loaded[i].sunm);
							}
						}
						free(loaded);
						free(suns);
						if (load_failed)
							return(1);
					} else {
					CMATRIX	*v2mat, *v3mat, *tds2, *tds3;
					CMATRIX	*v3dirmat = NULL, *tds3dir = NULL;
					CMATRIX	*sunm = NULL;
					CM_SUN	*suns = NULL;
					int	nsuns = 0;
					tds2 = cm_load(direct_low_path, cmtx->nrows,
							nsteps, DTfromHeader);
					tds3 = cm_load(direct_high_path, 0,
							nsteps, DTfromHeader);
					v2mat = cm_load(direct_low_view_path,
							rmtx->nrows, tds2->nrows, DTfromHeader);
					v3mat = cm_load(direct_high_view_path,
							rmtx->nrows, tds3->nrows, DTfromHeader);
					if (direct_high_view_sub_path) {
						tds3dir = cm_load(direct_high_sub_path,
								tds3->nrows, nsteps, DTfromHeader);
						v3dirmat = cm_load(direct_high_view_sub_path,
								rmtx->nrows, tds3dir->nrows,
								DTfromHeader);
					}
					if (visible_sun_path) {
						sunm = cm_load(visible_sun_path,
								rmtx->nrows, nsteps, DTfromHeader);
						suns = cm_load_suns(suns_rad_path, &nsuns);
						if (!sunm || !suns) {
							cm_free(v2mat);
							cm_free(v3mat);
							cm_free(tds2);
							cm_free(tds3);
							cm_free(v3dirmat);
							cm_free(tds3dir);
							cm_free(sunm);
							free(suns);
							return(1);
						}
					}
					dgp_values = cm_glare_reinhart_replace(Vmat, cmtx,
							v2mat, tds2, v3mat, tds3,
							v3dirmat, tds3dir, rmtx,
							sunm, suns, nsuns,
							occupancy, dgp_limit, dgp_threshold,
							views, vdir, vup, wdir);
					cm_free(v2mat);
					cm_free(v3mat);
					cm_free(tds2);
					cm_free(tds3);
					cm_free(v3dirmat);
					cm_free(tds3dir);
					cm_free(sunm);
					free(suns);
					}
				} else {
					CMATRIX	*v3mat = cm_load(direct_path, 0,
							cmtx->nrows, DTfromHeader);
					//TIMER(timer, "load direct matrix");
					dgp_values = cm_glare(v3mat, rmtx, cmtx,
							occupancy, dgp_limit, dgp_threshold,
							views, vdir, vup, wdir);
					cm_free(v3mat);
				}
				//TIMER(timer, "calculate dgp");
				free(views);
			}

			/* Check successful calculation */
			if (!dgp_values) return(1);
		}
#endif /* DC_GLARE */
		if (ofspec != NULL) {		/* multiple vector files? */
			const char	*wtype = (outfmt==DTascii) ? "w" : "wb";
			for (i = 0; i < nsteps; i++) {
				CMATRIX	*rvec = cm_column(rmtx, i);
				sprintf(fnbuf, ofspec, i);
				if ((ofp = fopen(fnbuf, wtype)) == NULL) {
					fprintf(stderr,
					"%s: cannot open '%s' for output\n",
							progname, fnbuf);
					return(1);
				}
#ifdef getc_unlocked
				flockfile(ofp);
#endif
				if (headout) {	/* header output */
					newheader("RADIANCE", ofp);
					printargs(argc, argv, ofp);
					fputnow(ofp);
					fprintf(ofp, "FRAME=%d\n", i);
					fprintf(ofp, "NROWS=%d\n", rvec->nrows);
					fputs("NCOLS=1\nNCOMP=3\n", ofp);
					if ((outfmt == DTfloat) | (outfmt == DTdouble))
						fputendian(ofp);
					fputformat(cm_fmt_id[outfmt], ofp);
					fputc('\n', ofp);
				}
				cm_write(rvec, outfmt, ofp);
				if (fclose(ofp) == EOF) {
					fprintf(stderr,
						"%s: error writing to '%s'\n",
							progname, fnbuf);
					return(1);
				}
				ofp = stdout;
				cm_free(rvec);
			}
		} else {
#ifdef getc_unlocked
			flockfile(ofp);
#endif
			if (outfmt != DTascii)
				SET_FILE_BINARY(ofp);
			if (headout) {		/* header output */
				newheader("RADIANCE", ofp);
				printargs(argc, argv, ofp);
				fputnow(ofp);
				fprintf(ofp, "NROWS=%d\n", rmtx->nrows);
#ifdef DC_GLARE
				fprintf(ofp, "NCOLS=%d\n", (!dgp_values || dgp_limit < 0) ? rmtx->ncols : 1);
				fprintf(ofp, "NCOMP=%d\n", dgp_values ? 1 : 3);
#else
				fprintf(ofp, "NCOLS=%d\n", rmtx->ncols);
				fputs("NCOMP=3\n", ofp);
#endif /* DC_GLARE */
				if ((outfmt == DTfloat) | (outfmt == DTdouble))
					fputendian(ofp);
				fputformat(cm_fmt_id[outfmt], ofp);
				fputc('\n', ofp);
			}
#ifdef DC_GLARE
			if (dgp_values) { /* Write glare autonomy */
				cm_write_glare(dgp_values, rmtx->nrows, dgp_limit < 0 ? rmtx->ncols : 1, outfmt, ofp);
				free(dgp_values);
				//TIMER(timer, "write");
			} 
			else
#endif /* DC_GLARE */
			cm_write(rmtx, outfmt, ofp);
		}
		cm_free(rmtx);
		cm_free(Vmat);
	}
	if (fflush(ofp) == EOF) {		/* final clean-up */
		fprintf(stderr, "%s: write error on output\n", progname);
		return(1);
	}
	cm_free(cmtx);
#ifdef DC_GLARE
	free(window_groups);
#endif
	return(0);
userr:
#ifdef DC_GLARE
	fprintf(stderr, "Usage: %s [-n nsteps][-i{f|d|h}][-o{f|d}][-l limit][-b threshold][-wr window.rad][{-sf occupancy|-ss start -se end}]{-vf views [-vi{f|d}]|-vd x y z}[-vu x y z] DCdirect DCtotal [skyf]\n",
		progname);
	fprintf(stderr, "   or: %s [-n nsteps][-i{f|d|h}][-o{f|d}][-l limit][-b threshold][-wr window.rad][{-sf occupancy|-ss start -se end}]{-vf views [-vi{f|d}]|-vd x y z}[-vu x y z] DCdirect Vspec Tbsdf Dmat.dat [skyf]\n",
		progname);
	fprintf(stderr, "   or: %s [-n nsteps][-i{f|d|h}][-o{f|d}][-l limit][-b threshold][-wr window.rad][-sunm sunm.mtx -suns suns.rad][-wVtotalMF1 VtotalMF1 -wTDSallMF1 TDSallMF1 -wVsunMF1 VsunMF1 -wTDSsunMF1 TDSsunMF1 -wVsunAllMF10 VsunAllMF10 -wTDSsunAllMF10 TDSsunAllMF10][-wVsunOnlyMF10 VsunOnlyBlackMF10 -wTDSsunOnlyMF10 TDSsunOnlyBlackMF10][{-sf occupancy|-ss start -se end}]{-vf views [-vi{f|d}]|-vd x y z}[-vu x y z] Ev\n",
		progname);
	fprintf(stderr, "   or: %s [common options] {-wgroup {window.rad|-} -wVtotalMF1 V1 -wTDSallMF1 TDS1 -wVsunMF1 V2 -wTDSsunMF1 TDS2 -wVsunAllMF10 V3 -wTDSsunAllMF10 TDS3 [-wVsunOnlyMF10 V3sun -wTDSsunOnlyMF10 TDS3sun] [-wVisibleSun sunm]}... [-suns suns.rad] Ev\n",
		progname);
	fprintf(stderr, "       Each -wgroup starts an independent Reinhart window-direction hemisphere; following -w* options belong to that group. '-' uses the default global -Y apex.\n");
	free(window_groups);
#else
	fprintf(stderr, "Usage: %s [-n nsteps][-o ospec][-i{f|d|h}][-o{f|d}] DCspec [skyf]\n",
				progname);
	fprintf(stderr, "   or: %s [-n nsteps][-o ospec][-i{f|d|h}][-o{f|d}] Vspec Tbsdf Dmat.dat [skyf]\n",
				progname);
#endif /* DC_GLARE */
	return(1);
}
