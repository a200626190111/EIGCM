#ifndef _RAD_CMGLARE_H_
#define _RAD_CMGLARE_H_

#include "rtio.h"
#include "time.h"
#include "fvect.h"
#include "cmatrix.h"

#define DC_GLARE /* Perform glare autonomy calculation */

#ifdef __cplusplus
extern "C" {
#endif

#ifdef DC_GLARE
#define TIMER(c, m) do {	\
	clock_t temp = clock(); \
	fprintf(stderr, "%s: %g s\n", m, 1.0 * (temp - c) / CLOCKS_PER_SEC); \
	c = temp; } while(0)
#else
#define TIMER(c, m)
#endif

typedef struct cm_sun_s {
	FVECT	dir;
	double	omega;
} CM_SUN;

typedef struct cm_window_group_s {
	const CMATRIX	*v1;
	const CMATRIX	*tds1;
	const CMATRIX	*v2;
	const CMATRIX	*tds2;
	const CMATRIX	*v3;
	const CMATRIX	*tds3;
	const CMATRIX	*v3direct;
	const CMATRIX	*tds3direct;
	const CMATRIX	*sunm;
	FVECT		wdir;
} CM_WINDOW_GROUP;

int cm_load_window_dir(FVECT wdir, const char *fspec);
CM_SUN* cm_load_suns(const char *fspec, int *nsuns);
float* cm_glare(const CMATRIX *dcmx, const CMATRIX *evmx, const CMATRIX *smx, const int *occupied, const double dgp_limit, const double dgp_threshold, const FVECT *views, const FVECT dir, const FVECT up, const FVECT wdir);
float* cm_glare_reinhart_replace(const CMATRIX *v1, const CMATRIX *tds1, const CMATRIX *v2, const CMATRIX *tds2, const CMATRIX *v3, const CMATRIX *tds3, const CMATRIX *v3direct, const CMATRIX *tds3direct, const CMATRIX *evmx, const CMATRIX *sunm, const CM_SUN *suns, const int nsuns, const int *occupied, const double dgp_limit, const double dgp_threshold, const FVECT *views, const FVECT dir, const FVECT up, const FVECT wdir);
float* cm_glare_reinhart_groups(const CM_WINDOW_GROUP *groups, const int ngroups, const CMATRIX *evmx, const CM_SUN *suns, const int nsuns, const int *occupied, const double dgp_limit, const double dgp_threshold, const FVECT *views, const FVECT dir, const FVECT up);
int cm_load_schedule(const int count, int* schedule, FILE *fp);
FVECT* cm_load_views(const int nrows, const int inform, FILE *fp);
int cm_write_glare(const float *mp, const int nrows, const int ncols, const int dtype, FILE *fp);

#ifdef __cplusplus
}
#endif
#endif	/* _RAD_CMGLARE_H_ */
