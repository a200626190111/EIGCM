/* RCSid $Id$ */

/*
 * Header file for rcontrib modules
 */

#include "platform.h"
#include "rtprocess.h"
#include "ray.h"
#include "func.h"
#include "lookup.h"

extern int		gargc;		/* global argc */
extern char		**gargv;	/* global argv */
extern char		*octname;	/* global octree name */

extern int		nproc;		/* number of processes requested */
extern int		nchild;		/* number of children (-1 in child) */

extern int		rc_worker;	/* internal worker mode? */
extern int		rc_worker_id;	/* internal worker identifier */

extern int		inpfmt;		/* input format */
extern int		outfmt;		/* output format */

extern int		header;		/* output header? */
extern int		force_open;	/* truncate existing output? */
extern int		recover;	/* recover previous output? */
extern int		accumulate;	/* input rays per output record */
extern int		contrib;	/* computing contributions? */
extern int		rc_rflux_inactive_sentinel; /* accept masked rfluxmtx samples */
extern int		rc_adaptive_accumulate; /* stop converged accumulation records */
extern int		rc_adaptive_initial;
extern int		rc_adaptive_minimum;
extern int		rc_adaptive_zero_minimum;
extern int		rc_adaptive_stable_rounds;
extern double		rc_adaptive_rms_tolerance;
extern double		rc_adaptive_peak_tolerance;
extern double		rc_adaptive_absolute_floor;
extern char		*rc_adaptive_report;

#define RCPATH_ALL	0
#define RCPATH_EXPLICIT	1
#define RCPATH_RESIDUAL	2

extern int		rc_path_output;	/* optical path output mode */
extern unsigned int	rc_path_mirror_orders; /* explicitly replaced mirror orders */
extern int		rc_dual_path_output; /* total followed by no-diffuse bins */
extern int		rc_dual_source_output; /* total followed by physical-source bins */
extern int		rc_path_direct_max_reflections; /* limit for direct component */
extern int		rc_path_direct_max_ambient; /* ambient-event limit for direct */
extern int		rc_path_direct_straight_only; /* reject redirected direct paths */

extern int		xres;		/* horizontal (scan) size */
extern int		yres;		/* vertical resolution */

extern int		using_stdout;	/* are we using stdout? */

extern int		imm_irrad;	/* compute immediate irradiance? */
extern int		lim_dist;	/* limit distance? */

extern int		account;	/* current accumulation count */
extern RNUMBER		raysleft;	/* number of rays left to trace */
extern long		waitflush;	/* how long until next flush */

extern RNUMBER		lastray;	/* last ray number sent */
extern RNUMBER		lastdone;	/* last ray processed */

extern int		report_intvl;	/* reporting interval (seconds) */

typedef double		DCOLORV;	/* double-precision color type */

/*
 * The MODCONT structure is used to accumulate ray contributions
 * for a particular modifier, which may be subdivided into bins
 * if binv evaluates > 0.  If outspec contains a %s in it, this will
 * be replaced with the modifier name.  If outspec contains a %d in it,
 * this will be used to create one output file per bin, otherwise all bins
 * will be written to the same file, in order.  If the global outfmt
 * is 'c', then a common-exponent pixel will be output for each bin value
 * and the file will conform to a RADIANCE picture if NCSAMP==3 and
 * xres and yres are set.
 */
typedef struct {
	const char	*outspec;	/* output file specification */
	const char	*modname;	/* modifier name */
	const char	*params;	/* parameter list */
	EPNODE		*binv;		/* bin value expression */
	int		bin0;		/* starting bin offset */
	int		base_nbins;	/* directional bins before path components */
	int		ncomponents;	/* number of path components in output */
	int		nbins;		/* number of contribution bins */
	DCOLORV		cbin[1];	/* contribution bins (extends struct) */
} MODCONT;			/* modifier contribution */

#define DCOLORSIZ	(sizeof(DCOLORV)*NCSAMP)
#define mcsize(nb)	(sizeof(MODCONT)-sizeof(DCOLORV)+(nb)*DCOLORSIZ)
#define mcbin(mp,bi)	((mp)->cbin + (bi)*NCSAMP)

extern LUTAB		modconttab;	/* modifier contribution table */

/*
 * The STREAMOUT structure holds an open FILE pointer and a count of
 * the number of RGB triplets per record, or 0 if unknown.
 */
typedef struct {
	FILE		*ofp;		/* output file pointer */
	int		outpipe;	/* output is to a pipe */
	int		reclen;		/* triplets/record */
	int		xr, yr;		/* output resolution for picture */
} STREAMOUT;

extern LUTAB		ofiletab;	/* output stream table */

#ifndef MAXPROCESS
#define MAXPROCESS	128
#endif

extern char		**modname;		/* ordered modifier name list */
extern int		nmods;			/* number of modifiers */
extern int		modasiz;		/* allocated modifier array size */

extern char		RCCONTEXT[];		/* special evaluation context */

extern void		process_rcontrib(void);	/* trace ray contributions */

extern STREAMOUT *	getostream(const char *ospec, const char *mname,
							int bn, int noopen);

extern void		mod_output(MODCONT *mp);
extern void		end_record(void);

extern MODCONT		*addmodifier(char *modn, char *outf,
					char *prms, char *binv, int bincnt);
extern void		addmodfile(char *fname, char *outf,
					char *prms, char *binv, int bincnt);

extern int		rc_set_path_output(const char *mode);
extern int		rc_set_path_components(const char *components);
extern int		rc_set_source_components(const char *components);
extern int		rc_set_path_mirror_orders(const char *orders);
extern void		rc_add_path_mirror_file(char *fname);
extern const char	*rc_path_output_name(void);
extern void		rc_enable_path_filter(void);

extern void		reload_output(void);
extern void		recover_output(void);

extern int		getvec(FVECT vec);

extern int		in_rchild(void);
extern void		end_children(int immed);

extern void		put_zero_record(int ndx);

extern int		morays(void);		/* reached end of input? */

extern void		parental_loop(void);	/* controlling process */

extern void		feeder_loop(void);	/* feeder process */

extern void		rcontrib(void);		/* main calculation loop */
