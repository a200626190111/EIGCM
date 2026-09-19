#ifndef lint
static const char RCSid[] = "$Id$";
#endif
/*
 * Accumulate ray contributions for a set of materials
 * Initialization and calculation routines
 */

#include "copyright.h"

#include <stdlib.h>

#include "rcontrib.h"
#include "otypes.h"
#include "otspecial.h"
#include "rfluxmtx_protocol.h"
#include "source.h"

#ifndef  RCO_BSDF_INSTRUMENT
#define  RCO_BSDF_INSTRUMENT (getenv("RCO_BSDF_INSTRUMENT") != NULL)
#endif

CUBE	thescene;			/* our scene */
OBJECT	nsceneobjs;			/* number of objects in our scene */

int	dimlist[MAXDIM];		/* sampling dimensions */
int	ndims = 0;			/* number of sampling dimensions */
unsigned long	samplendx = 0;		/* index for this sample */

void	(*trace)() = NULL;		/* trace call (NULL before rcinit) */

int	do_irrad = 0;			/* compute irradiance? */
int	rc_rflux_inactive_sentinel = 0;	/* internal rfluxmtx protocol */
int	rc_adaptive_accumulate = 0;	/* adaptive -c accumulation */
int	rc_adaptive_initial = 32;
int	rc_adaptive_minimum = 64;
int	rc_adaptive_zero_minimum = 64;
int	rc_adaptive_stable_rounds = 2;
double	rc_adaptive_rms_tolerance = .02;
double	rc_adaptive_peak_tolerance = .10;
double	rc_adaptive_absolute_floor = 1e-9;
char	*rc_adaptive_report = NULL;

int	rc_path_output = RCPATH_ALL;	/* optical path output mode */
unsigned int rc_path_mirror_orders = (1u << 1) | (1u << 2);
int	rc_dual_path_output = 0;	/* output total bins, then no-diffuse bins */
int	rc_dual_source_output = 0;	/* output total bins, then physical-source bins */
int	rc_path_direct_max_reflections = -1;
int	rc_path_direct_max_ambient = -1;
int	rc_path_direct_straight_only = 0;

static char	**rc_path_mirror_modifiers = NULL;
static int	rc_path_mirror_count = 0;
static int	rc_path_mirror_capacity = 0;

int	rand_samp = 1;			/* pure Monte Carlo sampling? */

double	dstrsrc = 0.9;			/* square source distribution */
double	shadthresh = 0.;		/* shadow threshold */
double	shadcert = .75;			/* shadow certainty */
int	directrelay = 3;		/* number of source relays */
int	vspretest = 512;		/* virtual source pretest density */
int	directvis = 1;			/* sources visible? */
double	srcsizerat = .2;		/* maximum ratio source size/dist. */

COLOR	cextinction = BLKCOLOR;		/* global extinction coefficient */
COLOR	salbedo = BLKCOLOR;		/* global scattering albedo */
double	seccg = 0.;			/* global scattering eccentricity */
double	ssampdist = 0.;			/* scatter sampling distance */

double	specthresh = .02;		/* specular sampling threshold */
double	specjitter = 1.;		/* specular sampling jitter */

int	backvis = 1;			/* back face visibility */

int	maxdepth = -10;			/* maximum recursion depth */
double	minweight = 2e-3;		/* minimum ray weight */

char	*ambfile = NULL;		/* ambient file name */
COLOR	ambval = BLKCOLOR;		/* ambient value */
int	ambvwt = 0;			/* initial weight for ambient value */
double	ambacc = 0.;			/* ambient accuracy */
int	ambres = 256;			/* ambient resolution */
int	ambdiv = 350;			/* ambient divisions */
int	ambssamp = 0;			/* ambient super-samples */
int	ambounce = 1;			/* ambient bounces */
char	*amblist[AMBLLEN+1];		/* ambient include/exclude list */
int	ambincl = -1;			/* include == 1, exclude == 0 */

int	account;			/* current accumulation count */
RNUMBER	raysleft;			/* number of rays left to trace */
long	waitflush;			/* how long until next flush */

RNUMBER	lastray = 0;			/* last ray number sent */
RNUMBER	lastdone = 0;			/* last ray output */

static void	trace_contrib(RAY *r);	/* our trace callback */

static void mcfree(void *p) { epfree((*(MODCONT *)p).binv,1); free(p); }

LUTAB	modconttab = LU_SINIT(NULL,mcfree);	/* modifier lookup table */

static int
rc_path_is_selected_mirror(const char *name)
{
	int	i;

	if (name == NULL)
		return(0);
	for (i = 0; i < rc_path_mirror_count; i++)
		if (!strcmp(name, rc_path_mirror_modifiers[i]))
			return(1);
	return(0);
}


static void
rc_path_add_mirror_modifier(const char *name)
{
	char	**newlist;
	int	i;

	if (name == NULL || !*name)
		return;
	for (i = 0; i < rc_path_mirror_count; i++)
		if (!strcmp(name, rc_path_mirror_modifiers[i]))
			return;
	if (rc_path_mirror_count >= rc_path_mirror_capacity) {
		rc_path_mirror_capacity += rc_path_mirror_capacity/2 + 32;
		newlist = (char **)realloc(rc_path_mirror_modifiers,
				rc_path_mirror_capacity*sizeof(char *));
		if (newlist == NULL)
			error(SYSTEM, "out of memory loading path mirror modifiers");
		rc_path_mirror_modifiers = newlist;
	}
	rc_path_mirror_modifiers[rc_path_mirror_count++] = savqstr(name);
}


void
rc_add_path_mirror_file(char *fname)
{
	char	*path = getpath(fname, getrlibpath(), R_OK);
	char	mod[MAXSTR];
	FILE	*fp;

	if (path == NULL || (fp = fopen(path, "r")) == NULL) {
		if (path == NULL)
			sprintf(errmsg, "cannot find path mirror modifier file '%s'", fname);
		else
			sprintf(errmsg, "cannot load path mirror modifier file '%s'", path);
		error(SYSTEM, errmsg);
	}
	while (fgetword(mod, sizeof(mod), fp) != NULL)
		rc_path_add_mirror_modifier(mod);
	fclose(fp);
}


int
rc_set_path_output(const char *mode)
{
	if (!strcmp(mode, "all"))
		rc_path_output = RCPATH_ALL;
	else if (!strcmp(mode, "explicit"))
		rc_path_output = RCPATH_EXPLICIT;
	else if (!strcmp(mode, "residual"))
		rc_path_output = RCPATH_RESIDUAL;
	else
		return(0);
	return(1);
}


int
rc_set_path_components(const char *components)
{
	if (strcmp(components, "total,direct"))
		return(0);
	rc_dual_path_output = 1;
	return(1);
}


int
rc_set_source_components(const char *components)
{
	if (strcmp(components, "total,physical"))
		return(0);
	rc_dual_source_output = 1;
	return(1);
}


const char *
rc_path_output_name(void)
{
	switch (rc_path_output) {
	case RCPATH_EXPLICIT:
		return("explicit");
	case RCPATH_RESIDUAL:
		return("residual");
	}
	return("all");
}


int
rc_set_path_mirror_orders(const char *orders)
{
	const char	*cp = orders;
	char		*ep;
	unsigned int	mask = 0;
	long		order;

	if (orders == NULL || !*orders)
		return(0);
	while (*cp) {
		order = strtol(cp, &ep, 10);
		if (ep == cp || order <= 0 || order >= 8*(long)sizeof(mask))
			return(0);
		mask |= 1u << order;
		if (*ep == '\0')
			break;
		if (*ep != ',')
			return(0);
		cp = ep + 1;
		if (!*cp)
			return(0);
	}
	rc_path_mirror_orders = mask;
	return(1);
}


static void
rc_path_append_event(RAY *ray, unsigned int event)
{
	if (ray->rpath_event_count < 2*sizeof(ray->rpath_signature))
		ray->rpath_signature |= (event & 0xfu) <<
				(4*ray->rpath_event_count);
	if (ray->rpath_event_count != (unsigned short)~0u)
		ray->rpath_event_count++;
	switch (event) {
	case RPE_MIRROR:
		if (ray->rpath_mirror_count != (unsigned short)~0u)
			ray->rpath_mirror_count++;
		break;
	case RPE_BSDF_REFL:
	case RPE_BSDF_TRANS:
		if (ray->rpath_bsdf_count != (unsigned short)~0u)
			ray->rpath_bsdf_count++;
		break;
	case RPE_OTHER_SPEC:
		if (ray->rpath_other_count != (unsigned short)~0u)
			ray->rpath_other_count++;
		break;
	case RPE_DIFFUSE:
		if (ray->rpath_diffuse_count != (unsigned short)~0u)
			ray->rpath_diffuse_count++;
		break;
	}
}


static void
rc_path_mark_redirect(RAY *ray)
{
	if (ray->rpath_redirect_count != (unsigned short)~0u)
		ray->rpath_redirect_count++;
}


static int
rc_path_classify_spawn(RAY *ray, const RAY *parent)
{
	OBJREC	*material;
	int	rtype;

	if (ray == NULL || parent == NULL || parent->ro == NULL)
		return(1);
	material = findmaterial(parent->ro);
	if (material == NULL)
		return(1);
	rtype = ray->rtype;
	if (rtype & AMBIENT) {
		rc_path_mark_redirect(ray);
		rc_path_append_event(ray, RPE_DIFFUSE);
		return(1);
	}
	/* Thin glass transmits straight through and does not redirect the path. */
	if (material->otype == MAT_GLASS && (rtype & (TRANS|TSHADOW)))
		return(1);
	if ((material->otype == MAT_BSDF || material->otype == MAT_ABSDF) &&
			(rtype & (RSHADOW|TSHADOW|REFLECTED|REFRACTED|
				RSPECULAR|TSPECULAR))) {
		if ((rtype & RAYREFL) ||
				DOT(ray->rdir, parent->rdir) < 1. - 1e-7)
			rc_path_mark_redirect(ray);
		rc_path_append_event(ray, rtype & RAYREFL ?
				RPE_BSDF_REFL : RPE_BSDF_TRANS);
		return(1);
	}
	if ((rtype & RAYREFL) &&
			rc_path_is_selected_mirror(material->oname)) {
		rc_path_mark_redirect(ray);
		rc_path_append_event(ray, RPE_MIRROR);
		return(1);
	}
	if (rtype & (RSHADOW|TSHADOW|REFLECTED|REFRACTED|
			RSPECULAR|TSPECULAR)) {
		if ((rtype & RAYREFL) ||
				DOT(ray->rdir, parent->rdir) < 1. - 1e-7)
			rc_path_mark_redirect(ray);
		rc_path_append_event(ray, RPE_OTHER_SPEC);
	}
	return(1);
}


static int
rc_path_is_explicit(const RAY *ray)
{
	if (ray->rpath_bsdf_count || ray->rpath_other_count ||
			ray->rpath_diffuse_count)
		return(0);
	if (!ray->rpath_mirror_count)
		return(1);
	if (ray->rpath_mirror_count >= 8*sizeof(rc_path_mirror_orders))
		return(0);
	return((rc_path_mirror_orders &
			(1u << ray->rpath_mirror_count)) != 0);
}


static int
rc_path_accept(const RAY *ray)
{
	int	is_explicit;

	if (rc_path_output == RCPATH_ALL)
		return(1);
	is_explicit = rc_path_is_explicit(ray);
	return(rc_path_output == RCPATH_EXPLICIT ?
			is_explicit : !is_explicit);
}


static int
rc_path_accept_direct(const RAY *ray)
{
	int	reflection_level = ray->rlvl;

	if (rc_path_direct_straight_only &&
			(ray->rpath_redirect_count || ray->rpath_diffuse_count))
		return(0);
	if (rc_path_direct_max_ambient >= 0 &&
			ray->rpath_diffuse_count > rc_path_direct_max_ambient)
		return(0);
	/* RSHADOW increments rlvl but bypasses the normal -lr depth test. */
	if ((ray->crtype & RSHADOW) && reflection_level > 0)
		reflection_level--;
	if (rc_path_direct_max_reflections >= 0 &&
			reflection_level > rc_path_direct_max_reflections)
		return(0);
	return(1);
}


void
rc_enable_path_filter(void)
{
	if (rc_path_output == RCPATH_ALL && !rc_dual_path_output &&
			!rc_dual_source_output)
		return;
	if (rc_dual_source_output && (rc_dual_path_output ||
			rc_path_output != RCPATH_ALL))
		error(USER,
			"--source-components cannot be combined with path-component output");
	if (rc_path_output != RCPATH_ALL && !direct_specular_only)
		error(USER,
			"path output filtering requires --direct-specular-only");
	if (rc_dual_path_output && rc_path_output != RCPATH_ALL)
		error(USER, "--path-components cannot be combined with --path-output");
	if (rc_dual_path_output && direct_specular_only)
		error(USER,
			"--path-components total,direct requires diffuse terms enabled");
	if (ray_spawn_check != NULL && ray_spawn_check != rc_path_classify_spawn)
		error(INTERNAL, "ray spawn check already configured");
	ray_spawn_check = rc_path_classify_spawn;
	if (rc_dual_path_output)
		direct_component_tracking = 1;
}

/************************** INITIALIZATION ROUTINES ***********************/

const char *
formstr(				/* return format identifier */
	int  f
)
{
	switch (f) {
	case 'a': return("ascii");
	case 'f': return("float");
	case 'd': return("double");
	case 'c': return(NCSAMP==3 ? COLRFMT : SPECFMT);
	}
	return("unknown");
}


/* Add modifier to our list to track */
MODCONT *
addmodifier(char *modn, char *outf, char *prms, char *binv, int bincnt)
{
	static int	lastNCS = 0;
	LUENT		*lep = lu_find(&modconttab,modn);
	MODCONT		*mp;
	EPNODE		*ebinv;
	int		i;

	if (!lastNCS)
		lastNCS = NCSAMP;
	else if (NCSAMP != lastNCS)
		error(INTERNAL,
		"number of spectral samples must be set before first modifier");
	if (lep->data != NULL) {
		sprintf(errmsg, "duplicate modifier '%s'", modn);
		error(USER, errmsg);
	}
	if (!strcmp(modn, VOIDID)) {
		sprintf(errmsg, "cannot track '%s' modifier", VOIDID);
		error(USER, errmsg);
	}
	if (nmods >= modasiz) {		/* need bigger modifier array */
		modasiz += modasiz/2 + 64;
		if (modname == NULL)
			modname = (char **)malloc(modasiz*sizeof(char *));
		else
			modname = (char **)realloc(modname, modasiz*sizeof(char *));
		if (modname == NULL)
			error(SYSTEM, "out of memory in addmodifier()");
	}
	modname[nmods++] = modn;	/* XXX assumes static string */
	lep->key = modn;		/* XXX assumes static string */
	if (binv == NULL)
		binv = "0";		/* use single bin if unspecified */
	ebinv = eparse(binv);
	if (ebinv->type == NUM) {	/* check value if constant */
		bincnt = (int)(evalue(ebinv) + 1.5);
		if (bincnt != 1) {
			sprintf(errmsg, "illegal non-zero constant for bin (%s)",
					binv);
			error(USER, errmsg);
		}
	} else if (bincnt <= 0) {
		sprintf(errmsg,
			"unspecified or illegal bin count for modifier '%s'",
				modn);
		error(USER, errmsg);
	}
					/* initialize results holder */
	mp = (MODCONT *)malloc(mcsize(bincnt*(1 + rc_dual_path_output +
			rc_dual_source_output)));
	if (mp == NULL)
		error(SYSTEM, "out of memory in addmodifier");
	mp->outspec = outf;		/* XXX assumes static string */
	mp->modname = modn;		/* XXX assumes static string */
	mp->params = prms;		/* XXX assumes static string */
	mp->binv = ebinv;
	mp->bin0 = 0;
	mp->base_nbins = bincnt;
	mp->ncomponents = 1 + rc_dual_path_output + rc_dual_source_output;
	mp->nbins = bincnt*mp->ncomponents;
	memset(mp->cbin, 0, DCOLORSIZ*mp->nbins);
					/* figure out starting bin */
	while (!getostream(mp->outspec, mp->modname, mp->bin0, 1))
		mp->bin0++;
					/* allocate other output streams */
	for (i = 0; ++i < mp->nbins; )
		getostream(mp->outspec, mp->modname, mp->bin0+i, 1);
	lep->data = (char *)mp;
	return(mp);
}


/* Add modifiers from a file list */
void
addmodfile(char *fname, char *outf, char *prms, char *binv, int bincnt)
{
	char	*path = getpath(fname, getrlibpath(), R_OK);
	char	mod[MAXSTR];
	FILE	*fp;

	if (path == NULL || (fp = fopen(path, "r")) == NULL) {
		if (path == NULL)
			sprintf(errmsg, "cannot find modifier file '%s'", fname);
		else
			sprintf(errmsg, "cannot load modifier file '%s'", path);
		error(SYSTEM, errmsg);
	}
	while (fgetword(mod, sizeof(mod), fp) != NULL)
		addmodifier(savqstr(mod), outf, prms, binv, bincnt);
	fclose(fp);
}


/* Check if we have any more rays left (and report progress) */
int
morays(void)
{
	static RNUMBER	total_rays;
	static time_t	tstart, last_report;
	time_t		tnow;

	if (!raysleft)
		return(1);	/* unknown total, so nothing to do or say */

	if (report_intvl > 0 && (tnow = time(0)) >= last_report+report_intvl) {
		if (!total_rays) {
			total_rays = raysleft;
			tstart = tnow;
		} else {
			sprintf(errmsg, "%.2f%% done after %.3f hours\n",
					100.-100.*raysleft/total_rays,
					(1./3600.)*(tnow - tstart));
			eputs(errmsg);
		}
		last_report = tnow;
	}
	return(--raysleft);
}


/* Quit program */
void
quit(
	int  code
)
{
	if (nchild > 0)		/* close children if any */
		end_children(code != 0);
	else if (nchild < 0)
		_exit(code);	/* avoid flush() in child */
	exit(code);
}


/* Initialize our process(es) */
static void
rcinit(void)
{
	int	i;

	if (nproc > MAXPROCESS)
		sprintf(errmsg, "too many processes requested -- reducing to %d",
				nproc = MAXPROCESS);
	if (nproc > 1)
		cow_memshare();		/* preload auxiliary data */
	trace = trace_contrib;		/* set up trace call-back */
	rc_enable_path_filter();
	for (i = 0; i < nsources; i++)	/* tracing to sources as well */
		source[i].sflags |= SFOLLOW;
	if (yres > 0) {			/* set up flushing & ray counts */
		if (xres > 0)
			raysleft = (RNUMBER)xres*yres;
		else
			raysleft = yres;
	} else
		raysleft = 0;
	if ((account = accumulate) > 1)
		raysleft *= accumulate;
	waitflush = (yres > 0) & (xres > 1) ? 0 : xres;

	if (nproc > 1 && in_rchild())	/* forked child? */
		return;			/* return to main processing loop */

	if (recover) {			/* recover previous output? */
		if (accumulate <= 0)
			reload_output();
		else
			recover_output();
	}
	if (nproc == 1)			/* single process? */
		return;
					/* else run appropriate controller */
	if (accumulate <= 0)
		feeder_loop();
	else
		parental_loop();
	quit(0);			/* parent musn't return! */
}

/************************** MAIN CALCULATION PROCESS ***********************/

/* Our trace call to sum contributions */
static void
trace_contrib(RAY *r)
{
	MODCONT	*mp;
	double	bval;
	int	bn;
	SCOLOR	contr;
	DCOLORV	*dvp;
	int	i;

	if (r->ro == NULL || r->ro->omod == OVOID)
		return;
						/* shadow ray not on source? */
	if (r->rsrc >= 0 && source[r->rsrc].so != r->ro)
		return;

	mp = (MODCONT *)lu_find(&modconttab,objptr(r->ro->omod)->oname)->data;

	if (mp == NULL)				/* not in our list? */
		return;
	if (!rc_path_accept(r))
		return;
						/* zero contribution? */
	if (contrib) {
		for (i = NCSAMP; i--; )
			if (r->rcoef[i]*r->rcol[i] > FTINY)
				break;		/* something non-zero */
		if (i < 0)
			return;
	} else if (sintens(r->rcoef) <= FTINY)
		return;

	worldfunc(RCCONTEXT, r);		/* else set context */
	set_eparams((char *)mp->params);
	if ((bval = evalue(mp->binv)) <= -.5)	/* and get bin number */
		return;				/* silently ignore negatives */
	if ((bn = (int)(bval + .5)) >= mp->base_nbins) {
		sprintf(errmsg, "bad bin number (%d ignored)", bn);
		error(WARNING, errmsg);
		return;
	}
	raycontrib(contr, r, PRIMARY);		/* compute coefficient */
	if (contrib)
		smultscolor(contr, r->rcol);	/* -> contribution */
	if (RCO_BSDF_INSTRUMENT)
		fprintf(stderr,
			"RCO_BSDF_CPU_INSTRUMENT trace_contrib "
			"samplendx=%lu mod=%s obj=%s crtype=%u rsrc=%d "
			"rcoef=(%.9g,%.9g,%.9g) rcol=(%.9g,%.9g,%.9g) "
			"contr=(%.9g,%.9g,%.9g) contrib_mode=%d\n",
			samplendx, objptr(r->ro->omod)->oname, r->ro->oname,
			(unsigned int)r->crtype, r->rsrc,
			r->rcoef[0], r->rcoef[1], r->rcoef[2],
			r->rcol[0], r->rcol[1], r->rcol[2],
			contr[0], contr[1], contr[2], contrib);
	dvp = mcbin(mp, bn);
	for (i = 0; i < NCSAMP; i++)		/* add it in */
		*dvp++ += contr[i];
	if (rc_dual_path_output && rc_path_accept_direct(r)) {
		dvp = mcbin(mp, mp->base_nbins + bn);
		for (i = 0; i < NCSAMP; i++)
			*dvp++ += contr[i] * r->rpath_direct_ratio[i];
	} else if (rc_dual_source_output && r->rsrc >= 0 &&
			!(source[r->rsrc].sflags & SVIRTUAL) &&
			r->rpath_redirect_count == 0 &&
			r->rpath_diffuse_count == 0) {
		dvp = mcbin(mp, mp->base_nbins + bn);
		for (i = 0; i < NCSAMP; i++)
			*dvp++ += contr[i];
	}
}


/* Evaluate irradiance contributions */
static void
eval_irrad(FVECT org, FVECT dir)
{
	RAY	thisray;

	VSUM(thisray.rorg, org, dir, 1.1e-4);
	thisray.rdir[0] = -dir[0];
	thisray.rdir[1] = -dir[1];
	thisray.rdir[2] = -dir[2];
	thisray.rmax = 0.0;
	rayorigin(&thisray, PRIMARY, NULL, NULL);
					/* pretend we hit surface */
	thisray.rxt = thisray.rot = 1e-5;
	thisray.rod = 1.0;
	VCOPY(thisray.ron, dir);
	VSUM(thisray.rop, org, dir, 1e-4);
	samplendx++;			/* compute result */
	(*ofun[Lamb.otype].funp)(&Lamb, &thisray);
}


/* Evaluate radiance contributions */
static void
eval_rad(FVECT org, FVECT dir, double dmax)
{
	RAY	thisray;
					/* set up ray */
	VCOPY(thisray.rorg, org);
	VCOPY(thisray.rdir, dir);
	thisray.rmax = dmax;
	rayorigin(&thisray, PRIMARY, NULL, NULL);
	samplendx++;			/* call ray evaluation */
	rayvalue(&thisray);
}


/* Accumulate and/or output ray contributions (child or only process) */
static void
done_contrib(void)
{
	MODCONT	*mp;
	int	i;

	if (account <= 0 || --account)
		return;			/* not time yet */

	for (i = 0; i < nmods; i++) {	/* output records & clear */
		mp = (MODCONT *)lu_find(&modconttab,modname[i])->data;
		mod_output(mp);
		memset(mp->cbin, 0, DCOLORSIZ*mp->nbins);
	}
	end_record();			/* end lines & flush if time */

	account = accumulate;		/* reset accumulation counter */
}


/* Principal calculation loop (called by main) */
void
rcontrib(void)
{
	static int	ignore_warning_given = 0;
	FVECT		orig, direc;
	double		d;
					/* initialize (& fork more of us) */
	rcinit();
					/* load rays from stdin & process */
#ifdef getc_unlocked
	flockfile(stdin);		/* avoid mutex overhead */
#endif
	while (getvec(orig) == 0 && getvec(direc) == 0) {
		const int inactive_sample = rc_rflux_inactive_sentinel &&
				orig[0] == RFLUX_INACTIVE_X &&
				orig[1] == RFLUX_INACTIVE_Y &&
				orig[2] == RFLUX_INACTIVE_Z;
		d = normalize(direc);
		if (nchild != -1 && (d == 0.0) & (accumulate == 0)) {
			if (!ignore_warning_given++)
				error(WARNING,
				"dummy ray(s) ignored during accumulation\n");
			continue;
		}
		if (lastray+1 < lastray)
			lastray = lastdone = 0;
		++lastray;
		if (inactive_sample) {			/* masked sender sample */
			;
		} else if (d == 0.0) {			/* zero ==> flush */
			if ((yres <= 0) | (xres <= 1))
				waitflush = 1;		/* flush after */
			if (nchild == -1)
				account = 1;
		} else if (imm_irrad) {			/* else compute */
			eval_irrad(orig, direc);
		} else {
			eval_rad(orig, direc, lim_dist ? d : 0.0);
		}
		done_contrib();		/* accumulate/output */
		++lastdone;
		if (!morays())
			break;		/* preemptive EOI */
	}
	if (nchild != -1 && (accumulate <= 0) | (account < accumulate)) {
		if (account < accumulate) {
			error(WARNING, "partial accumulation in final record");
			accumulate -= account;
		}
		account = 1;		/* output accumulated totals */
		done_contrib();
	}
	lu_done(&ofiletab);		/* close output files */
	if (raysleft)
		error(USER, "unexpected EOF on input");
}
