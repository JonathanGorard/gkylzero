#include <gkyl_alloc.h>
#include <gkyl_array.h>
#include <gkyl_array_rio.h>
#include <gkyl_basis.h>
#include <gkyl_geo_gyrokinetic.h>
#include <gkyl_math.h>
#include <gkyl_range.h>
#include <gkyl_rect_grid.h>

#include <math.h>
#include <string.h>

struct gkyl_geo_gyrokinetic {
  struct gkyl_rect_grid rzgrid; // RZ grid on which psi(R,Z) is defined
  const struct gkyl_array *psiRZ; // psi(R,Z) DG representation
  struct gkyl_range rzlocal; // local range over which psiRZ is defined
  int num_rzbasis; // number of basis functions in RZ

  struct { int max_iter; double eps; } root_param;
  struct { int max_level; double eps; } quad_param;

  // pointer to root finder (depends on polyorder)
  struct RdRdZ_sol (*calc_roots)(const double *psi, double psi0, double Z,
    double xc[2], double dx[2]);

  struct gkyl_geo_gyrokinetic_stat stat; 
  double B0;
  double R0;
};

// some helper functions
static inline double
choose_closest(double ref, double R[2], double out[2])
{
  return fabs(R[0]-ref) < fabs(R[1]-ref) ? out[0] : out[1];
}

static inline double SQ(double x) { return x*x; }

static inline int
get_idx(int dir, double x, const struct gkyl_rect_grid *grid, const struct gkyl_range *range)
{
  double xlower = grid->lower[dir], dx = grid->dx[dir];
  int idx = range->lower[dir] + (int) floor((x-xlower)/dx);
  return idx <= range->upper[dir] ? idx : range->upper[dir];
}

// struct for solutions to roots
struct RdRdZ_sol {
  int nsol;
  double R[2], dRdZ[2];
};

// Compute roots R(psi,Z) and dR/dZ(psi,Z) in a p=1 DG cell
static inline struct RdRdZ_sol
calc_RdR_p1(const double *psi, double psi0, double Z, double xc[2], double dx[2])
{
  struct RdRdZ_sol sol = { .nsol = 0 };

  double y = (Z-xc[1])/(dx[1]*0.5);
  
  double rnorm = (-(1.732050807568877*psi[2]*y)/(3.0*psi[3]*y+1.732050807568877*psi[1]))+(2.0*psi0)/(3.0*psi[3]*y+1.732050807568877*psi[1])-(1.0*psi[0])/(3.0*psi[3]*y+1.732050807568877*psi[1]) ;

  if ((-1<=rnorm) && (rnorm < 1)) {
    double drdznorm = -(3.0*(2.0*psi[3]*psi0-1.0*psi[0]*psi[3]+psi[1]*psi[2]))/SQ(3.0*psi[3]*y+1.732050807568877*psi[1]) ;
    
    sol.nsol = 1;
    sol.R[0] = rnorm*dx[0]*0.5 + xc[0];
    sol.dRdZ[0] = drdznorm*dx[0]/dx[1];
  }
  return sol;
}

// Compute roots R(psi,Z) and dR/dZ(psi,Z) in a p=2 DG cell
static inline struct RdRdZ_sol
calc_RdR_p2(const double *psi, double psi0, double Z, double xc[2], double dx[2])
{
  struct RdRdZ_sol sol = { .nsol = 0 };
  double y = (Z-xc[1])/(dx[1]*0.5);

  double aq = 2.904737509655563*psi[6]*y+1.677050983124842*psi[4]; 
  double bq = 2.904737509655563*psi[7]*SQ(y)+1.5*psi[3]*y-0.9682458365518543*psi[7]+0.8660254037844386*psi[1]; 
  double cq = 1.677050983124842*psi[5]*SQ(y)-0.9682458365518543*psi[6]*y+0.8660254037844386*psi[2]*y-1.0*psi0-0.5590169943749475*psi[5]-0.5590169943749475*psi[4]+0.5*psi[0]; 
  double delta2 = bq*bq - 4*aq*cq;

  if (delta2 > 0) {
    double r1, r2;
    double delta = sqrt(delta2);
    // compute both roots
    if (bq>=0) {
      r1 = (-bq-delta)/(2*aq);
      r2 = 2*cq/(-bq-delta);
    }
    else {
      r1 = 2*cq/(-bq+delta);
      r2 = (-bq+delta)/(2*aq);
    }

    int sidx = 0;
    if ((-1<=r1) && (r1 < 1)) {
      sol.nsol += 1;
      sol.R[sidx] = r1*dx[0]*0.5 + xc[0];

      double x = r1;
      double C = 5.809475019311126*psi[7]*x*y+3.354101966249685*psi[5]*y+2.904737509655563*psi[6]*SQ(x)+1.5*psi[3]*x-0.9682458365518543*psi[6]+0.8660254037844386*psi[2]; 
      double A = 2.904737509655563*psi[7]*SQ(y)+5.809475019311126*psi[6]*x*y+1.5*psi[3]*y+3.354101966249685*psi[4]*x-0.9682458365518543*psi[7]+0.8660254037844386*psi[1];
      sol.dRdZ[sidx] = -C/A*dx[0]/dx[1];
      
      sidx += 1;
    }
    if ((-1<=r2) && (r2 < 1)) {
      sol.nsol += 1;
      sol.R[sidx] = r2*dx[0]*0.5 + xc[0];

      double x = r2;
      double C = 5.809475019311126*psi[7]*x*y+3.354101966249685*psi[5]*y+2.904737509655563*psi[6]*SQ(x)+1.5*psi[3]*x-0.9682458365518543*psi[6]+0.8660254037844386*psi[2]; 
      double A = 2.904737509655563*psi[7]*SQ(y)+5.809475019311126*psi[6]*x*y+1.5*psi[3]*y+3.354101966249685*psi[4]*x-0.9682458365518543*psi[7]+0.8660254037844386*psi[1];
      sol.dRdZ[sidx] = -C/A*dx[0]/dx[1];
      
      sidx += 1;
    }
  }
  return sol;
}

// Compute R(psi,Z) given a psi and Z. Can return multiple solutions
// or no solutions. The number of roots found is returned and are
// copied in the array R and dR. The calling function must ensure that
// these arrays are big enough to hold all roots required
static int
R_psiZ(const gkyl_geo_gyrokinetic *geo, double psi, double Z, int nmaxroots,
  double *R, double *dR)
{
  int zcell = get_idx(1, Z, &geo->rzgrid, &geo->rzlocal);

  int sidx = 0;
  int idx[2] = { 0, zcell };
  double dx[2] = { geo->rzgrid.dx[0], geo->rzgrid.dx[1] };
  
  struct gkyl_range rangeR;
  gkyl_range_deflate(&rangeR, &geo->rzlocal, (int[]) { 0, 1 }, (int[]) { 0, zcell });

  struct gkyl_range_iter riter;
  gkyl_range_iter_init(&riter, &rangeR);
  
  // loop over all R cells to find psi crossing
  while (gkyl_range_iter_next(&riter) && sidx<=nmaxroots) {
    long loc = gkyl_range_idx(&rangeR, riter.idx);
    const double *psih = gkyl_array_cfetch(geo->psiRZ, loc);

    double xc[2];
    idx[0] = riter.idx[0];
    gkyl_rect_grid_cell_center(&geo->rzgrid, idx, xc);

    struct RdRdZ_sol sol = geo->calc_roots(psih, psi, Z, xc, dx);
    
    if (sol.nsol > 0)
      for (int s=0; s<sol.nsol; ++s) {
        R[sidx] = sol.R[s];
        dR[sidx] = sol.dRdZ[s];
        sidx += 1;
      }
  }
  return sidx;
}

// Function context to pass to coutour integration function
struct contour_ctx {
  const gkyl_geo_gyrokinetic *geo;
  double psi, last_R;
  long ncall;
};

// Function to pass to numerical quadrature to integrate along a contour
static inline double
contour_func(double Z, void *ctx)
{
  struct contour_ctx *c = ctx;
  c->ncall += 1;
  double R[2] = { 0 }, dR[2] = { 0 };
  
  int nr = R_psiZ(c->geo, c->psi, Z, 2, R, dR);
  double dRdZ = nr == 1 ? dR[0] : choose_closest(c->last_R, R, dR);
  
  return nr>0 ? sqrt(1+dRdZ*dRdZ) : 0.0;
}

static inline double
phi_contour_func(double Z, void *ctx)
{
  struct contour_ctx *c = ctx;
  c->ncall += 1;
  double R[2] = { 0 }, dR[2] = { 0 };
  
  int nr = R_psiZ(c->geo, c->psi, Z, 2, R, dR);
  double dRdZ = nr == 1 ? dR[0] : choose_closest(c->last_R, R, dR);
  double r_curr = nr == 1 ? R[0] : choose_closest(c->last_R, R, R);

  struct gkyl_range_iter iter;
  iter.idx[0] = fmin(c->geo->rzlocal.lower[0] + (int) floor((r_curr - c->geo->rzgrid.lower[0])/c->geo->rzgrid.dx[0]), c->geo->rzlocal.upper[0]);
  iter.idx[1] = fmin(c->geo->rzlocal.lower[1] + (int) floor((Z - c->geo->rzgrid.lower[1])/c->geo->rzgrid.dx[1]), c->geo->rzlocal.upper[1]);
  long loc = gkyl_range_idx(&(c->geo->rzlocal), iter.idx);
  const double *psih = gkyl_array_cfetch(c->geo->psiRZ, loc);

  double xc[2];
  gkyl_rect_grid_cell_center(&(c->geo->rzgrid), iter.idx, xc);
  double x = (r_curr-xc[0])/(c->geo->rzgrid.dx[0]*0.5);
  double y = (Z-xc[1])/(c->geo->rzgrid.dx[1]*0.5);

  // if psi is polyorder 2 we can get grad psi
  // in cylindrical coords it is grad psi = dpsi/dR Rhat + dpsi/dZ zhat
  double dpsidx = 2.904737509655563*psih[7]*(y*y-0.3333333333333333)+5.809475019311126*psih[6]*x*y+1.5*psih[3]*y+3.354101966249684*psih[4]*x+0.8660254037844386*psih[1]; 
  double dpsidy =	5.809475019311126*psih[7]*x*y+3.354101966249684*psih[5]*y+2.904737509655563*psih[6]*(x*x-0.3333333333333333)+1.5*psih[3]*x+0.8660254037844386*psih[2];
  dpsidx = dpsidx*2.0/c->geo->rzgrid.dx[0];
  dpsidy = dpsidy*2.0/c->geo->rzgrid.dx[1];
  double grad_psi_mag = sqrt(dpsidx*dpsidx + dpsidy*dpsidy);
  double result  = (1/r_curr/sqrt(dpsidx*dpsidx + dpsidy*dpsidy)) *sqrt(1+dRdZ*dRdZ) ;
  return nr>0 ? result : 0.0;
}

// Integrates along a specified contour, optionally using a "memory"
// of previously computed values, or storing computed values in
// memory. The function basically breaks up the integral into a loop
// over z-cells. This needs to be done as the DG representation is,
// well, discontinuous, and adaptive quadrature struggles with such
// functions.
static double
integrate_psi_contour_memo(const gkyl_geo_gyrokinetic *geo, double psi,
  double zmin, double zmax, double rclose,
  bool use_memo, bool fill_memo, double *memo)
{
  struct contour_ctx ctx = {
    .geo = geo,
    .psi = psi,
    .ncall = 0,
    .last_R = rclose
  };

  int nlevels = geo->quad_param.max_level;
  double eps = geo->quad_param.eps;
  
  double dz = geo->rzgrid.dx[1];
  double zlo = geo->rzgrid.lower[1];
  int izlo = geo->rzlocal.lower[1], izup = geo->rzlocal.upper[1];
  
  int ilo = get_idx(1, zmin, &geo->rzgrid, &geo->rzlocal);
  int iup = get_idx(1, zmax, &geo->rzgrid, &geo->rzlocal);

  double res = 0.0;
  for (int i=ilo; i<=iup; ++i) {
    double z1 = gkyl_median(zmin, zlo+(i-izlo)*dz, zlo+(i-izlo+1)*dz);
    double z2 = gkyl_median(zmax, zlo+(i-izlo)*dz, zlo+(i-izlo+1)*dz);
    
    if (z1 < z2) {
      if (use_memo) {
        if (fill_memo) {
          struct gkyl_qr_res res_local =
            gkyl_dbl_exp(contour_func, &ctx, z1, z2, nlevels, eps);
          memo[i-izlo] = res_local.res;
          res += res_local.res;
        }
        else {
          if (z2-z1 == dz) {
            res += memo[i-izlo];
          }
          else {
            struct gkyl_qr_res res_local =
              gkyl_dbl_exp(contour_func, &ctx, z1, z2, nlevels, eps);
            res += res_local.res;
          }
        }
      }
      else {
        struct gkyl_qr_res res_local =
          gkyl_dbl_exp(contour_func, &ctx, z1, z2, nlevels, eps);
        res += res_local.res;
      }
    }
  }

  ((gkyl_geo_gyrokinetic *)geo)->stat.nquad_cont_calls += ctx.ncall;
  return res;
}

static double
integrate_phi_along_psi_contour_memo(const gkyl_geo_gyrokinetic *geo, double psi,
  double zmin, double zmax, double rclose,
  bool use_memo, bool fill_memo, double *memo)
{
  struct contour_ctx ctx = {
    .geo = geo,
    .psi = psi,
    .ncall = 0,
    .last_R = rclose
  };

  int nlevels = geo->quad_param.max_level;
  double eps = geo->quad_param.eps;
  
  double dz = geo->rzgrid.dx[1];
  double zlo = geo->rzgrid.lower[1];
  int izlo = geo->rzlocal.lower[1], izup = geo->rzlocal.upper[1];
  
  int ilo = get_idx(1, zmin, &geo->rzgrid, &geo->rzlocal);
  int iup = get_idx(1, zmax, &geo->rzgrid, &geo->rzlocal);

  double res = 0.0;
  for (int i=ilo; i<=iup; ++i) {
    double z1 = gkyl_median(zmin, zlo+(i-izlo)*dz, zlo+(i-izlo+1)*dz);
    double z2 = gkyl_median(zmax, zlo+(i-izlo)*dz, zlo+(i-izlo+1)*dz);
    
    if (z1 < z2) {
      if (use_memo) {
        if (fill_memo) {
          struct gkyl_qr_res res_local =
            gkyl_dbl_exp(phi_contour_func, &ctx, z1, z2, nlevels, eps);
          memo[i-izlo] = res_local.res;
          res += res_local.res;
        }
        else {
          if (z2-z1 == dz) {
            res += memo[i-izlo];
          }
          else {
            struct gkyl_qr_res res_local =
              gkyl_dbl_exp(phi_contour_func, &ctx, z1, z2, nlevels, eps);
            res += res_local.res;
          }
        }
      }
      else {
        struct gkyl_qr_res res_local =
          gkyl_dbl_exp(phi_contour_func, &ctx, z1, z2, nlevels, eps);
        res += res_local.res;
      }
    }
  }

  ((gkyl_geo_gyrokinetic *)geo)->stat.nquad_cont_calls += ctx.ncall;
  return res;
}

// Function context to pass to root finder
struct arc_length_ctx {
  const gkyl_geo_gyrokinetic *geo;
  double *arc_memo;
  double psi, rclose, zmin, arcL;
};


// Function to pass to root-finder to find Z location for given arc-length
static inline double
arc_length_func(double Z, void *ctx)
{
  struct arc_length_ctx *actx = ctx;
  double *arc_memo = actx->arc_memo;
  double psi = actx->psi, rclose = actx->rclose, zmin = actx->zmin, arcL = actx->arcL;
  double ival = integrate_psi_contour_memo(actx->geo, psi, zmin, Z, rclose,
    true, false, arc_memo) - arcL;
  return ival;
}

// Function to calculate phi given alpha
static inline double
phi_func(double alpha_curr, double Z, void *ctx)
{
  struct arc_length_ctx *actx = ctx;
  double *arc_memo = actx->arc_memo;
  double psi = actx->psi, rclose = actx->rclose, zmin = actx->zmin, arcL = actx->arcL;

  // Using convention from Noah Mandell's thesis Eq 5.104 phi = alpha at midplane
  double ival = 0;
  if(Z<0.0){
    ival = -integrate_phi_along_psi_contour_memo(actx->geo, psi, Z, 0.0, rclose, false, false, arc_memo);
  }
  else{
    ival = integrate_phi_along_psi_contour_memo(actx->geo, psi, 0.0, Z, rclose, false, false, arc_memo);
  }

  // Now multiply by RBphi
  double R[2] = {0};
  double dR[2] = {0};
  int nr = R_psiZ(actx->geo, psi, Z, 2, R, dR);
  double r_curr = nr == 1 ? R[0] : choose_closest(rclose, R, R);
  double Bphi = actx->geo->B0*actx->geo->R0/r_curr;
  ival = ival*r_curr*Bphi;

  // now keep in range 2pi
  while(ival < -M_PI){
    ival +=2*M_PI;
  }
  while(ival > M_PI){
    ival -=2*M_PI;
  }
  return alpha_curr + ival;
}



gkyl_geo_gyrokinetic*
gkyl_geo_gyrokinetic_new(const struct gkyl_geo_gyrokinetic_inp *inp)
{
  struct gkyl_geo_gyrokinetic *geo = gkyl_malloc(sizeof(*geo));

  geo->B0 = inp->B0;
  geo->R0 = inp->R0;

  geo->rzgrid = *inp->rzgrid;
  geo->psiRZ = gkyl_array_acquire(inp->psiRZ);
  geo->num_rzbasis = inp->rzbasis->num_basis;
  memcpy(&geo->rzlocal, inp->rzlocal, sizeof(struct gkyl_range));

  geo->root_param.eps =
    inp->root_param.eps > 0 ? inp->root_param.eps : 1e-10;
  geo->root_param.max_iter =
    inp->root_param.max_iter > 0 ? inp->root_param.max_iter : 100;

  geo->quad_param.max_level =
    inp->quad_param.max_levels > 0 ? inp->quad_param.max_levels : 10;
  geo->quad_param.eps =
    inp->quad_param.eps > 0 ? inp->quad_param.eps : 1e-10;

  if (inp->rzbasis->poly_order == 1)
    geo->calc_roots = calc_RdR_p1;
  else if (inp->rzbasis->poly_order == 2)
    geo->calc_roots = calc_RdR_p2;

  geo->stat = (struct gkyl_geo_gyrokinetic_stat) { };
  
  return geo;
}

double
gkyl_geo_gyrokinetic_integrate_psi_contour(const gkyl_geo_gyrokinetic *geo, double psi,
  double zmin, double zmax, double rclose)
{
  return integrate_psi_contour_memo(geo, psi, zmin, zmax, rclose,
    false, false, 0);
}

int
gkyl_geo_gyrokinetic_R_psiZ(const gkyl_geo_gyrokinetic *geo, double psi, double Z, int nmaxroots,
  double *R, double *dR)
{
  return R_psiZ(geo, psi, Z, nmaxroots, R, dR);
}

// write out nodal coordinates 
static void
write_nodal_coordinates(const char *nm, struct gkyl_range *nrange,
  struct gkyl_array *nodes)
{
  double lower[3] = { 0.0, 0.0, 0.0 };
  double upper[3] = { 1.0, 1.0, 1.0 };
  int cells[3];
  for (int i=0; i<nrange->ndim; ++i)
    cells[i] = gkyl_range_shape(nrange, i);
  
  struct gkyl_rect_grid grid;
  gkyl_rect_grid_init(&grid, 3, lower, upper, cells);

  gkyl_grid_sub_array_write(&grid, nrange, nodes, nm);
}




void nodal_array_to_modal_array(struct gkyl_array *nodal_array, struct gkyl_array *modal_array, struct gkyl_range *update_range, struct gkyl_range *nrange, const struct gkyl_geo_gyrokinetic_geo_inp *ginp){
  double xc[GKYL_MAX_DIM], xmu[GKYL_MAX_DIM];

  int num_ret_vals = ginp->cgrid->ndim;
  int num_basis = ginp->cbasis->num_basis;
  int cpoly_order = ginp->cbasis->poly_order;
  printf("converting to modal, numBasis = %d\n", num_basis);
  //initialize the nodes
  struct gkyl_array *nodes = gkyl_array_new(GKYL_DOUBLE, ginp->cgrid->ndim, ginp->cbasis->num_basis);
  ginp->cbasis->node_list(gkyl_array_fetch(nodes, 0));
  double fnodal[num_basis]; // to store nodal function values

  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, update_range);
  int nidx[3];
  long lin_nidx[num_basis];
  
  while (gkyl_range_iter_next(&iter)) {
     gkyl_rect_grid_cell_center(ginp->cgrid, iter.idx, xc);

    for (int i=0; i<num_basis; ++i) {
      const double* temp  = gkyl_array_cfetch(nodes,i);
      for( int j = 0; j < ginp->cgrid->ndim; j++){
        if(cpoly_order==1)
          nidx[j] = iter.idx[j] + (temp[j] + 1)/2 ;
        if (cpoly_order==2)
          nidx[j] = 2*iter.idx[j] + (temp[j] + 1) ;
      }
      lin_nidx[i] = gkyl_range_idx(nrange, nidx);
    }

    long lidx = gkyl_range_idx(update_range, iter.idx);
    double *arr_p = gkyl_array_fetch(modal_array, lidx); // pointer to expansion in cell
    double fao[num_basis*num_ret_vals];
  
    for (int i=0; i<num_basis; ++i) {
      double* temp = gkyl_array_fetch(nodal_array, lin_nidx[i]);
      for (int j=0; j<num_ret_vals; ++j) {
        fao[i*num_ret_vals + j] = temp[j];
      }
    }

    for (int i=0; i<num_ret_vals; ++i) {
      // copy so nodal values for each return value are contiguous
      // (recall that function can have more than one return value)
      for (int k=0; k<num_basis; ++k)
        fnodal[k] = fao[num_ret_vals*k+i];
      // transform to modal expansion
      ginp->cbasis->nodal_to_modal(fnodal, &arr_p[num_basis*i]);
    }
  }

}





void
gkyl_geo_gyrokinetic_calcgeom(const gkyl_geo_gyrokinetic *geo,
  const struct gkyl_geo_gyrokinetic_geo_inp *inp, struct gkyl_array *mapc2p, struct gkyl_range *conversion_range)
{
  int poly_order = inp->cbasis->poly_order;
  int nodes[3] = { 1, 1, 1 };
  if (poly_order == 1)
    for (int d=0; d<inp->cgrid->ndim; ++d)
      nodes[d] = inp->cgrid->cells[d]+2 + 1;
  if (poly_order == 2)
    for (int d=0; d<inp->cgrid->ndim; ++d)
      nodes[d] = 2*(inp->cgrid->cells[d]+2) + 1;

  for(int d=0; d<inp->cgrid->ndim; d++){
    printf("d[%d] = %d\n", d, nodes[d]);
  }

  struct gkyl_range nrange;
  gkyl_range_init_from_shape(&nrange, inp->cgrid->ndim, nodes);
  struct gkyl_array *mc2p = gkyl_array_new(GKYL_DOUBLE, inp->cgrid->ndim, nrange.volume);
  printf("cgrid ndim  = %d\n", inp->cgrid->ndim);

  printf("Checking the range volumes\n nrange.volume = %ld\n conversion_range.volume = %ld\n", nrange.volume, conversion_range->volume);

  struct gkyl_array *mc2p_xyz = gkyl_array_new(GKYL_DOUBLE, inp->cgrid->ndim, nrange.volume);

  enum { PH_IDX, AL_IDX, TH_IDX }; // arrangement of computational coordinates
  enum { R_IDX, Z_IDX }; // arrangement of physical coordinates  
  enum { X_IDX, Y_IDX, Zc_IDX }; // arrangement of cartesian coordinates
  
  double dtheta = inp->cgrid->dx[TH_IDX],
    dphi = inp->cgrid->dx[PH_IDX],
    dalpha = inp->cgrid->dx[AL_IDX];
  
  double theta_lo = inp->cgrid->lower[TH_IDX] - dtheta,
    phi_lo = inp->cgrid->lower[PH_IDX] - dphi,
    alpha_lo = inp->cgrid->lower[AL_IDX] - dalpha;

  double dx_fact = poly_order == 1 ? 1 : 0.5;
  dtheta *= dx_fact; dphi *= dx_fact; dalpha *= dx_fact;

  double rclose = inp->rclose;

  int nzcells = geo->rzgrid.cells[1];
  double *arc_memo = gkyl_malloc(sizeof(double[nzcells]));

  struct arc_length_ctx arc_ctx = {
    .geo = geo,
    .arc_memo = arc_memo
  };

  int cidx[3] = { 0 };
  
  for(int ia=nrange.lower[AL_IDX]; ia<=nrange.upper[AL_IDX]; ++ia){
    cidx[AL_IDX] = ia;
    double alpha_curr = alpha_lo + ia*dalpha;
    for (int ip=nrange.lower[PH_IDX]; ip<=nrange.upper[PH_IDX]; ++ip) {

      double zmin = inp->zmin, zmax = inp->zmax;

      double psi_curr = phi_lo + ip*dphi;
      printf("psi_curr = %g\n", psi_curr);
      double arcL = integrate_psi_contour_memo(geo, psi_curr, zmin, zmax, rclose,
        true, true, arc_memo);

      double delta_arcL = arcL/(poly_order*inp->cgrid->cells[TH_IDX]) * (inp->cgrid->upper[TH_IDX] - inp->cgrid->lower[TH_IDX])/2/M_PI;
      double delta_theta = delta_arcL*(2*M_PI/arcL);

      cidx[PH_IDX] = ip;


      double arcL_curr = 0.0;
      arcL_curr = (theta_lo + M_PI)/2/M_PI*arcL;
      double theta_curr = arcL_curr*(2*M_PI/arcL) - M_PI ;
      do {
        // set node coordinates of first node
        cidx[TH_IDX] = nrange.lower[TH_IDX];
        double *mc2p_n = gkyl_array_fetch(mc2p, gkyl_range_idx(&nrange, cidx));
        double R[2] = { 0 }, dR[2] = { 0 };
        int nr = R_psiZ(geo, psi_curr, zmin, 2, R, dR);
        double r_curr = choose_closest(rclose, R, R);
        mc2p_n[Z_IDX] = zmin;
        mc2p_n[R_IDX] = r_curr;

        arc_ctx.psi = psi_curr;
        arc_ctx.rclose = rclose;
        arc_ctx.zmin = zmin;
        arc_ctx.arcL = arcL_curr;

        double phi_curr = phi_func(alpha_curr, zmin, &arc_ctx);
        // convert to x,y,z
        double *mc2p_xyz_n = gkyl_array_fetch(mc2p_xyz, gkyl_range_idx(&nrange, cidx));
        mc2p_xyz_n[X_IDX] = mc2p_n[R_IDX]*cos(phi_curr);
        mc2p_xyz_n[Y_IDX] = mc2p_n[R_IDX]*sin(phi_curr);
        mc2p_xyz_n[Zc_IDX] = mc2p_n[Z_IDX];
      } while(0);

      // set node coordinates of rest of nodes
      for (int it=nrange.lower[TH_IDX]+1; it<nrange.upper[TH_IDX]; ++it) {
        arcL_curr += delta_arcL;
        double theta_curr = arcL_curr*(2*M_PI/arcL) - M_PI ; // this is wrong need total arcL factor. Edit: 8/23 AS Not sure about this comment, shold have put a date in original. Seems to work fine.
        //printf("theta_curr = %g, psicurr  = %g \n", theta_curr, psi_curr);

        arc_ctx.psi = psi_curr;
        arc_ctx.rclose = rclose;
        arc_ctx.zmin = zmin;
        arc_ctx.arcL = arcL_curr;

        struct gkyl_qr_res res = gkyl_ridders(arc_length_func, &arc_ctx,
          zmin, zmax, -arcL_curr, arcL-arcL_curr,
          geo->root_param.max_iter, 1e-10);
        double z_curr = res.res;
        ((gkyl_geo_gyrokinetic *)geo)->stat.nroot_cont_calls += res.nevals;

        double R[2] = { 0 }, dR[2] = { 0 };
        int nr = R_psiZ(geo, psi_curr, z_curr, 2, R, dR);
        double r_curr = choose_closest(rclose, R, R);

        cidx[TH_IDX] = it;
        double *mc2p_n = gkyl_array_fetch(mc2p, gkyl_range_idx(&nrange, cidx));
        mc2p_n[Z_IDX] = z_curr;
        mc2p_n[R_IDX] = r_curr;

        double phi_curr = phi_func(alpha_curr, z_curr, &arc_ctx);

        // convert to x,y,z
        double *mc2p_xyz_n = gkyl_array_fetch(mc2p_xyz, gkyl_range_idx(&nrange, cidx));
        mc2p_xyz_n[X_IDX] = mc2p_n[R_IDX]*cos(phi_curr);
        mc2p_xyz_n[Y_IDX] = mc2p_n[R_IDX]*sin(phi_curr);
        mc2p_xyz_n[Zc_IDX] = mc2p_n[Z_IDX];
        
      }

      arcL_curr += delta_arcL;
      theta_curr = arcL_curr*(2*M_PI/arcL) - M_PI ;
      do {
        // set node coordinates of last node
        cidx[TH_IDX] = nrange.upper[TH_IDX];
        double *mc2p_n = gkyl_array_fetch(mc2p, gkyl_range_idx(&nrange, cidx));
        mc2p_n[Z_IDX] = zmax;
        double R[2] = { 0 }, dR[2] = { 0 };    
        int nr = R_psiZ(geo, psi_curr, zmax, 2, R, dR);
        mc2p_n[R_IDX] = choose_closest(rclose, R, R);

        // need to set the arc ctx for phi
        arc_ctx.psi = psi_curr;
        arc_ctx.rclose = rclose;
        arc_ctx.zmin = zmin;
        arc_ctx.arcL = arcL_curr;
        double phi_curr = phi_func(alpha_curr, zmax, &arc_ctx);

        //do x,y,z
        double *mc2p_xyz_n = gkyl_array_fetch(mc2p_xyz, gkyl_range_idx(&nrange, cidx));
        mc2p_xyz_n[X_IDX] = mc2p_n[R_IDX]*cos(phi_curr);
        mc2p_xyz_n[Y_IDX] = mc2p_n[R_IDX]*sin(phi_curr);
        mc2p_xyz_n[Zc_IDX] = mc2p_n[Z_IDX];
      } while (0);
      //printf("last node theta_curr = %g, psicurr  = %g \n\n", theta_curr, psi_curr);
    }
    //end original loop
  }

  //printf("trying to write nodal coords\n");
  char str1[50] = "xyz";
  if (inp->write_node_coord_array){
    write_nodal_coordinates(inp->node_file_nm, &nrange, mc2p);
    write_nodal_coordinates(strcat(str1, inp->node_file_nm), &nrange, mc2p_xyz);
  }
  //printf("done writing nodal coords\n");

  nodal_array_to_modal_array(mc2p_xyz, mapc2p, conversion_range, &nrange, inp);
  //printf("converted to modal\n");

  gkyl_free(arc_memo);
  gkyl_array_release(mc2p);  
  gkyl_array_release(mc2p_xyz);  
}

struct gkyl_geo_gyrokinetic_stat
gkyl_geo_gyrokinetic_get_stat(const gkyl_geo_gyrokinetic *geo)
{
  return geo->stat;
}

void
gkyl_geo_gyrokinetic_release(gkyl_geo_gyrokinetic *geo)
{
  gkyl_array_release(geo->psiRZ);
  gkyl_free(geo);
}