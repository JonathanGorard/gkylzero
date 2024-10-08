#include <gkyl_alloc.h>
#include <gkyl_array.h>
#include <gkyl_array_rio.h>
#include <gkyl_basis.h>
#include <gkyl_math.h>
#include <gkyl_range.h>
#include <gkyl_rect_grid.h>
#include <gkyl_nodal_ops.h>
#include <gkyl_gk_geometry.h>
#include <gkyl_efit.h>
#include <gkyl_tok_geo_priv.h>
#include <assert.h>

#include <math.h>
#include <string.h>




double
tok_plate_psi_func(double s, void *ctx){
  // uses a pointer to the plate function to get R(s), Z(s)
  // Then calculates psi(R, Z)
  // will be used by ridders later
  
  struct plate_ctx *gc = ctx;
  double RZ[2];
  if(gc->lower==true)
    gc->geo->plate_func_lower(s, RZ);
  else
    gc->geo->plate_func_upper(s, RZ);

  double R = RZ[0];
  double Z = RZ[1];

  // Now find the cell where this R and Z is
  int rzidx[2];
  rzidx[0] = fmin(gc->geo->rzlocal.lower[0] + (int) floor((R - gc->geo->rzgrid.lower[0])/gc->geo->rzgrid.dx[0]), gc->geo->rzlocal.upper[0]);
  rzidx[1] = fmin(gc->geo->rzlocal.lower[1] + (int) floor((Z - gc->geo->rzgrid.lower[1])/gc->geo->rzgrid.dx[1]), gc->geo->rzlocal.upper[1]);
  long loc = gkyl_range_idx(&gc->geo->rzlocal, rzidx);
  const double *coeffs = gkyl_array_cfetch(gc->geo->psiRZ,loc);

  double xc[2];
  gkyl_rect_grid_cell_center(&gc->geo->rzgrid, rzidx, xc);
  double xy[2];
  xy[0] = (R-xc[0])/(gc->geo->rzgrid.dx[0]*0.5);
  xy[1] = (Z-xc[1])/(gc->geo->rzgrid.dx[1]*0.5);
  double psi = gc->geo->rzbasis.eval_expand(xy, coeffs);
  return psi - gc->psi_curr;
}



// Function to pass to root-finder to find Z location for given arc-length
static inline double
arc_length_func(double Z, void *ctx)
{
  struct arc_length_ctx *actx = ctx;
  double *arc_memo;
  double psi = actx->psi, rclose = actx->rclose, zmin = actx->zmin, arcL = actx->arcL;
  double zmax = actx->zmax;
  double ival = 0.0;

  if(actx->ftype==GKYL_CORE){
    if(actx->right==true){
      double *arc_memo = actx->arc_memo_right;
      ival = integrate_psi_contour_memo(actx->geo, psi, zmin, Z, rclose, true, false, arc_memo) - arcL;
    }
    else{
      double *arc_memo = actx->arc_memo_left;
      ival = integrate_psi_contour_memo(actx->geo, psi, Z, zmax, rclose, true, false, arc_memo)  - arcL + actx->arcL_right;
    }
  }
  else if(actx->ftype==GKYL_CORE_L){
    double *arc_memo = actx->arc_memo;
    ival = integrate_psi_contour_memo(actx->geo, psi, Z, zmax, rclose, true, false, arc_memo)  - arcL;
  }

  else if(actx->ftype==GKYL_CORE_R){
    double *arc_memo = actx->arc_memo;
    ival = integrate_psi_contour_memo(actx->geo, psi, zmin, Z, rclose, true, false, arc_memo) - arcL;
  }

  else if(actx->ftype==GKYL_PF_LO_L){
    double *arc_memo = actx->arc_memo;
    ival = integrate_psi_contour_memo(actx->geo, psi, Z, zmax, rclose, false, false, arc_memo)  - arcL;
  }
  else if(actx->ftype==GKYL_PF_LO_R){
    double *arc_memo = actx->arc_memo;
    ival = integrate_psi_contour_memo(actx->geo, psi, zmin, Z, rclose, false, false, arc_memo) - arcL;
  }
  else if(actx->ftype==GKYL_PF_UP_L){
    double *arc_memo = actx->arc_memo;
    ival = integrate_psi_contour_memo(actx->geo, psi, Z, zmax, rclose, false, false, arc_memo) - arcL;
  }
  else if(actx->ftype==GKYL_PF_UP_R){
    double *arc_memo = actx->arc_memo;
    ival = integrate_psi_contour_memo(actx->geo, psi, zmin, Z, rclose, false, false, arc_memo) - arcL;
  }
  else if( (actx->ftype==GKYL_SOL_DN_OUT) || (actx->ftype==GKYL_SOL_DN_OUT) || (actx->ftype==GKYL_SOL_DN_OUT_LO) || (actx->ftype==GKYL_SOL_DN_OUT_MID) || (actx->ftype==GKYL_SOL_DN_OUT_UP) ){
    double *arc_memo = actx->arc_memo;
    ival = integrate_psi_contour_memo(actx->geo, psi, zmin, Z, rclose, true, false, arc_memo) - arcL;
  }
  else if( (actx->ftype==GKYL_SOL_DN_IN) || (actx->ftype==GKYL_SOL_DN_IN) || (actx->ftype==GKYL_SOL_DN_IN_LO) || (actx->ftype==GKYL_SOL_DN_IN_MID) || (actx->ftype==GKYL_SOL_DN_IN_UP) ){
    double *arc_memo = actx->arc_memo;
    ival = integrate_psi_contour_memo(actx->geo, psi, Z, zmax, rclose, true, false, arc_memo) - arcL;
  }

  else if(actx->ftype==GKYL_SOL_SN_LO){
    if(actx->right==true){
      double *arc_memo = actx->arc_memo_right;
      ival = integrate_psi_contour_memo(actx->geo, psi, zmin, Z, rclose, false, false, arc_memo) - arcL;
    }
    else{
      double *arc_memo = actx->arc_memo_left;
      ival = integrate_psi_contour_memo(actx->geo, psi, Z, zmax, rclose, false, false, arc_memo)  - arcL + actx->arcL_right;
    }
  }

  return ival;
}

// Function to calculate phi given alpha
double
phi_func(double alpha_curr, double Z, void *ctx)
{
  struct arc_length_ctx *actx = ctx;
  double *arc_memo = actx->arc_memo;
  double psi = actx->psi, rclose = actx->rclose, zmin = actx->zmin, arcL = actx->arcL, zmax = actx->zmax;

  // Here we will abandon conventions about alpha and phi except for full core and full SN cases
  // The convention for phi only affects b_x - it does not affect any quantities used in axisymmetric simulations
  // I have not quite figured out full 3D yet. b_x presents a serious problem as of now. Akash Shukla 1/20/2024
  // The idea for axisymmetry is that I am avoiding starting integrals at the x-point to minimize issues
  double ival = 0;
  double phi_ref = 0.0;
  if (actx->ftype==GKYL_CORE){ 
    if(actx->right==true){ // phi = alpha at outboard midplane
      if(Z<actx->zmaxis)
        ival = -integrate_phi_along_psi_contour_memo(actx->geo, psi, Z, actx->zmaxis, rclose, false, false, arc_memo);
      else
        ival = integrate_phi_along_psi_contour_memo(actx->geo, psi, actx->zmaxis, Z, rclose, false, false, arc_memo);
    }
    else{// alpha = phi at inboard midplane
      if (Z<actx->zmaxis)
        ival = integrate_phi_along_psi_contour_memo(actx->geo, psi, Z, actx->zmaxis, rclose, false, false, arc_memo);
      else
        ival = -integrate_phi_along_psi_contour_memo(actx->geo, psi, actx->zmaxis, Z, rclose, false, false, arc_memo);
    }
  }
  else if (actx->ftype==GKYL_CORE_L){ 
    //if (Z<actx->zmaxis)
    //  ival = integrate_phi_along_psi_contour_memo(actx->geo, psi, Z, actx->zmaxis, rclose, false, false, arc_memo);
    //else
    //  ival = -integrate_phi_along_psi_contour_memo(actx->geo, psi, actx->zmaxis, Z, rclose, false, false, arc_memo);
    ival = integrate_phi_along_psi_contour_memo(actx->geo, psi, Z, actx->zmax, rclose, false, false, arc_memo);
    phi_ref = actx->phi_right;
    //printf("Z = %g, adding %g\n", actx->phi_right, ival = %g);
  }

  else if (actx->ftype==GKYL_CORE_R){ 
    if(Z<actx->zmaxis)
      ival = -integrate_phi_along_psi_contour_memo(actx->geo, psi, Z, actx->zmaxis, rclose, false, false, arc_memo);
    else
      ival = integrate_phi_along_psi_contour_memo(actx->geo, psi, actx->zmaxis, Z, rclose, false, false, arc_memo);
  }

  else if( (actx->ftype==GKYL_SOL_DN_OUT) || (actx->ftype==GKYL_SOL_DN_OUT_MID)){
    if (Z<actx->zmaxis)
      ival = -integrate_phi_along_psi_contour_memo(actx->geo, psi, Z, actx->zmaxis, rclose, false, false, arc_memo);
    else
      ival = integrate_phi_along_psi_contour_memo(actx->geo, psi, actx->zmaxis, Z, rclose, false, false, arc_memo);
  }
  else if(actx->ftype==GKYL_SOL_DN_OUT_LO){
    ival = integrate_phi_along_psi_contour_memo(actx->geo, psi, zmin, Z, rclose, false, false, arc_memo);
  }
  else if(actx->ftype==GKYL_SOL_DN_OUT_UP){
    ival = -integrate_phi_along_psi_contour_memo(actx->geo, psi, Z, zmax, rclose, false, false, arc_memo);
  }
  if( (actx->ftype==GKYL_SOL_DN_IN) || (actx->ftype==GKYL_SOL_DN_IN_MID) ){
    if (Z<actx->zmaxis)
      ival = integrate_phi_along_psi_contour_memo(actx->geo, psi, Z, actx->zmaxis, rclose, false, false, arc_memo);
    else
      ival = -integrate_phi_along_psi_contour_memo(actx->geo, psi, actx->zmaxis, Z, rclose, false, false, arc_memo);
  }
  else if(actx->ftype==GKYL_SOL_DN_IN_LO){
    ival = integrate_phi_along_psi_contour_memo(actx->geo, psi, zmin, Z, rclose, false, false, arc_memo);
  }
  else if(actx->ftype==GKYL_SOL_DN_IN_UP){
    ival = -integrate_phi_along_psi_contour_memo(actx->geo, psi, Z, zmax, rclose, false, false, arc_memo);
  }

  else if(actx->ftype==GKYL_SOL_SN_LO){ // alpha = phi at outboard midplane
    if (actx->right==true){
      if (Z<actx->zmaxis)
        ival = -integrate_phi_along_psi_contour_memo(actx->geo, psi, Z, actx->zmaxis, rclose, false, false, arc_memo);
      else
        ival = integrate_phi_along_psi_contour_memo(actx->geo, psi, actx->zmaxis, Z, rclose, false, false, arc_memo);
    }
    else{
      ival = integrate_phi_along_psi_contour_memo(actx->geo, psi, Z, actx->zmax, rclose, false, false, arc_memo);
      phi_ref = actx->phi_right;
    }
  }
  else if(actx->ftype==GKYL_PF_LO_R){
      ival = integrate_phi_along_psi_contour_memo(actx->geo, psi, zmin, Z, rclose, false, false, arc_memo);
  }
  else if(actx->ftype==GKYL_PF_LO_L){
      ival = -integrate_phi_along_psi_contour_memo(actx->geo, psi, zmin, Z, rclose, false, false, arc_memo);// + actx->phi_right;
  }
  else if(actx->ftype==GKYL_PF_UP_R){
      ival = -integrate_phi_along_psi_contour_memo(actx->geo, psi, Z, zmax, rclose, false, false, arc_memo);
  }
  else if(actx->ftype==GKYL_PF_UP_L){
      ival = integrate_phi_along_psi_contour_memo(actx->geo, psi, Z, zmax, rclose, false, false, arc_memo);// + actx->phi_right;
  }
  // Now multiply by fpol
  double R[4] = {0};
  double dR[4] = {0};
  int nr = R_psiZ(actx->geo, psi, Z, 4, R, dR);
  double r_curr = nr == 1 ? R[0] : choose_closest(rclose, R, R, nr);
  double psi_fpol = psi;
  //if (psi_fpol < actx->geo->psisep) // F = F(psi_sep) in the SOL. Convention of psi increases inward
  if ( (psi_fpol < actx->geo->fgrid.lower[0]) || (psi_fpol > actx->geo->fgrid.upper[0]) ) // F = F(psi_sep) in the SOL.
    psi_fpol = actx->geo->psisep;
  int idx = fmin(actx->geo->frange.lower[0] + (int) floor((psi_fpol - actx->geo->fgrid.lower[0])/actx->geo->fgrid.dx[0]), actx->geo->frange.upper[0]);
  long loc = gkyl_range_idx(&actx->geo->frange, &idx);
  const double *coeffs = gkyl_array_cfetch(actx->geo->fpoldg,loc);
  double fxc;
  gkyl_rect_grid_cell_center(&actx->geo->fgrid, &idx, &fxc);
  double fx = (psi_fpol-fxc)/(actx->geo->fgrid.dx[0]*0.5);
  double fpol = actx->geo->fbasis.eval_expand(&fx, coeffs);
  ival = ival*fpol;

  while(ival < -M_PI){
    ival +=2*M_PI;
  }
  while(ival > M_PI){
    ival -=2*M_PI;
  }
  return alpha_curr + ival + phi_ref;
}

double
dphidtheta_func(double Z, void *ctx)
{
  struct arc_length_ctx *actx = ctx;
  double *arc_memo = actx->arc_memo;
  double psi = actx->psi, rclose = actx->rclose, zmin = actx->zmin, arcL = actx->arcL, zmax = actx->zmax;

  // Get the integrand
  double integrand = 0.0;
  struct contour_ctx cctx = {
    .geo = actx->geo,
    .psi = psi,
    .ncall = 0,
    .last_R = rclose
  };
  integrand = dphidtheta_integrand(Z, &cctx);
  // Now multiply by fpol
  double R[4] = {0};
  double dR[4] = {0};
  int nr = R_psiZ(actx->geo, psi, Z, 4, R, dR);
  double r_curr = nr == 1 ? R[0] : choose_closest(rclose, R, R, nr);
  double psi_fpol = psi;
  if ( (psi_fpol < actx->geo->fgrid.lower[0]) || (psi_fpol > actx->geo->fgrid.upper[0]) ) // F = F(psi_sep) in the SOL.
    psi_fpol = actx->geo->psisep;
  int idx = fmin(actx->geo->frange.lower[0] + (int) floor((psi_fpol - actx->geo->fgrid.lower[0])/actx->geo->fgrid.dx[0]), actx->geo->frange.upper[0]);
  long loc = gkyl_range_idx(&actx->geo->frange, &idx);
  const double *coeffs = gkyl_array_cfetch(actx->geo->fpoldg,loc);
  double fxc;
  gkyl_rect_grid_cell_center(&actx->geo->fgrid, &idx, &fxc);
  double fx = (psi_fpol-fxc)/(actx->geo->fgrid.dx[0]*0.5);
  double fpol = actx->geo->fbasis.eval_expand(&fx, coeffs);
  integrand = integrand*fpol;
  integrand = integrand*actx->arcL_tot/2/M_PI;
  return integrand;
}




gkyl_tok_geo*
gkyl_tok_geo_new(const struct gkyl_tok_geo_efit_inp *inp)
{
  struct gkyl_tok_geo *geo = gkyl_malloc(sizeof(*geo));

  geo->efit = gkyl_efit_new(inp->filepath, inp->rzpoly_order, inp->rz_basis_type, inp->fluxpoly_order, inp->reflect, false);

  geo->plate_spec = inp->plate_spec;
  geo->plate_func_lower = inp->plate_func_lower;
  geo->plate_func_upper = inp->plate_func_upper;

  geo->rzbasis= *geo->efit->rzbasis;
  geo->rzgrid = *geo->efit->rzgrid;
  geo->psiRZ = gkyl_array_acquire(geo->efit->psizr);
  geo->psibyrRZ = gkyl_array_acquire(geo->efit->psibyrzr);
  geo->psibyr2RZ = gkyl_array_acquire(geo->efit->psibyr2zr);

  geo->num_rzbasis = geo->efit->rzbasis->num_basis;
  geo->rzlocal = *geo->efit->rzlocal;
  geo->rzlocal_ext = *geo->efit->rzlocal_ext;
  geo->fgrid = *geo->efit->fluxgrid;
  geo->fbasis = *geo->efit->fluxbasis;
  geo->frange = *geo->efit->fluxlocal;
  geo->frange_ext = *geo->efit->fluxlocal_ext;
  geo->fpoldg= gkyl_array_acquire(geo->efit->fpolflux);
  geo->qdg= gkyl_array_acquire(geo->efit->qflux);
  geo->psisep = geo->efit->sibry;
  geo->zmaxis = geo->efit->zmaxis;

  geo->root_param.eps =
    inp->root_param.eps > 0 ? inp->root_param.eps : 1e-10;
  geo->root_param.max_iter =
    inp->root_param.max_iter > 0 ? inp->root_param.max_iter : 100;

  geo->quad_param.max_level =
    inp->quad_param.max_levels > 0 ? inp->quad_param.max_levels : 10;
  geo->quad_param.eps =
    inp->quad_param.eps > 0 ? inp->quad_param.eps : 1e-10;

  if (geo->efit->rzbasis->poly_order == 1)
    geo->calc_roots = calc_RdR_p1;
  else if (geo->efit->rzbasis->poly_order == 2){
    if(inp->rz_basis_type == GKYL_BASIS_MODAL_SERENDIPITY)
      geo->calc_roots = calc_RdR_p2;
    else if(inp->rz_basis_type == GKYL_BASIS_MODAL_TENSOR)
      geo->calc_roots = calc_RdR_p2_tensor_nrc;
  }

  geo->stat = (struct gkyl_tok_geo_stat) { };

  
  return geo;
}

double
gkyl_tok_geo_integrate_psi_contour(const gkyl_tok_geo *geo, double psi,
  double zmin, double zmax, double rclose)
{
  return integrate_psi_contour_memo(geo, psi, zmin, zmax, rclose,
    false, false, 0);
}

int
gkyl_tok_geo_R_psiZ(const gkyl_tok_geo *geo, double psi, double Z, int nmaxroots,
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

  gkyl_grid_sub_array_write(&grid, nrange, 0, nodes, nm);
}


void gkyl_tok_geo_calc(struct gk_geometry* up, struct gkyl_range *nrange, double dzc[3], struct gkyl_tok_geo *geo, 
    struct gkyl_tok_geo_grid_inp *inp, struct gkyl_array *mc2p_nodal_fd, struct gkyl_array *mc2p_nodal, struct gkyl_array *mc2p,
    struct gkyl_array *mc2prz_nodal_fd, struct gkyl_array *mc2prz_nodal, struct gkyl_array *mc2prz, struct gkyl_array *dphidtheta_nodal)
{

  geo->rleft = inp->rleft;
  geo->rright = inp->rright;

  geo->exact_roots = inp->exact_roots;

  geo->rmax = inp->rmax;
  geo->rmin = inp->rmin;

  enum { PSI_IDX, AL_IDX, TH_IDX }; // arrangement of computational coordinates
  enum { X_IDX, Y_IDX, Z_IDX }; // arrangement of cartesian coordinates
  
  double dtheta = inp->cgrid.dx[TH_IDX],
    dpsi = inp->cgrid.dx[PSI_IDX],
    dalpha = inp->cgrid.dx[AL_IDX];
  
  double theta_lo = up->grid.lower[TH_IDX] + (up->local.lower[TH_IDX] - up->global.lower[TH_IDX])*up->grid.dx[TH_IDX],
    psi_lo = up->grid.lower[PSI_IDX] + (up->local.lower[PSI_IDX] - up->global.lower[PSI_IDX])*up->grid.dx[PSI_IDX],
    alpha_lo = up->grid.lower[AL_IDX] + (up->local.lower[AL_IDX] - up->global.lower[AL_IDX])*up->grid.dx[AL_IDX];

  double dx_fact = up->basis.poly_order == 1.0/up->basis.poly_order;
  dtheta *= dx_fact; dpsi *= dx_fact; dalpha *= dx_fact;

  // used for finite differences 
  double delta_alpha = dalpha*1e-4;
  double delta_psi = dpsi*1e-6;
  double delta_theta = dtheta*1e-4;
  dzc[0] = delta_psi;
  dzc[1] = delta_alpha;
  dzc[2] = delta_theta;
  int modifiers[5] = {0, -1, 1, -2, 2};

  double rclose = inp->rclose;
  double rright = inp->rright;
  double rleft = inp->rleft;


  int nzcells = geo->rzgrid.cells[1];
  double *arc_memo = gkyl_malloc(sizeof(double[nzcells]));
  double *arc_memo_left = gkyl_malloc(sizeof(double[nzcells]));
  double *arc_memo_right = gkyl_malloc(sizeof(double[nzcells]));

  struct arc_length_ctx arc_ctx = {
    .geo = geo,
    .arc_memo = arc_memo,
    .arc_memo_right = arc_memo_right,
    .arc_memo_left = arc_memo_left,
    .ftype = inp->ftype,
    .zmaxis = geo->zmaxis
  };
  struct plate_ctx pctx = {
    .geo = geo
  };

  int cidx[3] = { 0 };
  for(int ia=nrange->lower[AL_IDX]; ia<=nrange->upper[AL_IDX]; ++ia){
    cidx[AL_IDX] = ia;
    for(int ia_delta = 0; ia_delta < 5; ia_delta++){ // should be <5
      if((ia == nrange->lower[AL_IDX]) && (up->local.lower[AL_IDX]== up->global.lower[AL_IDX]) ){
        if(ia_delta == 1 || ia_delta == 3)
          continue; // want to use one sided stencils at edge
      }
      else if((ia == nrange->upper[AL_IDX])  && (up->local.upper[AL_IDX]== up->global.upper[AL_IDX])){
          if(ia_delta == 2 || ia_delta == 4)
            continue; // want to use one sided stencils at edge
      }
      else{ //interior
        if( ia_delta == 3 || ia_delta == 4)
          continue; //dont do two away
      }

      double alpha_curr = alpha_lo + ia*dalpha + modifiers[ia_delta]*delta_alpha;
      // This is the convention described in Noah Mandell's Thesis Eq 5.104. comp coord y = -alpha.
      alpha_curr*=-1.0;

      for (int ip=nrange->lower[PSI_IDX]; ip<=nrange->upper[PSI_IDX]; ++ip) {
        int ip_delta_max = 5;// should be 5
        if(ia_delta != 0)
          ip_delta_max = 1;
        for(int ip_delta = 0; ip_delta < ip_delta_max; ip_delta++){
          if((ip == nrange->lower[PSI_IDX]) && (up->local.lower[PSI_IDX]== up->global.lower[PSI_IDX]) ){
            if(ip_delta == 1 || ip_delta == 3)
              continue; // want to use one sided stencils at edge
          }
          else if((ip == nrange->upper[PSI_IDX]) && (up->local.upper[PSI_IDX]== up->global.upper[PSI_IDX])){
            if(ip_delta == 2 || ip_delta == 4)
              continue; // want to use one sided stencils at edge
          }
          else{ // interior 
            if( ip_delta == 3 || ip_delta == 4)
              continue; //dont do two away
          }

          double psi_curr = psi_lo + ip*dpsi + modifiers[ip_delta]*delta_psi;


          double darcL, arcL_curr, arcL_lo;

          // For double null blocks this should set arc_ctx :
          // zmin, zmax, rclose, arcL_tot for all blocks. No left and right
          // For a full core case:
          // also set phi_right and arcL_right
          // For a single null case:
          // also set zmin_left and zmin_right 
          tok_find_endpoints(inp, geo, &arc_ctx, &pctx, psi_curr, alpha_curr, arc_memo, arc_memo_left, arc_memo_right);
          if(ip==0 && ia==0 && ip_delta==0 && ia_delta==0){
            if(inp->ftype==GKYL_CORE_R) 
              printf("In right core block, bottom xpt at z = %1.16f, top at z = %1.16f\n", arc_ctx.zmin, arc_ctx.zmax);
            if(inp->ftype==GKYL_CORE_L) 
              printf("In left core block, bottom xpt at z = %1.16f, top at z = %1.16f\n", arc_ctx.zmin, arc_ctx.zmax);
          }

          darcL = arc_ctx.arcL_tot/(up->basis.poly_order*inp->cgrid.cells[TH_IDX]) * (inp->cgrid.upper[TH_IDX] - inp->cgrid.lower[TH_IDX])/2/M_PI;
          // at the beginning of each theta loop we need to reset things
          cidx[PSI_IDX] = ip;
          arcL_curr = 0.0;
          arcL_lo = (theta_lo + M_PI)/2/M_PI*arc_ctx.arcL_tot;
          double ridders_min, ridders_max;
          // set node coordinates
          for (int it=nrange->lower[TH_IDX]; it<=nrange->upper[TH_IDX]; ++it) {
            int it_delta_max = 5; // should be 5
            if(ia_delta != 0 || ip_delta != 0 )
              it_delta_max = 1;
            for(int it_delta = 0; it_delta < it_delta_max; it_delta++){
              if((it == nrange->lower[TH_IDX]) && (up->local.lower[TH_IDX]== up->global.lower[TH_IDX])){
                if(it_delta == 1 || it_delta == 3)
                  continue; // want to use one sided stencils at edge
              }
              else if((it == nrange->upper[TH_IDX]) && (up->local.upper[TH_IDX]== up->global.upper[TH_IDX])){
                if(it_delta == 2 || it_delta == 4)
                  continue; // want to use one sided stencils at edge
              }
              else{
                if( it_delta == 3 || it_delta == 4)
                  continue; //dont do two away
              }
              arcL_curr = arcL_lo + it*darcL + modifiers[it_delta]*delta_theta*(arc_ctx.arcL_tot/2/M_PI);
              double theta_curr = arcL_curr*(2*M_PI/arc_ctx.arcL_tot) - M_PI ; 

              tok_set_ridders(inp, &arc_ctx, psi_curr, arcL_curr, &rclose, &ridders_min, &ridders_max);

              struct gkyl_qr_res res = gkyl_ridders(arc_length_func, &arc_ctx,
                arc_ctx.zmin, arc_ctx.zmax, ridders_min, ridders_max,
                geo->root_param.max_iter, 1e-10);
              double z_curr = res.res;
              
              if( inp->ftype == GKYL_PF_UP_L ||inp->ftype == GKYL_PF_LO_L || inp->ftype == GKYL_CORE_L || inp->ftype == GKYL_SOL_DN_IN|| inp->ftype == GKYL_SOL_DN_IN_UP || inp->ftype == GKYL_SOL_DN_IN_MID || inp->ftype == GKYL_SOL_DN_IN_LO) {
                if(it == nrange->upper[TH_IDX] && (up->local.upper[TH_IDX]== up->global.upper[TH_IDX]) && it_delta == 0) z_curr = arc_ctx.zmin;
                if(it == nrange->lower[TH_IDX] && (up->local.lower[TH_IDX]== up->global.lower[TH_IDX]) && it_delta == 0) z_curr = arc_ctx.zmax;
              }
              else {
                if(it == nrange->upper[TH_IDX] && (up->local.upper[TH_IDX]== up->global.upper[TH_IDX]) && it_delta == 0) z_curr = arc_ctx.zmax;
                if(it == nrange->lower[TH_IDX] && (up->local.lower[TH_IDX]== up->global.lower[TH_IDX]) && it_delta == 0) z_curr = arc_ctx.zmin;
              }


              ((gkyl_tok_geo *)geo)->stat.nroot_cont_calls += res.nevals;
              double R[4] = { 0 }, dR[4] = { 0 };
              int nr = R_psiZ(geo, psi_curr, z_curr, 4, R, dR);
              double r_curr = choose_closest(rclose, R, R, nr);

              // For all blocks on the inner edge with z boundaries we will need to match the entire outer edge
              if(inp->ftype == GKYL_CORE_L){ // Match the core right boundary at upper and lower theta ends
                if ((it == nrange->lower[TH_IDX]) && (up->local.lower[TH_IDX]== up->global.lower[TH_IDX]))
                  r_curr = choose_closest(inp->rright, R, R, nr);
                if((it == nrange->upper[TH_IDX]) && (up->local.upper[TH_IDX]== up->global.upper[TH_IDX]))
                  r_curr = choose_closest(inp->rright, R, R, nr);
              }

              if(inp->ftype == GKYL_PF_LO_L){ // Match the right pf lo boundary at lower theta end
                if ((it == nrange->lower[TH_IDX]) && (up->local.lower[TH_IDX]== up->global.lower[TH_IDX])) {
                  r_curr = choose_closest(inp->rright, R, R, nr);
                }
              }

              if(inp->ftype == GKYL_PF_UP_L){ // Match the right pf lo boundary at lower theta end
                if ((it == nrange->upper[TH_IDX]) && (up->local.upper[TH_IDX]== up->global.upper[TH_IDX])) {
                  r_curr = choose_closest(inp->rright, R, R, nr);
                }
              }

              // For all blocks on the inner edge with x boundaries we will need to match the X-point
              if(inp->ftype == GKYL_SOL_DN_IN_UP){ // Match the right side
                if((ip == nrange->upper[PSI_IDX]) && (up->local.upper[PSI_IDX]== up->global.upper[PSI_IDX])) {
                  if ((it == nrange->upper[TH_IDX]) && (up->local.upper[TH_IDX]== up->global.upper[TH_IDX])) {
                    r_curr = choose_closest(inp->rright, R, R, nr);
                  }
                }
              }

              if(inp->ftype == GKYL_SOL_DN_IN_MID){ // Match the right side
                if((ip == nrange->upper[PSI_IDX]) && (up->local.upper[PSI_IDX]== up->global.upper[PSI_IDX])) {
                  if ((it == nrange->lower[TH_IDX]) && (up->local.lower[TH_IDX]== up->global.lower[TH_IDX])) {
                    r_curr = choose_closest(inp->rright, R, R, nr);
                  }
                  if ((it == nrange->upper[TH_IDX]) && (up->local.upper[TH_IDX]== up->global.upper[TH_IDX])) {
                    r_curr = choose_closest(inp->rright, R, R, nr);
                  }
                }
              }

              if(inp->ftype == GKYL_SOL_DN_IN_LO){ // Match the right side
                if((ip == nrange->upper[PSI_IDX]) && (up->local.upper[PSI_IDX]== up->global.upper[PSI_IDX])) {
                  if ((it == nrange->lower[TH_IDX]) && (up->local.lower[TH_IDX]== up->global.lower[TH_IDX])) {
                    r_curr = choose_closest(inp->rright, R, R, nr);
                  }
                }
              }

              if(nr==0){
                printf(" ip = %d, it = %d, ia = %d, ip_delta = %d, it_delta = %d, ia_delta = %d\n", ip, it, ia, ip_delta, it_delta, ia_delta);
                printf("Failed to find a root at psi = %g, Z = %1.16f\n", psi_curr, z_curr);
                assert(false);
              }

              cidx[TH_IDX] = it;
              int lidx = 0;
              if (ip_delta != 0)
                lidx = 3 + 3*(ip_delta-1);
              if (ia_delta != 0)
                lidx = 15 + 3*(ia_delta-1);
              if (it_delta != 0)
                lidx = 27 + 3*(it_delta-1);

              if(ia_delta==0 && ip_delta==0 && it_delta==0)
                lidx = 0;

              double phi_curr = phi_func(alpha_curr, z_curr, &arc_ctx);
              double *mc2p_fd_n = gkyl_array_fetch(mc2p_nodal_fd, gkyl_range_idx(nrange, cidx));
              double *mc2p_n = gkyl_array_fetch(mc2p_nodal, gkyl_range_idx(nrange, cidx));
              double *mc2prz_fd_n = gkyl_array_fetch(mc2prz_nodal_fd, gkyl_range_idx(nrange, cidx));
              double *mc2prz_n = gkyl_array_fetch(mc2prz_nodal, gkyl_range_idx(nrange, cidx));
              double *dphidtheta_n = gkyl_array_fetch(dphidtheta_nodal, gkyl_range_idx(nrange, cidx));

              mc2p_fd_n[lidx+X_IDX] = r_curr*cos(phi_curr);
              mc2p_fd_n[lidx+Y_IDX] = r_curr*sin(phi_curr);
              mc2p_fd_n[lidx+Z_IDX] = z_curr;
              mc2prz_fd_n[lidx+X_IDX] = r_curr;
              mc2prz_fd_n[lidx+Y_IDX] = z_curr;
              mc2prz_fd_n[lidx+Z_IDX] = phi_curr;

              if(ip_delta==0 && ia_delta==0 && it_delta==0){
                mc2p_n[X_IDX] = r_curr*cos(phi_curr);
                mc2p_n[Y_IDX] = r_curr*sin(phi_curr);
                mc2p_n[Z_IDX] = z_curr;
                mc2prz_n[X_IDX] = r_curr;
                mc2prz_n[Y_IDX] = z_curr;
                mc2prz_n[Z_IDX] = phi_curr;
                dphidtheta_n[0] = dphidtheta_func(z_curr, &arc_ctx);
              }
            }
          }
        }
      }
    }
  }
  struct gkyl_nodal_ops *n2m =  gkyl_nodal_ops_new(&inp->cbasis, &inp->cgrid, false);
  gkyl_nodal_ops_n2m(n2m, &inp->cbasis, &inp->cgrid, nrange, &up->local, 3, mc2p_nodal, mc2p);
  gkyl_nodal_ops_n2m(n2m, &inp->cbasis, &inp->cgrid, nrange, &up->local, 3, mc2prz_nodal, mc2prz);
  gkyl_nodal_ops_release(n2m);

  char str1[50] = "xyz";
  char str2[50] = "allxyz";
  if (inp->write_node_coord_array){
    write_nodal_coordinates(strcat(str1, inp->node_file_nm), nrange, mc2p_nodal);
    write_nodal_coordinates(strcat(str2, inp->node_file_nm), nrange, mc2p_nodal_fd);
  }
  gkyl_free(arc_memo);
  gkyl_free(arc_memo_left);
  gkyl_free(arc_memo_right);
}


struct gkyl_tok_geo_stat
gkyl_tok_geo_get_stat(const gkyl_tok_geo *geo)
{
  return geo->stat;
}

void
gkyl_tok_geo_release(gkyl_tok_geo *geo)
{
  gkyl_array_release(geo->psiRZ);
  gkyl_array_release(geo->psibyrRZ);
  gkyl_array_release(geo->psibyr2RZ);
  gkyl_array_release(geo->fpoldg);
  gkyl_array_release(geo->qdg);
  gkyl_efit_release(geo->efit);
  gkyl_free(geo);
}
