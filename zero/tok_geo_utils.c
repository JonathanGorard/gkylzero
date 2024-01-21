#include <gkyl_tok_geo_priv.h>

// Helper functions for finding turning points when necessary


// This function will set zmax to be the upper turning point location
void find_upper_turning_point(struct gkyl_tok_geo *geo, double psi_curr, double zlo, double *zmax)
{
    //Find the turning points
    double zlo_last;
    double zup=*zmax;
    zlo_last = zlo;
    double R[4], dR[4];
    while(true){
      int nlo = R_psiZ(geo, psi_curr, zlo, 4, R, dR);
      if(nlo==2){
        if(fabs(zlo-zup)<1e-12){
          *zmax = zlo;
          break;
        }
        zlo_last = zlo;
        zlo = (zlo+zup)/2;
      }
      if(nlo==0){
        zup = zlo;
        zlo = zlo_last;
      }
    }
}

// This function will set zmin to be the upper lower point location
void find_lower_turning_point(struct gkyl_tok_geo *geo, double psi_curr, double zup, double *zmin)
{
    int nup = 0;
    double zlo=*zmin;
    double zup_last = zup;
    double R[4], dR[4];
    while(true){
      int nup = R_psiZ(geo, psi_curr, zup, 4, R, dR);
      if(nup==2){
        if(fabs(zlo-zup)<1e-12){
          *zmin = zup;
          break;
        }
        zup_last = zup;
        zup = (zlo+zup)/2;
      }
      if(nup==0){
        zlo = zup;
        zup = zup_last;
      }
    }

}

// Sets zmax if plate is specified
void set_upper_plate(struct gkyl_tok_geo *geo, struct arc_length_ctx* arc_ctx, struct plate_ctx* pctx, double psi_curr)
{
      double rzplate[2];
      pctx->psi_curr = psi_curr;
      pctx->lower=false;
      double a = 0;
      double b = 1;
      double fa = tok_plate_psi_func(a, pctx);
      double fb = tok_plate_psi_func(b, pctx);
      struct gkyl_qr_res res = gkyl_ridders(tok_plate_psi_func, pctx,
        a, b, fa, fb, geo->root_param.max_iter, 1e-10);
      double smax = res.res;
      geo->plate_func_upper(smax, rzplate);
      arc_ctx->zmax = rzplate[1];
}

// Sets zmin if plate is specified
void set_lower_plate(struct gkyl_tok_geo *geo, struct arc_length_ctx* arc_ctx, struct plate_ctx* pctx, double psi_curr)
{
      double rzplate[2];
      pctx->psi_curr = psi_curr;
      pctx->lower=true;
      double a = 0;
      double b = 1;
      double fa = tok_plate_psi_func(a, pctx);
      double fb = tok_plate_psi_func(b, pctx);
      struct gkyl_qr_res res = gkyl_ridders(tok_plate_psi_func, pctx,
        a, b, fa, fb, geo->root_param.max_iter, 1e-10);
      double smin = res.res;
      geo->plate_func_lower(smin, rzplate);
      arc_ctx->zmin = rzplate[1];
}

void
tok_find_endpoints(struct gkyl_tok_geo_grid_inp* inp, struct gkyl_tok_geo *geo, struct arc_length_ctx* arc_ctx, struct plate_ctx* pctx, double psi_curr, double alpha_curr, double* arc_memo, double* arc_memo_left, double* arc_memo_right){
  enum { PH_IDX, AL_IDX, TH_IDX }; // arrangement of computational coordinates
  enum { X_IDX, Y_IDX, Z_IDX }; // arrangement of cartesian coordinates


  // Set psicurr no matter what
  arc_ctx->psi = psi_curr;

  if(inp->ftype == GKYL_CORE){
    // Immediately set rleft and rright. Will need both
    arc_ctx->rright = inp->rright;
    arc_ctx->rleft = inp->rleft;

    arc_ctx->zmax = inp->zxpt_up + 1e-1; // Initial guess. Give  the bisection some room
    double zlo = geo->zmaxis;
    find_upper_turning_point(geo, psi_curr, zlo, &arc_ctx->zmax);
    arc_ctx->zmin = inp->zxpt_lo - 1e-1; // Initial guess
    double zup = geo->zmaxis;
    find_lower_turning_point(geo, psi_curr, zup, &arc_ctx->zmin);
    // Done finding turning points
    arc_ctx->arcL_right = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, arc_ctx->zmax, arc_ctx->rright,
      true, true, arc_memo_right);
    arc_ctx->right = false;
    double arcL_l = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, arc_ctx->zmax, arc_ctx->rleft,
      true, true, arc_memo_left);
    arc_ctx->arcL_tot = arcL_l + arc_ctx->arcL_right;

    arc_ctx->right = true;
    arc_ctx->phi_right = 0.0;
    arc_ctx->rclose = arc_ctx->rright;
    arc_ctx->phi_right = phi_func(alpha_curr, arc_ctx->zmax, arc_ctx) - alpha_curr;
  }

  if(inp->ftype == GKYL_CORE_L){
    // Immediately set rclose
    arc_ctx->rclose = inp->rleft;
    // Find turning points to set zmin and zmax
    arc_ctx->zmax = inp->zxpt_up + 1e-1; // Initial guess
    double zlo = geo->zmaxis;
    find_upper_turning_point(geo, psi_curr, zlo, &arc_ctx->zmax);
    arc_ctx->zmin = inp->zxpt_lo - 1e-1; // Initial guess
    double zup = geo->zmaxis;
    find_lower_turning_point(geo, psi_curr, zup, &arc_ctx->zmin);
    // Set arc length
    arc_ctx->arcL_tot = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, arc_ctx->zmax, arc_ctx->rclose,
      true, true, arc_memo);
  }

  if(inp->ftype == GKYL_CORE_R){
    // Immediately set rclose
    arc_ctx->rclose = inp->rright;
    // Find turning points to set zmin and zmax
    arc_ctx->zmax = inp->zxpt_up + 1e-1; // Initial guess
    double zlo = geo->zmaxis;
    find_upper_turning_point(geo, psi_curr, zlo, &arc_ctx->zmax);
    arc_ctx->zmin = inp->zxpt_lo - 1e-1; // Initial guess
    double zup = geo->zmaxis;
    find_lower_turning_point(geo, psi_curr, zup, &arc_ctx->zmin);
    // Set arc length
    arc_ctx->arcL_tot = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, arc_ctx->zmax, arc_ctx->rright,
      true, true, arc_memo);
  }

  else if(inp->ftype == GKYL_PF_LO_L){
    // Immediately set rclose
    arc_ctx->rclose = inp->rleft;
    // Immediately set zmin
    arc_ctx->zmin = arc_ctx->zmin;
    //Find the  upper turning point to set zmax
    arc_ctx->zmax = inp->zxpt_lo; // Initial guess
    double zlo = arc_ctx->zmin;
    find_upper_turning_point(geo, psi_curr, zlo, &arc_ctx->zmax);
    // Set arc length
    arc_ctx->arcL_tot = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, arc_ctx->zmax, arc_ctx->rclose,
      true, true, arc_memo);
  }

  else if(inp->ftype == GKYL_PF_LO_R){
    // Immediately set rclose
    arc_ctx->rclose = inp->rright;
    // Immediately set zmin
    arc_ctx->zmin = inp->zmin;
    //Find the  upper turning point to set zmax
    arc_ctx->zmax = inp->zxpt_lo; // Initial guess
    double zlo = arc_ctx->zmin;
    find_upper_turning_point(geo, psi_curr, zlo, &arc_ctx->zmax);
    // Set arc length
    arc_ctx->arcL_tot = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, arc_ctx->zmax, arc_ctx->rclose,
      true, true, arc_memo);
  }

  else if(inp->ftype == GKYL_PF_UP_L){
    // Immediately set rclose
    arc_ctx->rclose = inp->rleft;
    // Immediately set zmax
    arc_ctx->zmax = inp->zmax;
    //Find the lower turning point to set zmin
    arc_ctx->zmin = inp->zxpt_up; // Initial guess
    double zup = arc_ctx->zmax;
    find_lower_turning_point(geo, psi_curr, zup, &arc_ctx->zmin);
    // Done finding turning point
    arc_ctx->arcL_tot = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, inp->zmax_left, arc_ctx->rclose,
      true, true, arc_memo);
  }

  else if(inp->ftype == GKYL_PF_UP_R){
    // Immediately set rclose
    arc_ctx->rclose = inp->rright;
    // Immediately set zmax
    arc_ctx->zmax = inp->zmax;
    //Find the lower turning point to set zmin
    arc_ctx->zmin = inp->zxpt_up; // Initial guess
    double zup = arc_ctx->zmax;
    find_lower_turning_point(geo, psi_curr, zup, &arc_ctx->zmin);
    // Done finding turning point
    arc_ctx->arcL_tot = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, inp->zmax_right, arc_ctx->rclose,
      true, true, arc_memo);
  }

  else if(inp->ftype==GKYL_SOL_DN_OUT){
    // Immediately set rclose
    arc_ctx->rclose = inp->rright;
    // Set zmin and zmax either fixed or with plate
    if (geo->plate_spec){
      set_upper_plate(geo, arc_ctx, pctx, arc_ctx->psi);
      set_lower_plate(geo, arc_ctx, pctx, arc_ctx->psi);
    }
    else{
      arc_ctx->zmin = inp->zmin;
      arc_ctx->zmax = inp->zmax;
    }
    // Set the arc length
    arc_ctx->arcL_tot = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, arc_ctx->zmax, arc_ctx->rclose, true, true, arc_memo);
  }

  else if(inp->ftype==GKYL_SOL_DN_OUT_LO){
    // Immediately set rclose
    arc_ctx->rclose = inp->rright;
    // Set zmax to be the lower x-point
    arc_ctx->zmax = inp->zxpt_lo;
    // Set zmin either fixed or with plate
    if (geo->plate_spec){
      set_lower_plate(geo, arc_ctx, pctx, arc_ctx->psi);
    }
    else{
      arc_ctx->zmin = inp->zmin;
    }
    // Set the arc length
    arc_ctx->arcL_tot = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, arc_ctx->zmax, arc_ctx->rclose, true, true, arc_memo);
  }

  else if(inp->ftype==GKYL_SOL_DN_OUT_MID){
    // Immediately set rclose
    arc_ctx->rclose = inp->rright;
    // Set zmin and zmax to be the x-points
    arc_ctx->zmax = inp->zxpt_up;
    arc_ctx->zmin = inp->zxpt_lo;
    // Set the arc length
    arc_ctx->arcL_tot = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, arc_ctx->zmax, arc_ctx->rclose, true, true, arc_memo);
  }

  else if(inp->ftype==GKYL_SOL_DN_OUT_UP){
    // Immediately set rclose
    arc_ctx->rclose = inp->rright;
    // Set zmin to be the upper x-point
    arc_ctx->zmin = inp->zxpt_up;
    // Set zmax either fixed or with plate
    if (geo->plate_spec){
      set_upper_plate(geo, arc_ctx, pctx, arc_ctx->psi);
    }
    else{
      arc_ctx->zmax = inp->zmax;
    }
    // Set the arc length
    arc_ctx->arcL_tot = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, arc_ctx->zmax, arc_ctx->rclose, true, true, arc_memo);
  }

  else if(inp->ftype==GKYL_SOL_DN_IN){
    // Immediately set rclose
    arc_ctx->rclose = inp->rleft;
    // Set zmin and zmax either fixed or with plate
    if (geo->plate_spec){
      set_upper_plate(geo, arc_ctx, pctx, arc_ctx->psi);
      set_lower_plate(geo, arc_ctx, pctx, arc_ctx->psi);
    }
    else{
      arc_ctx->zmin = inp->zmin;
      arc_ctx->zmax = inp->zmax;
    }
    // Set the arc Length
    arc_ctx->arcL_tot = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, arc_ctx->zmax, arc_ctx->rclose, true, true, arc_memo);
  }

  else if(inp->ftype==GKYL_SOL_DN_IN_LO){
    // Immediately set rclose
    arc_ctx->rclose = inp->rleft;
    // Set zmax to be the lower x-point
    arc_ctx->zmax = inp->zxpt_lo;
    // Set zmin either fixed or with plate
    if (geo->plate_spec){
      set_lower_plate(geo, arc_ctx, pctx, arc_ctx->psi);
    }
    else{
      arc_ctx->zmin = inp->zmin;
    }
    // Set the arc Length
    arc_ctx->arcL_tot = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, arc_ctx->zmax, arc_ctx->rclose, true, true, arc_memo);
  }

  else if(inp->ftype==GKYL_SOL_DN_IN_MID){
    // Immediately set rclose
    arc_ctx->rclose = inp->rleft;
    // Set zmin and zmax to be the x-points
    arc_ctx->zmax = inp->zxpt_up;
    arc_ctx->zmin = inp->zxpt_lo;
    // Set the arc Length
    arc_ctx->arcL_tot = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, arc_ctx->zmax, arc_ctx->rclose, true, true, arc_memo);
  }

  else if(inp->ftype==GKYL_SOL_DN_IN_UP){
    // Immediately set rclose
    arc_ctx->rclose = inp->rleft;
    // Set zmin to be the upper x-point
    arc_ctx->zmin = inp->zxpt_up;
    // Set zmax either fixed or with plate
    if (geo->plate_spec){
      set_upper_plate(geo, arc_ctx, pctx, arc_ctx->psi);
    }
    else{
      arc_ctx->zmax = inp->zmax;
    }
    // Set the arc Length
    arc_ctx->arcL_tot = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin, arc_ctx->zmax, arc_ctx->rclose, true, true, arc_memo);
  }

  else if(inp->ftype == GKYL_SOL_SN_LO){
    // Immediately set rleft and rright. Will need both
    arc_ctx->rright = inp->rright;
    arc_ctx->rleft = inp->rleft;
    //Find the  upper turning point
    double zlo = fmax(inp->zmin_left, inp->zmin_right);
    find_upper_turning_point(geo, psi_curr, zlo, &arc_ctx->zmax);

    // Set zmin left and zmin right wither with plate or fixed
    // This one can't be used with the general func for setting upper and lower plates because it uses zmin left and zmin right
    if (geo->plate_spec){
      double rzplate[2];
      pctx->psi_curr = psi_curr;
      pctx->lower=false;
      double a = 0;
      double b = 1;
      double fa = tok_plate_psi_func(a, pctx);
      double fb = tok_plate_psi_func(b, pctx);
      struct gkyl_qr_res res = gkyl_ridders(tok_plate_psi_func, pctx,
        a, b, fa, fb, geo->root_param.max_iter, 1e-10);
      double smax = res.res;
      geo->plate_func_upper(smax, rzplate);
      arc_ctx->zmin_left = rzplate[1];

      pctx->lower=true;
      a = 0;
      b = 1;
      fa = tok_plate_psi_func(a, pctx);
      fb = tok_plate_psi_func(b, pctx);
      res = gkyl_ridders(tok_plate_psi_func, pctx,
        a, b, fa, fb, geo->root_param.max_iter, 1e-10);
      double smin = res.res;
      geo->plate_func_lower(smin, rzplate);
      arc_ctx->zmin_right = rzplate[1];
    }
    else{
      arc_ctx->zmin_left = inp->zmin_left;
      arc_ctx->zmin_right = inp->zmin_right;
    }

    // Done finding turning point
    arc_ctx->arcL_right = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin_right, arc_ctx->zmax, arc_ctx->rright,
      true, true, arc_memo_right);
    arc_ctx->right = false;
    double arcL_l = integrate_psi_contour_memo(geo, psi_curr, arc_ctx->zmin_left, arc_ctx->zmax, arc_ctx->rleft,
      true, true, arc_memo_left);
    arc_ctx->arcL_tot = arcL_l + arc_ctx->arcL_right;

    arc_ctx->right = true;
    arc_ctx->phi_right = 0.0;
    arc_ctx->rclose = arc_ctx->rright;
    arc_ctx->psi = psi_curr;
    arc_ctx->zmin = arc_ctx->zmin_right;
    arc_ctx->phi_right = phi_func(alpha_curr, arc_ctx->zmax, arc_ctx) - alpha_curr;
  }


}



void
tok_set_ridders(struct gkyl_tok_geo_grid_inp* inp, struct arc_length_ctx* arc_ctx, double psi_curr, double arcL_curr,double* rclose, double *ridders_min, double* ridders_max){


  if(inp->ftype==GKYL_CORE){
    if(arcL_curr <= arc_ctx->arcL_right){
      *rclose = arc_ctx->rright;
      arc_ctx->right = true;
      *ridders_min = -arcL_curr;
      *ridders_max = arc_ctx->arcL_tot-arcL_curr;
    }
    else{
      *rclose = arc_ctx->rleft;
      arc_ctx->right = false;
      *ridders_min = arc_ctx->arcL_tot - arcL_curr;
      *ridders_max = -arcL_curr + arc_ctx->arcL_right;
    }
  }
  else if(inp->ftype==GKYL_CORE_L){
      *rclose = arc_ctx->rleft;
      *ridders_min = arc_ctx->arcL_tot - arcL_curr;
      *ridders_max = -arcL_curr;
  }
  else if(inp->ftype==GKYL_CORE_R){
      *rclose = arc_ctx->rright;
      *ridders_min = -arcL_curr;
      *ridders_max = arc_ctx->arcL_tot-arcL_curr;
  }
//  if(inp->ftype==GKYL_PF_LO){
//    if(arcL_curr <= arc_ctx->arcL_right){
//      *rclose = arc_ctx->rright;
//      arc_ctx->right = true;
//      *ridders_min = -arcL_curr;
//      *ridders_max = arc_ctx->arcL_tot-arcL_curr;
//    }
//    else{
//      *rclose = arc_ctx->rleft;
//      arc_ctx->right = false;
//      *ridders_min = arc_ctx->arcL_tot - arcL_curr;
//      *ridders_max = -arcL_curr + arc_ctx->arcL_right;
//    }
//  }
//  if(inp->ftype==GKYL_PF_UP){
//    if(arcL_curr > arc_ctx->arcL_left){
//      *rclose = arc_ctx->rright;
//      arc_ctx->right = true;
//      *ridders_min = arc_ctx->arcL_left - arcL_curr;
//      *ridders_max = arc_ctx->arcL_tot - arcL_curr;
//    }
//    else{
//      *rclose = arc_ctx->rleft;
//      arc_ctx->right = false;
//      *ridders_min = arc_ctx->arcL_left - arcL_curr;
//      *ridders_max = -arcL_curr;
//    }
//  }
  else if( (arc_ctx->ftype==GKYL_SOL_DN_OUT) || (arc_ctx->ftype==GKYL_SOL_DN_OUT) || (arc_ctx->ftype==GKYL_SOL_DN_OUT_LO) || (arc_ctx->ftype==GKYL_SOL_DN_OUT_MID) || (arc_ctx->ftype==GKYL_SOL_DN_OUT_UP) ){
    *ridders_min = -arcL_curr;
    *ridders_max = arc_ctx->arcL_tot-arcL_curr;
    *rclose = arc_ctx->rclose;
  }
  else if( (arc_ctx->ftype==GKYL_SOL_DN_IN) || (arc_ctx->ftype==GKYL_SOL_DN_IN) || (arc_ctx->ftype==GKYL_SOL_DN_IN_LO) || (arc_ctx->ftype==GKYL_SOL_DN_IN_MID) || (arc_ctx->ftype==GKYL_SOL_DN_IN_UP) ){
    *ridders_min = arc_ctx->arcL_tot-arcL_curr;
    *ridders_max = -arcL_curr;
    *rclose = arc_ctx->rclose;
  }
  else if(arc_ctx->ftype==GKYL_SOL_SN_LO){
    if(arcL_curr <= arc_ctx->arcL_right){
      *rclose = arc_ctx->rright;
      arc_ctx->right = true;
      *ridders_min = -arcL_curr;
      *ridders_max = arc_ctx->arcL_tot-arcL_curr;
    }
    else{
      *rclose = arc_ctx->rleft;
      arc_ctx->right = false;
      *ridders_min = arc_ctx->arcL_tot - arcL_curr;
      *ridders_max = -arcL_curr + arc_ctx->arcL_right;
    }
  }

  arc_ctx->arcL = arcL_curr;
  arc_ctx->rclose = *rclose; // This would be unnecessary for all double null block cases. Only needed for SN and full core
}
