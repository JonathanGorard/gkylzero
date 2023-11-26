#pragma once 
#include <math.h>
#include <gkyl_mat.h>
#include <gkyl_math.h> 
#include <gkyl_wave_geom.h> 
#include <gkyl_wv_eqn.h> 
#include <gkyl_util.h> 
EXTERN_C_BEG 

GKYL_CU_DH void fluid_vars_pressure_1x_ser_p1(double param, const double *fluid, const double *u, double* GKYL_RESTRICT p, double* GKYL_RESTRICT p_surf); 
GKYL_CU_DH int fluid_vars_u_set_1x_ser_p1(int count, struct gkyl_nmat *A, struct gkyl_nmat *rhs, const double *fluid); 
GKYL_CU_DH void fluid_vars_u_copy_1x_ser_p1(int count, struct gkyl_nmat *x, double* GKYL_RESTRICT u, double* GKYL_RESTRICT u_surf); 
GKYL_CU_DH double euler_vol_1x_ser_p1(const double *w, const double *dxv, double gas_gamma, const double *u, const double *p, const double *fluid, double* GKYL_RESTRICT out); 
GKYL_CU_DH void fluid_vars_limiterx_1x_ser_p1(double limiter_fac, const struct gkyl_wv_eqn *wv_eqn, double *fluid_l, double *fluid_c, double *fluid_r); 
GKYL_CU_DH double euler_surfx_1x_ser_p1(const double *w, const double *dxv, const struct gkyl_wv_eqn *wv_eqn, 
    const struct gkyl_wave_cell_geom *geom_l, const struct gkyl_wave_cell_geom *geom_r, 
    const double *u_surf_l, const double *u_surf_c, const double *u_surf_r, 
    const double *p_surf_l, const double *p_surf_c, const double *p_surf_r, 
    const double *fluid_l, const double *fluid_c, const double *fluid_r, double* GKYL_RESTRICT out); 

GKYL_CU_DH void fluid_vars_pressure_1x_ser_p2(double param, const double *fluid, const double *u, double* GKYL_RESTRICT p, double* GKYL_RESTRICT p_surf); 
GKYL_CU_DH int fluid_vars_u_set_1x_ser_p2(int count, struct gkyl_nmat *A, struct gkyl_nmat *rhs, const double *fluid); 
GKYL_CU_DH void fluid_vars_u_copy_1x_ser_p2(int count, struct gkyl_nmat *x, double* GKYL_RESTRICT u, double* GKYL_RESTRICT u_surf); 
GKYL_CU_DH double euler_vol_1x_ser_p2(const double *w, const double *dxv, double gas_gamma, const double *u, const double *p, const double *fluid, double* GKYL_RESTRICT out); 
GKYL_CU_DH void fluid_vars_limiterx_1x_ser_p2(double limiter_fac, const struct gkyl_wv_eqn *wv_eqn, double *fluid_l, double *fluid_c, double *fluid_r); 
GKYL_CU_DH double euler_surfx_1x_ser_p2(const double *w, const double *dxv, const struct gkyl_wv_eqn *wv_eqn, 
    const struct gkyl_wave_cell_geom *geom_l, const struct gkyl_wave_cell_geom *geom_r, 
    const double *u_surf_l, const double *u_surf_c, const double *u_surf_r, 
    const double *p_surf_l, const double *p_surf_c, const double *p_surf_r, 
    const double *fluid_l, const double *fluid_c, const double *fluid_r, double* GKYL_RESTRICT out); 

GKYL_CU_DH void fluid_vars_pressure_2x_ser_p1(double param, const double *fluid, const double *u, double* GKYL_RESTRICT p, double* GKYL_RESTRICT p_surf); 
GKYL_CU_DH int fluid_vars_u_set_2x_ser_p1(int count, struct gkyl_nmat *A, struct gkyl_nmat *rhs, const double *fluid); 
GKYL_CU_DH void fluid_vars_u_copy_2x_ser_p1(int count, struct gkyl_nmat *x, double* GKYL_RESTRICT u, double* GKYL_RESTRICT u_surf); 
GKYL_CU_DH double euler_vol_2x_ser_p1(const double *w, const double *dxv, double gas_gamma, const double *u, const double *p, const double *fluid, double* GKYL_RESTRICT out); 
GKYL_CU_DH void fluid_vars_limiterx_2x_ser_p1(double limiter_fac, const struct gkyl_wv_eqn *wv_eqn, double *fluid_l, double *fluid_c, double *fluid_r); 
GKYL_CU_DH double euler_surfx_2x_ser_p1(const double *w, const double *dxv, const struct gkyl_wv_eqn *wv_eqn, 
    const struct gkyl_wave_cell_geom *geom_l, const struct gkyl_wave_cell_geom *geom_r, 
    const double *u_surf_l, const double *u_surf_c, const double *u_surf_r, 
    const double *p_surf_l, const double *p_surf_c, const double *p_surf_r, 
    const double *fluid_l, const double *fluid_c, const double *fluid_r, double* GKYL_RESTRICT out); 
GKYL_CU_DH void fluid_vars_limitery_2x_ser_p1(double limiter_fac, const struct gkyl_wv_eqn *wv_eqn, double *fluid_l, double *fluid_c, double *fluid_r); 
GKYL_CU_DH double euler_surfy_2x_ser_p1(const double *w, const double *dxv, const struct gkyl_wv_eqn *wv_eqn, 
    const struct gkyl_wave_cell_geom *geom_l, const struct gkyl_wave_cell_geom *geom_r, 
    const double *u_surf_l, const double *u_surf_c, const double *u_surf_r, 
    const double *p_surf_l, const double *p_surf_c, const double *p_surf_r, 
    const double *fluid_l, const double *fluid_c, const double *fluid_r, double* GKYL_RESTRICT out); 

GKYL_CU_DH void fluid_vars_pressure_3x_ser_p1(double param, const double *fluid, const double *u, double* GKYL_RESTRICT p, double* GKYL_RESTRICT p_surf); 
GKYL_CU_DH int fluid_vars_u_set_3x_ser_p1(int count, struct gkyl_nmat *A, struct gkyl_nmat *rhs, const double *fluid); 
GKYL_CU_DH void fluid_vars_u_copy_3x_ser_p1(int count, struct gkyl_nmat *x, double* GKYL_RESTRICT u, double* GKYL_RESTRICT u_surf); 
GKYL_CU_DH double euler_vol_3x_ser_p1(const double *w, const double *dxv, double gas_gamma, const double *u, const double *p, const double *fluid, double* GKYL_RESTRICT out); 
GKYL_CU_DH void fluid_vars_limiterx_3x_ser_p1(double limiter_fac, const struct gkyl_wv_eqn *wv_eqn, double *fluid_l, double *fluid_c, double *fluid_r); 
GKYL_CU_DH double euler_surfx_3x_ser_p1(const double *w, const double *dxv, const struct gkyl_wv_eqn *wv_eqn, 
    const struct gkyl_wave_cell_geom *geom_l, const struct gkyl_wave_cell_geom *geom_r, 
    const double *u_surf_l, const double *u_surf_c, const double *u_surf_r, 
    const double *p_surf_l, const double *p_surf_c, const double *p_surf_r, 
    const double *fluid_l, const double *fluid_c, const double *fluid_r, double* GKYL_RESTRICT out); 
GKYL_CU_DH void fluid_vars_limitery_3x_ser_p1(double limiter_fac, const struct gkyl_wv_eqn *wv_eqn, double *fluid_l, double *fluid_c, double *fluid_r); 
GKYL_CU_DH double euler_surfy_3x_ser_p1(const double *w, const double *dxv, const struct gkyl_wv_eqn *wv_eqn, 
    const struct gkyl_wave_cell_geom *geom_l, const struct gkyl_wave_cell_geom *geom_r, 
    const double *u_surf_l, const double *u_surf_c, const double *u_surf_r, 
    const double *p_surf_l, const double *p_surf_c, const double *p_surf_r, 
    const double *fluid_l, const double *fluid_c, const double *fluid_r, double* GKYL_RESTRICT out); 
GKYL_CU_DH void fluid_vars_limiterz_3x_ser_p1(double limiter_fac, const struct gkyl_wv_eqn *wv_eqn, double *fluid_l, double *fluid_c, double *fluid_r); 
GKYL_CU_DH double euler_surfz_3x_ser_p1(const double *w, const double *dxv, const struct gkyl_wv_eqn *wv_eqn, 
    const struct gkyl_wave_cell_geom *geom_l, const struct gkyl_wave_cell_geom *geom_r, 
    const double *u_surf_l, const double *u_surf_c, const double *u_surf_r, 
    const double *p_surf_l, const double *p_surf_c, const double *p_surf_r, 
    const double *fluid_l, const double *fluid_c, const double *fluid_r, double* GKYL_RESTRICT out); 

GKYL_CU_DH void fluid_vars_pressure_2x_tensor_p2(double param, const double *fluid, const double *u, double* GKYL_RESTRICT p, double* GKYL_RESTRICT p_surf); 
GKYL_CU_DH int fluid_vars_u_set_2x_tensor_p2(int count, struct gkyl_nmat *A, struct gkyl_nmat *rhs, const double *fluid); 
GKYL_CU_DH void fluid_vars_u_copy_2x_tensor_p2(int count, struct gkyl_nmat *x, double* GKYL_RESTRICT u, double* GKYL_RESTRICT u_surf); 
GKYL_CU_DH double euler_vol_2x_tensor_p2(const double *w, const double *dxv, double gas_gamma, const double *u, const double *p, const double *fluid, double* GKYL_RESTRICT out); 
GKYL_CU_DH void fluid_vars_limiterx_2x_tensor_p2(double limiter_fac, const struct gkyl_wv_eqn *wv_eqn, double *fluid_l, double *fluid_c, double *fluid_r); 
GKYL_CU_DH double euler_surfx_2x_tensor_p2(const double *w, const double *dxv, const struct gkyl_wv_eqn *wv_eqn, 
    const struct gkyl_wave_cell_geom *geom_l, const struct gkyl_wave_cell_geom *geom_r, 
    const double *u_surf_l, const double *u_surf_c, const double *u_surf_r, 
    const double *p_surf_l, const double *p_surf_c, const double *p_surf_r, 
    const double *fluid_l, const double *fluid_c, const double *fluid_r, double* GKYL_RESTRICT out); 
GKYL_CU_DH void fluid_vars_limitery_2x_tensor_p2(double limiter_fac, const struct gkyl_wv_eqn *wv_eqn, double *fluid_l, double *fluid_c, double *fluid_r); 
GKYL_CU_DH double euler_surfy_2x_tensor_p2(const double *w, const double *dxv, const struct gkyl_wv_eqn *wv_eqn, 
    const struct gkyl_wave_cell_geom *geom_l, const struct gkyl_wave_cell_geom *geom_r, 
    const double *u_surf_l, const double *u_surf_c, const double *u_surf_r, 
    const double *p_surf_l, const double *p_surf_c, const double *p_surf_r, 
    const double *fluid_l, const double *fluid_c, const double *fluid_r, double* GKYL_RESTRICT out);

inline static double minmod(double a, double b, double c)
{
  return gkyl_minmod_3(a, b, c);
  // double sa = GKYL_SGN(a);
  // double sb = GKYL_SGN(b);
  // double sc = GKYL_SGN(c);
  // if( (sa==sb) && (sb==sc) ) {
  //   if (sa<0)
  //     return GKYL_MAX(GKYL_MAX(a,b),c);
  //   else
  //     return GKYL_MIN(GKYL_MIN(a,b),c);
  // }
  // else {
  //    return 0;
  // }
}

EXTERN_C_END 
