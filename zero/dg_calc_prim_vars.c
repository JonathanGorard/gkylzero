#include <assert.h>

#include <gkyl_alloc.h>
#include <gkyl_array_ops.h>
#include <gkyl_array_ops_priv.h>
#include <gkyl_dg_calc_prim_vars.h>
#include <gkyl_dg_calc_prim_vars_priv.h>
#include <gkyl_util.h>

void gkyl_calc_prim_vars_u_from_statevec(gkyl_dg_bin_op_mem *mem, struct gkyl_basis basis, const struct gkyl_range *range,
  const struct gkyl_array* statevec, struct gkyl_array* u_i)
{
  // Find number of components of flow vector
  int num_comp = u_i->ncomp/basis.num_basis;
  for (int i = 0; i<num_comp; ++i)
    gkyl_dg_div_op_range(mem, basis, 
      i, u_i, i+1, statevec, 0, statevec, range);  
}

void gkyl_calc_prim_vars_u_from_rhou(gkyl_dg_bin_op_mem *mem, struct gkyl_basis basis, const struct gkyl_range *range,
  const struct gkyl_array* rho, const struct gkyl_array* rhou, struct gkyl_array* u_i)
{
  // Find number of components of flow vector
  int num_comp = u_i->ncomp/basis.num_basis;
  for (int i = 0; i<num_comp; ++i)
    gkyl_dg_div_op_range(mem, basis, 
      i, u_i, i, rhou, 0, rho, range);  
}

void gkyl_calc_prim_vars_p_from_statevec(struct gkyl_basis basis, const struct gkyl_range *range,
  const double p_fac,  const struct gkyl_array* u_i, const struct gkyl_array* statevec,
  struct gkyl_array* p_ij)
{
  int cdim = basis.ndim;
  int poly_order = basis.poly_order;
  euler_pressure_t pressure = choose_ser_euler_pressure_kern(cdim, poly_order);
  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, range);
  while (gkyl_range_iter_next(&iter)) {
    long loc = gkyl_range_idx(range, iter.idx);

    const double *statevec_d = gkyl_array_cfetch(statevec, loc);
    const double *u_i_d = gkyl_array_cfetch(u_i, loc);
    double *p_ij_d = gkyl_array_fetch(p_ij, loc);
    pressure(p_fac, u_i_d, statevec_d, p_ij_d);
  }
}

void gkyl_calc_prim_vars_pkpm(struct gkyl_basis basis, const struct gkyl_range *range,
  const struct gkyl_array* bvar, const struct gkyl_array* vlasov_pkpm_moms, const struct gkyl_array* euler_pkpm, 
  struct gkyl_array* u_i, struct gkyl_array* p_ij, struct gkyl_array* T_ij, 
  struct gkyl_array* rho_inv, struct gkyl_array* T_perp_over_m, struct gkyl_array* T_perp_over_m_inv)
{
// Check if more than one of the output arrays is on device? 
// Probably a better way to do this (JJ: 11/16/22)
#ifdef GKYL_HAVE_CUDA
  if (gkyl_array_is_cu_dev(u_i)) {
    return gkyl_calc_prim_vars_pkpm_cu(basis, range, bvar, vlasov_pkpm_moms, euler_pkpm, 
      u_i, p_ij, T_ij, 
      rho_inv, T_perp_over_m, T_perp_over_m_inv);
  }
#endif

  int cdim = basis.ndim;
  int poly_order = basis.poly_order;
  euler_pkpm_prim_vars_t pkpm_prim_vars = choose_ser_euler_pkpm_prim_vars_kern(cdim, poly_order);
  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, range);
  while (gkyl_range_iter_next(&iter)) {
    long loc = gkyl_range_idx(range, iter.idx);

    const double *bvar_d = gkyl_array_cfetch(bvar, loc);
    const double *vlasov_pkpm_moms_d = gkyl_array_cfetch(vlasov_pkpm_moms, loc);
    const double *euler_pkpm_d = gkyl_array_cfetch(euler_pkpm, loc);
    
    double *u_i_d = gkyl_array_fetch(u_i, loc);
    double *p_ij_d = gkyl_array_fetch(p_ij, loc);
    double *T_ij_d = gkyl_array_fetch(T_ij, loc);
    double *rho_inv_d = gkyl_array_fetch(rho_inv, loc);
    double *T_perp_over_m_d = gkyl_array_fetch(T_perp_over_m, loc);
    double *T_perp_over_m_inv_d = gkyl_array_fetch(T_perp_over_m_inv, loc);
    pkpm_prim_vars(bvar_d, vlasov_pkpm_moms_d, euler_pkpm_d, 
      u_i_d, p_ij_d, T_ij_d, 
      rho_inv_d, T_perp_over_m_d, T_perp_over_m_inv_d);
  }
}

void gkyl_calc_prim_vars_pkpm_source(struct gkyl_basis basis, const struct gkyl_range *range,
  const struct gkyl_array* qmem, const struct gkyl_array* vlasov_pkpm_moms, const struct gkyl_array* euler_pkpm,
  struct gkyl_array* rhs)
{
#ifdef GKYL_HAVE_CUDA
  if (gkyl_array_is_cu_dev(rhs)) {
    return gkyl_calc_prim_vars_pkpm_source_cu(basis, range, 
      qmem, vlasov_pkpm_moms, euler_pkpm, rhs);
  }
#endif

  int cdim = basis.ndim;
  int poly_order = basis.poly_order;
  euler_pkpm_source_t pkpm_source = choose_ser_euler_pkpm_source_kern(cdim, poly_order);
  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, range);
  while (gkyl_range_iter_next(&iter)) {
    long loc = gkyl_range_idx(range, iter.idx);

    const double *qmem_d = gkyl_array_cfetch(qmem, loc);
    const double *vlasov_pkpm_moms_d = gkyl_array_cfetch(vlasov_pkpm_moms, loc);
    const double *euler_pkpm_d = gkyl_array_cfetch(euler_pkpm, loc);

    double *rhs_d = gkyl_array_fetch(rhs, loc);
    pkpm_source(qmem_d, vlasov_pkpm_moms_d, euler_pkpm_d, rhs_d);
  }
}

void gkyl_calc_prim_vars_pkpm_dist_mirror_force(struct gkyl_basis basis, 
  const struct gkyl_range *conf_range, const struct gkyl_range *phase_range,
  const struct gkyl_array* T_perp_over_m, const struct gkyl_array* T_perp_over_m_inv, 
  const struct gkyl_array* fIn, const struct gkyl_array* F_k_p_1,
  struct gkyl_array* g_dist_source, struct gkyl_array* F_k_m_1)
{
#ifdef GKYL_HAVE_CUDA
  if (gkyl_array_is_cu_dev(g_dist_source)) {
    return gkyl_calc_prim_vars_pkpm_dist_mirror_force_cu(basis, conf_range, phase_range, 
      T_perp_over_m, T_perp_over_m_inv, fIn, F_k_p_1, g_dist_source, F_k_m_1);
  }
#endif  

  int cdim = basis.ndim;
  int poly_order = basis.poly_order;
  pkpm_dist_mirror_force_t pkpm_dist_mirror_force = choose_ser_pkpm_dist_mirror_force_kern(cdim, poly_order);
  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, phase_range);
  while (gkyl_range_iter_next(&iter)) {
    long loc_conf = gkyl_range_idx(conf_range, iter.idx);
    long loc_phase = gkyl_range_idx(phase_range, iter.idx);

    const double *T_perp_over_m_d = gkyl_array_cfetch(T_perp_over_m, loc_conf);
    const double *T_perp_over_m_inv_d = gkyl_array_cfetch(T_perp_over_m_inv, loc_conf);
    const double *fIn_d = gkyl_array_cfetch(fIn, loc_phase);
    const double *F_k_p_1_d = gkyl_array_cfetch(F_k_p_1, loc_phase);

    double *g_dist_source_d = gkyl_array_fetch(g_dist_source, loc_phase);
    double *F_k_m_1_d = gkyl_array_fetch(F_k_m_1, loc_phase);

    pkpm_dist_mirror_force(T_perp_over_m_d, T_perp_over_m_inv_d, fIn_d, F_k_p_1_d, g_dist_source_d, F_k_m_1_d);
  }  
}

void gkyl_calc_prim_vars_pkpm_recovery(const struct gkyl_rect_grid *grid, 
  struct gkyl_basis basis, const struct gkyl_range *range, double nuHyp, 
  const struct gkyl_array* bvar, const struct gkyl_array* u_i, 
  const struct gkyl_array* p_ij, const struct gkyl_array* vlasov_pkpm_moms, const struct gkyl_array* euler_pkpm, 
  const struct gkyl_array* rho_inv, const struct gkyl_array* T_perp_over_m, const struct gkyl_array* T_perp_over_m_inv, 
  const struct gkyl_array* nu, const struct gkyl_array* nu_vthsq,  
  struct gkyl_array* div_p, struct gkyl_array* pkpm_accel_vars)
{
// Check if more than one of the output arrays is on device? 
// Probably a better way to do this (JJ: 11/16/22)
#ifdef GKYL_HAVE_CUDA
  if (gkyl_array_is_cu_dev(div_p)) {
    return gkyl_calc_prim_vars_pkpm_recovery_cu(grid, basis, range, nuHyp, 
      bvar, u_i, p_ij, vlasov_pkpm_moms, euler_pkpm, 
      rho_inv, T_perp_over_m, T_perp_over_m_inv, 
      nu, nu_vthsq, 
      div_p, pkpm_accel_vars);
  }
#endif

  int cdim = basis.ndim;
  int poly_order = basis.poly_order;
  euler_pkpm_recovery_t pkpm_recovery[3];
  // Fetch the kernels in each direction
  for (int d=0; d<cdim; ++d) 
    pkpm_recovery[d] = choose_ser_euler_pkpm_recovery_kern(d, cdim, poly_order);

  int idxl[GKYL_MAX_DIM], idxc[GKYL_MAX_DIM], idxr[GKYL_MAX_DIM];

  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, range);
  while (gkyl_range_iter_next(&iter)) {
    gkyl_copy_int_arr(cdim, iter.idx, idxc);
    long linc = gkyl_range_idx(range, idxc);

    const double *bvar_c = gkyl_array_cfetch(bvar, linc);
    const double *u_i_c = gkyl_array_cfetch(u_i, linc);
    const double *p_ij_c = gkyl_array_cfetch(p_ij, linc);
    const double *vlasov_pkpm_moms_c = gkyl_array_cfetch(vlasov_pkpm_moms, linc);
    const double *euler_pkpm_c = gkyl_array_cfetch(euler_pkpm, linc);

    // Only need rho_inv, T_perp_over_m, T_perp_over_m_inv, nu, and nu_vthsq in center cell
    const double *rho_inv_d = gkyl_array_cfetch(rho_inv, linc);
    const double *T_perp_over_m_d = gkyl_array_cfetch(T_perp_over_m, linc);
    const double *T_perp_over_m_inv_d = gkyl_array_cfetch(T_perp_over_m_inv, linc);
    const double *nu_d = gkyl_array_cfetch(nu, linc);
    const double *nu_vthsq_d = gkyl_array_cfetch(nu_vthsq, linc);

    double *div_p_d = gkyl_array_fetch(div_p, linc);
    double *pkpm_accel_vars_d = gkyl_array_fetch(pkpm_accel_vars, linc);

    for (int dir=0; dir<cdim; ++dir) {
      gkyl_copy_int_arr(cdim, iter.idx, idxl);
      gkyl_copy_int_arr(cdim, iter.idx, idxr);

      idxl[dir] = idxl[dir]-1; idxr[dir] = idxr[dir]+1;

      long linl = gkyl_range_idx(range, idxl); 
      long linr = gkyl_range_idx(range, idxr);

      const double *bvar_l = gkyl_array_cfetch(bvar, linl);
      const double *bvar_r = gkyl_array_cfetch(bvar, linr);

      const double *u_i_l = gkyl_array_cfetch(u_i, linl);
      const double *u_i_r = gkyl_array_cfetch(u_i, linr);

      const double *p_ij_l = gkyl_array_cfetch(p_ij, linl);
      const double *p_ij_r = gkyl_array_cfetch(p_ij, linr);

      const double *vlasov_pkpm_moms_l = gkyl_array_cfetch(vlasov_pkpm_moms, linl);
      const double *vlasov_pkpm_moms_r = gkyl_array_cfetch(vlasov_pkpm_moms, linr);

      const double *euler_pkpm_l = gkyl_array_cfetch(euler_pkpm, linl);
      const double *euler_pkpm_r = gkyl_array_cfetch(euler_pkpm, linr);

      pkpm_recovery[dir](grid->dx, nuHyp, 
        bvar_l, bvar_c, bvar_r, u_i_l, u_i_c, u_i_r, 
        p_ij_l, p_ij_c, p_ij_r, vlasov_pkpm_moms_l, vlasov_pkpm_moms_c, vlasov_pkpm_moms_r, 
        euler_pkpm_l, euler_pkpm_c, euler_pkpm_r, 
        rho_inv_d, T_perp_over_m_d, T_perp_over_m_inv_d, 
        nu_d, nu_vthsq_d, 
        div_p_d, pkpm_accel_vars_d);
    }
  }
}