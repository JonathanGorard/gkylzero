#include <assert.h>
#include <math.h>
#include <time.h>

#include <gkyl_alloc.h>
#include <gkyl_mom_type.h>
#include <gkyl_mom_gyrokinetic.h>
#include <gkyl_mom_vlasov.h>
#include <gkyl_mom_vlasov_pkpm.h>
#include <gkyl_mom_vlasov_sr.h>
#include <gkyl_dg_updater_moment.h>
#include <gkyl_dg_updater_moment_priv.h>
#include <gkyl_mom_calc.h>
#include <gkyl_util.h>

struct gkyl_mom_type*
gkyl_dg_updater_moment_acquire_type(const gkyl_dg_updater_moment* moment)
{
  return gkyl_mom_type_acquire(moment->type);
}

int
gkyl_dg_updater_moment_num_mom(const gkyl_dg_updater_moment* moment)
{
  return gkyl_mom_type_num_mom(moment->type);
}

struct gkyl_dg_updater_moment*
gkyl_dg_updater_moment_new(const struct gkyl_rect_grid *grid, 
  const struct gkyl_basis *cbasis, const struct gkyl_basis *pbasis, 
  const struct gkyl_range *conf_range, const struct gkyl_range *vel_range,
  enum gkyl_model_id model_id, const char *mom, 
  bool is_integrated, double mass, bool use_gpu)
{
  gkyl_dg_updater_moment *up = gkyl_malloc(sizeof(gkyl_dg_updater_moment));
  up->model_id = model_id;
  if (up->model_id == GKYL_MODEL_SR) {
    if (is_integrated)
      up->type = gkyl_int_mom_vlasov_sr_new(cbasis, pbasis, conf_range, vel_range, use_gpu);
    else
      up->type = gkyl_mom_vlasov_sr_new(cbasis, pbasis, conf_range, vel_range, mom, use_gpu);
  }
  else if (up->model_id == GKYL_MODEL_PKPM) {
    up->type = gkyl_mom_vlasov_pkpm_new(cbasis, pbasis, mass, is_integrated, use_gpu);
  }
  else {
    if (is_integrated)
      up->type = gkyl_int_mom_vlasov_new(cbasis, pbasis, use_gpu);
    else
      up->type = gkyl_mom_vlasov_new(cbasis, pbasis, mom, use_gpu);
  }

  up->up_moment = gkyl_mom_calc_new(grid, up->type, use_gpu);

  up->moment_tm = 0.0;
  
  return up;
}

void
gkyl_dg_updater_moment_advance(struct gkyl_dg_updater_moment *moment,
  const struct gkyl_range *update_phase_rng, const struct gkyl_range *update_conf_rng,
  const struct gkyl_array *p_over_gamma, const struct gkyl_array *gamma, 
  const struct gkyl_array *gamma_inv, const struct gkyl_array *V_drift, 
  const struct gkyl_array *GammaV2, const struct gkyl_array *GammaV_inv, 
  const struct gkyl_array* GKYL_RESTRICT fIn, struct gkyl_array* GKYL_RESTRICT mout)
{
  // Set arrays needed
  // Assumes a particular order of the arrays
  // TO DO: More intelligent way to do these aux field sets? (JJ: 09/08/22)
  if (moment->model_id == GKYL_MODEL_SR) {
    gkyl_mom_vlasov_sr_set_auxfields(moment->type, 
      (struct gkyl_mom_vlasov_sr_auxfields) { .p_over_gamma = p_over_gamma, 
        .gamma = gamma, .gamma_inv = gamma_inv, 
        .V_drift = V_drift, .GammaV2 = GammaV2, .GammaV_inv = GammaV_inv });
  }
  
  struct timespec wst = gkyl_wall_clock();
  gkyl_mom_calc_advance(moment->up_moment, update_phase_rng, update_conf_rng, fIn, mout);
  moment->moment_tm += gkyl_time_diff_now_sec(wst);
}

struct gkyl_dg_updater_moment_tm
gkyl_dg_updater_moment_get_tm(const gkyl_dg_updater_moment *moment)
{
  return (struct gkyl_dg_updater_moment_tm) {
    .moment_tm = moment->moment_tm,
  };
}

void
gkyl_dg_updater_moment_release(gkyl_dg_updater_moment* moment)
{
  gkyl_mom_type_release(moment->type);
  gkyl_mom_calc_release(moment->up_moment);
  gkyl_free(moment);
}

#ifdef GKYL_HAVE_CUDA

void
gkyl_dg_updater_moment_advance_cu(gkyl_dg_updater_moment *moment,
  const struct gkyl_range *update_phase_rng, const struct gkyl_range *update_conf_rng,
  const struct gkyl_array *p_over_gamma, const struct gkyl_array *gamma, 
  const struct gkyl_array *gamma_inv, const struct gkyl_array *V_drift, 
  const struct gkyl_array *GammaV2, const struct gkyl_array *GammaV_inv, 
  const struct gkyl_array* GKYL_RESTRICT fIn, struct gkyl_array* GKYL_RESTRICT mout)
{
  // Set arrays needed
  // Assumes a particular order of the arrays
  // TO DO: More intelligent way to do these aux field sets? (JJ: 09/08/22)
  if (moment->model_id == GKYL_MODEL_SR) {
    gkyl_mom_vlasov_sr_set_auxfields(moment->type, 
      (struct gkyl_mom_vlasov_sr_auxfields) { .p_over_gamma = p_over_gamma, 
        .gamma = gamma, .gamma_inv = gamma_inv, 
        .V_drift = V_drift, .GammaV2 = GammaV2, .GammaV_inv = GammaV_inv });
  }
  
  struct timespec wst = gkyl_wall_clock();
  gkyl_mom_calc_advance_cu(moment->up_moment, update_phase_rng, update_conf_rng, fIn, mout);
  moment->moment_tm += gkyl_time_diff_now_sec(wst);
}

#endif

#ifndef GKYL_HAVE_CUDA

void
gkyl_dg_updater_moment_advance_cu(gkyl_dg_updater_moment *moment,
  const struct gkyl_range *update_phase_rng, const struct gkyl_range *update_conf_rng,
  const struct gkyl_array *p_over_gamma, const struct gkyl_array *gamma, 
  const struct gkyl_array *gamma_inv, const struct gkyl_array *V_drift, 
  const struct gkyl_array *GammaV2, const struct gkyl_array *GammaV_inv, 
  const struct gkyl_array* GKYL_RESTRICT fIn, struct gkyl_array* GKYL_RESTRICT mout)
{
  assert(false);
}

#endif