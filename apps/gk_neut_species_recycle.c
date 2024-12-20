#include <assert.h>
#include <gkyl_mom_canonical_pb.h>
#include <gkyl_gyrokinetic_priv.h>

void
gk_neut_species_recycle_init(struct gkyl_gyrokinetic_app *app, struct gk_recycle_wall *recyc,
  int dir, enum gkyl_edge_loc edge, void *ctx, bool use_gpu)
{
  struct gkyl_bc_emission_ctx *params = ctx;
  recyc->params = params;
  recyc->num_species = params->num_species;
  recyc->edge = edge;
  recyc->dir = dir;
  recyc->elastic = params->elastic;
  recyc->t_bound = params->t_bound;
}

void
gk_neut_species_recycle_cross_init(struct gkyl_gyrokinetic_app *app, struct gk_neut_species *s,
  struct gk_recycle_wall *recyc)
{
  int cdim = app->cdim;
  int vdim = app->vdim+1; // from gk_neut_species
  int bdir = (recyc->edge == GKYL_LOWER_EDGE) ? 2*recyc->dir : 2*recyc->dir+1;
 
  int ghost[GKYL_MAX_DIM];
  for (int d=0; d<cdim; ++d) {
    ghost[d] = 1;
  }
  for (int d=0; d<vdim; ++d) {
    ghost[cdim+d] = 0;
  }

  recyc->rec_frac = 1.; // HARDCODED
  
  recyc->emit_grid = &s->bflux.boundary_grid[bdir];
  recyc->emit_buff_r = &s->bflux.flux_r[bdir];
  recyc->emit_cbuff_r = &s->bflux.conf_r[bdir];
  recyc->emit_ghost_r = (recyc->edge == GKYL_LOWER_EDGE) ? &s->lower_ghost[recyc->dir] : &s->upper_ghost[recyc->dir];
  recyc->emit_skin_r = (recyc->edge == GKYL_LOWER_EDGE) ? &s->lower_skin[recyc->dir] : &s->upper_skin[recyc->dir];
  recyc->buffer = (recyc->edge == GKYL_LOWER_EDGE) ? s->bc_buffer_lo_recyc : s->bc_buffer_up_recyc;
  
  recyc->f_emit = mkarr(app->use_gpu, app->neut_basis.num_basis, recyc->emit_buff_r->volume);
  gkyl_array_clear(recyc->f_emit, 0.0);
  gkyl_array_accumulate(recyc->f_emit, 1.0, recyc->buffer);
  struct gkyl_array *proj_buffer = mkarr(false, app->neut_basis.num_basis, recyc->emit_buff_r->volume);

  // Copy LTE to f_emit for bcs
  //gkyl_array_set_range_to_range(recyc->f_emit, 1, recyc->buffer, recyc->emit_buff_r, recyc->emit_buff_r);
  //gkyl_array_copy_from_buffer(recyc->f_emit, recyc->buffer, recyc->emit_buff_r);
  const char *fmt = "recyc_f_buffer_%d.gkyl";
  int sz = gkyl_calc_strlen(fmt, recyc->edge);
  char fileNm[sz+1]; // ensures no buffer overflow
  snprintf(fileNm, sizeof fileNm, fmt, recyc->edge);
  gkyl_grid_sub_array_write(recyc->emit_grid, recyc->emit_buff_r, 0, recyc->f_emit, fileNm);

  // Calculate the flux
  gkyl_bc_emission_flux_ranges(&recyc->emit_normal_r, recyc->dir + cdim, recyc->emit_buff_r,
    ghost, recyc->edge);
  // RENAME...
  recyc->init_flux = mkarr(app->use_gpu, app->confBasis.num_basis, recyc->emit_cbuff_r->volume);
  recyc->init_bflux_arr = s->bflux.flux_arr[bdir];
  /* const char *fmt1 = "recyc_emit_flux_edge_%d.gkyl"; */
  /* int sz1 = gkyl_calc_strlen(fmt1, recyc->edge); */
  /* char fileNm1[sz1+1]; // ensures no buffer overflow */
  /* snprintf(fileNm1, sizeof fileNm1, fmt1, recyc->edge); */
  /* gkyl_grid_sub_array_write(recyc->emit_grid, recyc->emit_buff_r, 0, recyc->emit_bflux_arr, fileNm1); */

  struct gkyl_mom_canonical_pb_auxfields can_pb_inp = {.hamil = s->hamil};
  recyc->init_flux_slvr = gkyl_dg_updater_moment_new(recyc->emit_grid, &app->confBasis,
    &app->neut_basis, recyc->emit_cbuff_r, &s->local_vel, recyc->emit_buff_r, s->model_id,
    &can_pb_inp, "M0", false, app->use_gpu);
  
  gkyl_dg_updater_moment_advance(recyc->init_flux_slvr, &recyc->emit_normal_r, recyc->emit_cbuff_r, recyc->init_bflux_arr,
				 recyc->init_flux);

  // try to write out the emit_flux. This is zero for some reason? Find out when bflux is called! Needed for neutrals, too.

  if (app->use_gpu) {
      recyc->mem_geo = gkyl_dg_bin_op_mem_cu_dev_new(recyc->emit_cbuff_r->volume, app->confBasis.num_basis);
    }
    else {
      recyc->mem_geo = gkyl_dg_bin_op_mem_new(recyc->emit_cbuff_r->volume, app->confBasis.num_basis);
    }
  
  // Initialize elastic component of emission
  if (recyc->elastic) {
    recyc->elastic_yield = mkarr(app->use_gpu, app->neut_basis.num_basis, recyc->emit_buff_r->volume);
    recyc->elastic_update = gkyl_bc_emission_elastic_new(recyc->params->elastic_model,
      recyc->elastic_yield, recyc->dir, recyc->edge, cdim, vdim, s->info.mass, s->f->ncomp, recyc->emit_grid,
      recyc->emit_buff_r, app->poly_order, app->basis_on_dev.basis, &app->neut_basis, proj_buffer,
      app->use_gpu);
  }

  // Initialize inelastic emission spectrums
  for (int i=0; i<recyc->num_species; ++i) {
    recyc->impact_species[i] = gk_find_species(app, recyc->params->in_species[i]);
    recyc->impact_grid[i] = &recyc->impact_species[i]->bflux.boundary_grid[bdir];

    recyc->impact_skin_r[i] = (recyc->edge == GKYL_LOWER_EDGE) ? &recyc->impact_species[i]->lower_skin[recyc->dir] : &recyc->impact_species[i]->upper_skin[recyc->dir];
    recyc->impact_ghost_r[i] = (recyc->edge == GKYL_LOWER_EDGE) ? &recyc->impact_species[i]->lower_ghost[recyc->dir] : &recyc->impact_species[i]->upper_ghost[recyc->dir];
    recyc->impact_buff_r[i] = &recyc->impact_species[i]->bflux.flux_r[bdir];
    recyc->impact_cbuff_r[i] = &recyc->impact_species[i]->bflux.conf_r[bdir];

    recyc->flux_slvr[i] = gkyl_dg_updater_moment_gyrokinetic_new(recyc->impact_grid[i], &app->confBasis,
      &app->basis, recyc->emit_cbuff_r, recyc->impact_species[i]->info.mass, recyc->impact_species[i]->vel_map,
      app->gk_geom, "M0", 0, app->use_gpu);
    
    recyc->flux[i] = mkarr(app->use_gpu, app->confBasis.num_basis, recyc->impact_cbuff_r[i]->volume);
    recyc->bflux_arr[i] = recyc->impact_species[i]->bflux.flux_arr[bdir];
    const char *fmt2 = "recyc_impact_edge_%d.gkyl"; 
    int sz2 = gkyl_calc_strlen(fmt2, recyc->edge);
    char fileNm2[sz2+1]; // ensures no buffer overflow
    snprintf(fileNm2, sizeof fileNm2, fmt2, recyc->edge);
    //gkyl_grid_sub_array_write(recyc->impact_grid[i], recyc->impact_buff_r[i], 0, recyc->bflux_arr[i], fileNm2);

    gkyl_bc_emission_flux_ranges(&recyc->impact_normal_r[i], recyc->dir + cdim, recyc->impact_buff_r[i],
      ghost, recyc->edge);
  }
  gkyl_array_release(proj_buffer);
}

void
gk_neut_species_recycle_apply_bc(struct gkyl_gyrokinetic_app *app, const struct gk_recycle_wall *recyc,
  struct gkyl_array *fout)
{
  // Optional scaling of emission with time
  double t_scale = 1.0;
  /* if (recyc->t_bound) */
  /*   t_scale = sin(M_PI*tcurr/(2.0*recyc->t_bound)); */

  gkyl_array_clear(recyc->f_emit, 0.0); // Zero emitted distribution before beginning accumulate
  gkyl_array_accumulate(recyc->f_emit, 1.0, recyc->buffer);
  //gkyl_array_set_range_to_range(recyc->f_emit, recyc->rec_frac, recyc->buffer, recyc->emit_buff_r, recyc->emit_buff_r);

  // Elastic emission contribution
  if (recyc->elastic) {
    gkyl_bc_emission_elastic_advance(recyc->elastic_update, recyc->emit_skin_r, recyc->buffer, fout,
      recyc->f_emit, recyc->elastic_yield, &app->neut_basis);
  }
  // Inelastic emission contribution
  for (int i=0; i<recyc->num_species; ++i) {
    int species_idx;
    species_idx = gk_find_species_idx(app, recyc->impact_species[i]->info.name);
    
    gkyl_dg_updater_moment_gyrokinetic_advance(recyc->flux_slvr[i], &recyc->impact_normal_r[i],
      recyc->emit_cbuff_r, recyc->bflux_arr[i], recyc->flux[i]);

    // bin op divide
    gkyl_dg_div_op_range(recyc->mem_geo, app->confBasis, 0, recyc->flux[i], 0, recyc->flux[i],
			 0, recyc->init_flux, recyc->emit_cbuff_r);
    
    // conf mult onto f_emit
    /* void gkyl_dg_mul_conf_phase_op_range(struct gkyl_basis *cbasis, */
    /* struct gkyl_basis *pbasis, struct gkyl_array* pout, */
    /* const struct gkyl_array* cop, const struct gkyl_array* pop, */
    /* const struct gkyl_range *crange, const struct gkyl_range *prange); */
    gkyl_dg_mul_conf_phase_op_range(&app->confBasis, &app->neut_basis, recyc->f_emit, recyc->flux[i],
				    recyc->f_emit, recyc->impact_cbuff_r[i], recyc->emit_buff_r);
    
  }
  const char *fmt3 = "recyc_f_emit_edge_%d.gkyl"; 
  int sz3 = gkyl_calc_strlen(fmt3, recyc->edge);
  char fileNm3[sz3+1]; // ensures no buffer overflow
  snprintf(fileNm3, sizeof fileNm3, fmt3, recyc->edge);
  gkyl_grid_sub_array_write(recyc->emit_grid, recyc->emit_buff_r, 0, recyc->f_emit, fileNm3);
  
  gkyl_array_set_range_to_range(fout, t_scale, recyc->f_emit, recyc->emit_ghost_r,
    recyc->emit_buff_r);

}

void
gk_neut_species_recycle_release(const struct gk_recycle_wall *recyc)
{
  gkyl_array_release(recyc->f_emit);
  gkyl_array_release(recyc->init_flux);
  gkyl_dg_updater_moment_release(recyc->init_flux_slvr);
  if (recyc->elastic) {
    gkyl_array_release(recyc->elastic_yield);
    gkyl_bc_emission_elastic_release(recyc->elastic_update);
  }
  for (int i=0; i<recyc->num_species; ++i) {
    gkyl_array_release(recyc->yield[i]);
    gkyl_array_release(recyc->spectrum[i]);
    gkyl_array_release(recyc->weight[i]);
    gkyl_array_release(recyc->flux[i]);
    gkyl_array_release(recyc->k[i]);
    gkyl_dg_updater_moment_release(recyc->flux_slvr[i]);
    gkyl_bc_emission_spectrum_release(recyc->update[i]);
  }
}
