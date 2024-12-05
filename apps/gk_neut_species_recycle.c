#include <assert.h>
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

  recyc->emit_grid = &s->bflux.boundary_grid[bdir];
  recyc->emit_buff_r = &s->bflux.flux_r[bdir]; 
  recyc->emit_ghost_r = (recyc->edge == GKYL_LOWER_EDGE) ? &s->lower_ghost[recyc->dir] : &s->upper_ghost[recyc->dir];
  recyc->emit_skin_r = (recyc->edge == GKYL_LOWER_EDGE) ? &s->lower_skin[recyc->dir] : &s->upper_skin[recyc->dir];
  recyc->buffer = s->bc_buffer;

  recyc->f_emit = mkarr(app->use_gpu, app->neut_basis.num_basis, recyc->emit_buff_r->volume);
  struct gkyl_array *proj_buffer = mkarr(false, app->neut_basis.num_basis, recyc->emit_buff_r->volume);

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
      &app->basis, recyc->impact_cbuff_r[i], recyc->impact_species[i]->info.mass, recyc->impact_species[i]->vel_map,
      app->gk_geom, 0, 1, app->use_gpu);
    
    recyc->yield[i] = mkarr(app->use_gpu, app->basis.num_basis, recyc->impact_buff_r[i]->volume);
    recyc->spectrum[i] = mkarr(app->use_gpu, app->neut_basis.num_basis, recyc->emit_buff_r->volume);
    recyc->weight[i] = mkarr(app->use_gpu, app->confBasis.num_basis,
      recyc->impact_cbuff_r[i]->volume);
    recyc->flux[i] = mkarr(app->use_gpu, app->confBasis.num_basis, recyc->impact_cbuff_r[i]->volume);
    recyc->bflux_arr[i] = recyc->impact_species[i]->bflux.flux_arr[bdir];
    recyc->k[i] = mkarr(app->use_gpu, app->confBasis.num_basis, recyc->impact_cbuff_r[i]->volume);

    // what should second arg be? 
    gkyl_bc_emission_flux_ranges(&recyc->impact_normal_r[i], cdim, recyc->impact_buff_r[i],
      ghost, recyc->edge);

    recyc->update[i] = gkyl_bc_emission_spectrum_new(recyc->params->spectrum_model[i],
      recyc->params->yield_model[i], recyc->yield[i], recyc->spectrum[i], recyc->dir, recyc->edge,
      cdim, vdim, recyc->impact_species[i]->info.mass, s->info.mass, recyc->impact_buff_r[i],
      recyc->emit_buff_r, recyc->impact_grid[i], recyc->emit_grid, app->poly_order,
      &app->neut_basis, proj_buffer, app->use_gpu);
    
  }
  gkyl_array_release(proj_buffer);
}

void
gk_neut_species_recycle_apply_bc(struct gkyl_gyrokinetic_app *app, const struct gk_recycle_wall *recyc,
  struct gkyl_array *fout)
{
  //printf("Applying recycle bcs...\n");
  // Optional scaling of emission with time
  double t_scale = 1.0;
  /* if (recyc->t_bound) */
  /*   t_scale = sin(M_PI*tcurr/(2.0*recyc->t_bound)); */

  gkyl_array_clear(recyc->f_emit, 0.0); // Zero emitted distribution before beginning accumulate

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
      recyc->impact_cbuff_r[i], recyc->bflux_arr[i], recyc->flux[i]);
    
    gkyl_bc_emission_spectrum_advance(recyc->update[i], recyc->impact_buff_r[i],
      recyc->impact_cbuff_r[i], recyc->emit_buff_r, recyc->bflux_arr[i],
      recyc->f_emit, recyc->yield[i], recyc->spectrum[i], recyc->weight[i], recyc->flux[i],
      recyc->k[i]);
  }
  gkyl_array_set_range_to_range(fout, t_scale, recyc->f_emit, recyc->emit_ghost_r,
    recyc->emit_buff_r);
}

void
gk_neut_species_recycle_release(const struct gk_recycle_wall *recyc)
{
  gkyl_array_release(recyc->f_emit);
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
