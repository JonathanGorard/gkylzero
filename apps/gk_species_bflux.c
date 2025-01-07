#include <assert.h>
#include <gkyl_gyrokinetic_priv.h>
#include <gkyl_dg_updater_moment_gyrokinetic.h>

void 
gk_species_bflux_init(struct gkyl_gyrokinetic_app *app, struct gk_species *s, struct gk_boundary_fluxes *bflux)
{ 
  // Allocate solver.
  bflux->flux_slvr = gkyl_ghost_surf_calc_new(&s->grid, s->eqn_gyrokinetic, app->cdim, app->use_gpu);
  int cdim = app->cdim;
  int ndim = app->cdim + app->vdim;
  int cells[GKYL_MAX_DIM], ghost[GKYL_MAX_DIM];
  double lower[GKYL_MAX_DIM], upper[GKYL_MAX_DIM];
  for (int d=0; d<ndim; ++d) {
    cells[d] = s->grid.cells[d];
    lower[d] = s->grid.lower[d];
    upper[d] = s->grid.upper[d];
    ghost[d] = 0;
  }

  // Initialize moment solver. This is for ambipotential solve.
  for (int i=0; i<2*app->cdim; ++i) {
    gk_species_moment_init(app, s, &bflux->gammai[i], "M0");
  }

  // initialize moment solver
  for (int i=0; i<app->cdim; ++i) {
    cells[i] = 1;

    bflux->flux_arr[2*i] = mkarr(app->use_gpu, app->basis.num_basis, s->lower_ghost[i].volume);
    bflux->flux_arr[2*i+1] = mkarr(app->use_gpu, app->basis.num_basis, s->upper_ghost[i].volume);

    gkyl_range_init(&bflux->flux_r[2*i], ndim, s->lower_ghost[i].lower, s->lower_ghost[i].upper);
    gkyl_range_init(&bflux->flux_r[2*i+1], ndim, s->upper_ghost[i].lower, s->upper_ghost[i].upper);

    gkyl_range_init(&bflux->conf_r[2*i], app->cdim, s->lower_ghost[i].lower,
      s->lower_ghost[i].upper);
    gkyl_range_init(&bflux->conf_r[2*i+1], app->cdim, s->upper_ghost[i].lower,
      s->upper_ghost[i].upper);

    upper[i] = s->grid.lower[i] + s->grid.dx[i];

    gkyl_rect_grid_init(&bflux->boundary_grid[2*i], ndim, lower, upper, cells);
    gkyl_rect_grid_init(&bflux->conf_boundary_grid[2*i], cdim, lower, upper, cells);

    upper[i] = s->grid.upper[i];
    lower[i] = s->grid.upper[i] - s->grid.dx[i];

    gkyl_rect_grid_init(&bflux->boundary_grid[2*i+1], ndim, lower, upper, cells);
    gkyl_rect_grid_init(&bflux->conf_boundary_grid[2*i+1], cdim, lower, upper, cells);
	
    bflux->integ_moms[2*i] = gkyl_dg_updater_moment_gyrokinetic_new(&bflux->boundary_grid[2*i],
      &app->confBasis, &app->basis, &bflux->conf_r[2*i], s->info.mass, s->vel_map, app->gk_geom, 0, 1, app->use_gpu);
    bflux->integ_moms[2*i+1] = gkyl_dg_updater_moment_gyrokinetic_new(&bflux->boundary_grid[2*i+1],
      &app->confBasis, &app->basis, &bflux->conf_r[2*i+1], s->info.mass, s->vel_map, app->gk_geom, 0, 1, app->use_gpu);

    cells[i] = s->grid.cells[i];

    bflux->mom_arr[2*i] = mkarr(app->use_gpu, app->confBasis.num_basis, bflux->conf_r[2*i].volume);
    bflux->mom_arr[2*i+1] = mkarr(app->use_gpu, app->confBasis.num_basis, bflux->conf_r[2*i+1].volume);
  }
}

// Computes rhs of the boundary flux.
void
gk_species_bflux_rhs(gkyl_gyrokinetic_app *app, const struct gk_species *species,
  struct gk_boundary_fluxes *bflux, const struct gkyl_array *fin,
  struct gkyl_array *rhs)
{
  // Zero ghost cells before calculation to ensure there's no residual data.
  for (int j=0; j<app->cdim; ++j) {
    gkyl_array_clear_range(rhs, 0.0, &species->lower_ghost[j]);
    gkyl_array_clear_range(rhs, 0.0, &species->upper_ghost[j]);
  }
  // Ghost cells of the rhs array are filled with the bflux
  // This is overwritten by the boundary conditions and is not being stored,
  // it is only currently used to calculate moments for other applications.
  if (app->use_gpu) {
    gkyl_ghost_surf_calc_advance_cu(bflux->flux_slvr, &species->local_ext, fin, rhs);
  } else {
    gkyl_ghost_surf_calc_advance(bflux->flux_slvr, &species->local_ext, fin, rhs);
  }

  // Calculating density for use in ambipotential solve.
  for (int j=0; j<app->cdim; ++j) {
    gk_species_moment_calc(&bflux->gammai[2*j], species->lower_ghost[j], app->lower_ghost[j], rhs);
    gk_species_moment_calc(&bflux->gammai[2*j+1], species->upper_ghost[j], app->upper_ghost[j], rhs);
  }

  // Calculating integrated moments for use in the bflux source.
  for (int j=0; j<app->cdim; ++j) {
    gkyl_array_copy_range_to_range(bflux->flux_arr[2*j], rhs, &bflux->flux_r[2*j],
      &species->lower_ghost[j]);
    gkyl_array_copy_range_to_range(bflux->flux_arr[2*j+1], rhs, &bflux->flux_r[2*j+1],
      &species->upper_ghost[j]);
    
    gkyl_dg_updater_moment_gyrokinetic_advance(bflux->integ_moms[2*j], &bflux->flux_r[2*j],
      &bflux->conf_r[2*j], bflux->flux_arr[2*j], bflux->mom_arr[2*j]);
    gkyl_dg_updater_moment_gyrokinetic_advance(bflux->integ_moms[2*j+1], &bflux->flux_r[2*j+1],
      &bflux->conf_r[2*j+1], bflux->flux_arr[2*j+1], bflux->mom_arr[2*j+1]);
  }
}

void
gk_species_bflux_release(const struct gkyl_gyrokinetic_app *app, const struct gk_boundary_fluxes *bflux)
{
  gkyl_ghost_surf_calc_release(bflux->flux_slvr);
  for (int i=0; i<2*app->cdim; ++i) {
    gkyl_array_release(bflux->mom_arr[i]);
    gk_species_moment_release(app, &bflux->gammai[i]);
    gkyl_dg_updater_moment_release(bflux->integ_moms[i]);
    gkyl_array_release(bflux->flux_arr[i]);
  }
}
