#pragma once

#include <gkyl_array.h>
#include <gkyl_basis.h>
#include <gkyl_evalf_def.h>
#include <gkyl_range.h>
#include <gkyl_rect_grid.h>
#include <gkyl_wave_geom.h>
#include <gkyl_wv_eqn.h>

// Base reconstruction scheme to use
enum gkyl_mp_recon {
  GKYL_MP_C4, // centered fourth-order
  GKYL_MP_C6, // centered sixth-order
  GKYL_MP_U5, // upwind-biased 5th order
};

// Object type for updater
typedef struct gkyl_mp_scheme gkyl_mp_scheme;

// Parameters for constructor
struct gkyl_mp_scheme_inp {
  const struct gkyl_rect_grid *grid; // grid on which to solve equations
  const struct gkyl_wv_eqn *equation; // equation solver

  enum gkyl_mp_recon mp_recon; // base reconstruction to use

  int num_up_dirs; // number of update directions
  int update_dirs[GKYL_MAX_DIM]; // directions to update

  const struct gkyl_wave_geom *geom; // geometry
};

/**
 * Create new updater to update equations using Suresh-Hyunh
 * monotonicity-preserving algorithm
 *
 * @param winp Input for creating updater. See gkyl_mp_scheme_inp above.
 */
gkyl_mp_scheme* gkyl_mp_scheme_new(const struct gkyl_mp_scheme_inp *winp);

/**
 * Compute wave-propagation update. The update_rng MUST be a sub-range
 * of the range on which the array is defined. That is, it must be
 * either the same range as the array range, or one created using the
 * gkyl_sub_range_init method.
 *
 * @param mp Updater object
 * @param update_rng Range on which to compute.
 * @param qin Input to updater
 * @param cflrate CFL scalar rate (frequency) array (units of 1/[T])
 * @param rhs RHS of PDE 
 */
void gkyl_mp_scheme_advance(gkyl_mp_scheme *mp,
  const struct gkyl_range *update_range, const struct gkyl_array *qin,
  struct gkyl_array *cflrate, struct gkyl_array *rhs);

/**
 * Compute an estimate of maximum stable time-step for given input
 * state 'qin'
 *
 * @param mp Updater object
 * @param qin Input to compute dt for
 * @return maximum stable time-step
 */
double gkyl_mp_scheme_max_dt(const gkyl_mp_scheme *mp, const struct gkyl_range *update_range,
  const struct gkyl_array *qin);

/**
 * Delete updater.
 *
 * @param mp Updater to delete.
 */
void gkyl_mp_scheme_release(gkyl_mp_scheme* mp);