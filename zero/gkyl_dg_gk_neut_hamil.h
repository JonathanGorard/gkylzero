#pragma once

#include <gkyl_array.h>
#include <gkyl_range.h>
#include <gkyl_rect_grid.h>
#include <gkyl_basis.h>

// Object type
typedef struct gkyl_dg_gk_neut_hamil gkyl_dg_gk_neut_hamil;

/**
 * Create new updater to compute continuous Hamiltonian
 * needed for gk neut species canonical Poisson bracket
 * formulation. 
 * h = 1/2*g^ij*w_i*w_j
 * 
 * @param phase_grid Phase-space grid (for getting cell spacing and cell center in pressure calculation) 
 * @param conf_basis Configuration-space basis functions
 * @param vel_basis  Momentum (four-velocity)-space basis functions
 * @param use_gpu bool to determine if on GPU
 * @return New updater pointer.
 */
struct gkyl_dg_gk_neut_hamil* 
gkyl_dg_gk_neut_hamil_new(const struct gkyl_rect_grid *phase_grid,
  const struct gkyl_basis *conf_basis, const struct gkyl_basis *vel_basis, 
  bool use_gpu);

/**
 * Create new updater to compute relativistic variables on
 * NV-GPU. See new() method for documentation.
 */
struct gkyl_dg_gk_neut_hamil* 
gkyl_dg_gk_neut_hamil_cu_dev_new(const struct gkyl_rect_grid *phase_grid,
  const struct gkyl_basis *conf_basis, const struct gkyl_basis *vel_basis);

/**
 * Compute the Hamiltonian
 * Uses special kernels which convert between a Gauss-Lobatto nodal basis and
 * our modal basis to insure continuity of the Hamiltonian.
 *
 * @param up      Updater for computing Hamiltonian 
 * @param gij     Input array (6-vector) of geometric coefficients
 * @param hamil   Output array of Hamiltonian in phase space grid
 */
void gkyl_dg_gk_neut_hamil_calc(struct gkyl_dg_gk_neut_hamil *up, 
  const struct gkyl_array* gij, struct gkyl_array* hamil);

/**
 * Delete pointer to updater.
 *
 * @param up Updater to delete.
 */
void gkyl_dg_gk_neut_hamil_release(struct gkyl_dg_gk_neut_hamil *up);

/**
 * Host-side wrappers for sr vars operations on device
 */

void gkyl_dg_gk_neut_hamil_calc_cu(struct gkyl_dg_gk_neut_hamil *up, 
  const struct gkyl_array* gij, struct gkyl_array* hamil);