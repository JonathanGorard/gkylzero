#pragma once

#include <gkyl_array.h>
#include <gkyl_fpo_vlasov_coeff_recovery_priv.h>
#include <gkyl_basis.h>
#include <gkyl_range.h>
#include <gkyl_rect_grid.h>

// Object type
typedef struct gkyl_fpo_vlasov_coeff_recovery gkyl_fpo_vlasov_coeff_recovery;

/**
 * Create a new updater to compute drag and diffusion coefficients from input potentials.
 * Free with gkyl_fpo_vlasov_coeff_recovery_release.
*/
struct gkyl_fpo_vlasov_coeff_recovery*
gkyl_fpo_vlasov_coeff_recovery_new(const struct gkyl_rect_grid *grid,
    const struct gkyl_basis *phase_basis, const struct gkyl_range *phase_range, bool use_gpu);

/**
 * Create a new updater on device for NV-GPU.
*/
struct gkyl_fpo_vlasov_coeff_recovery*
gkyl_fpo_vlasov_coeff_recovery_cu_dev_new(const struct gkyl_rect_grid *grid, 
    const struct gkyl_basis *phase_basis, const struct gkyl_range *phase_range, bool use_gpu);

/**
 * Delete updater.
 *
*/
void gkyl_fpo_vlasov_coeff_recovery_release(struct gkyl_fpo_vlasov_coeff_recovery *up);
