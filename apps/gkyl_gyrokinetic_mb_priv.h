// Private header file for multi-block gyrokinetic solver.
#pragma once

#include <gkyl_gyrokinetic_priv.h>
#include <gkyl_gyrokinetic_mb.h>

// Top-level multiblock gyrokinetic App object.
struct gkyl_gyrokinetic_mb_app {
  char name[128]; // Name of app.
  struct gkyl_job_pool *job_pool; // Job pool.

  int cdim, vdim; // Conf, velocity space dimensions.
  int poly_order; // Polynomial order.
  double tcurr; // Current time.
  double cfl; // CFL number.

  bool use_gpu; // Should we use GPU (if present).

  struct gkyl_block_topo *btopo; // Block topology.

  int num_blocks, num_blocks_local; // Total and local number of blocks.
  int *block_idxs; // List of blocks handled on this rank.
  struct gkyl_gyrokinetic_app **blocks; // Pointers to blocks on this rank.

  struct gkyl_gyrokinetic_stat stat; // statistics
};