#pragma once

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <gkyl_alloc.h>
#include <gkyl_array.h>
#include <gkyl_array_ops.h>
#include <gkyl_array_rio.h>
#include <gkyl_block_topo.h>
#include <gkyl_fv_proj.h>
#include <gkyl_moment.h>
#include <gkyl_moment_em_coupling.h>
#include <gkyl_null_pool.h>
#include <gkyl_range.h>
#include <gkyl_rect_decomp.h>
#include <gkyl_rect_grid.h>
#include <gkyl_thread_pool.h>
#include <gkyl_util.h>
#include <gkyl_wave_geom.h>
#include <gkyl_wave_prop.h>
#include <gkyl_wv_apply_bc.h>
#include <gkyl_wv_euler.h>
#include <gkyl_wv_maxwell.h>
#include <rt_arg_parse.h>

#include <thpool.h>

//#define AMR_DEBUG
//#define AMR_USETHREADS

// Definitions of private structs and APIs attached to these objects, for use in the block AMR subsystem.

// Ranges indicating the extents of the skin and ghost regions, for the purpose of applying inter-block boundary conditions.
struct skin_ghost_ranges {
  struct gkyl_range lower_skin[2];
  struct gkyl_range lower_ghost[2];
  
  struct gkyl_range upper_skin[2];
  struct gkyl_range upper_ghost[2];
};

// Block-structured data (including grid, range and boundary condition information) for the Euler equations.
struct euler_block_data {
  struct gkyl_rect_grid grid;
  gkyl_fv_proj *fv_proj;

  struct gkyl_range ext_range;
  struct gkyl_range range;
  struct gkyl_array *fdup;
  struct gkyl_array *f[9];

  struct gkyl_wave_geom *geom;

  struct gkyl_wv_eqn *euler;
  gkyl_wave_prop *slvr[2];

  struct skin_ghost_ranges skin_ghost;
  struct gkyl_array *bc_buffer;

  struct gkyl_wv_apply_bc *lower_bc[2];
  struct gkyl_wv_apply_bc *upper_bc[2];
};

// Job pool information context for updating block-structured data for the Euler equations using threads.
struct euler_update_block_ctx {
  const struct euler_block_data *bdata;
  int bidx;
  int dir;
  double t_curr;
  double dt;
  struct gkyl_wave_prop_status stat;
};

// Context for copying job pool information for the block-structured Euler equations.
struct euler_copy_job_ctx {
  int bidx;
  const struct gkyl_array *inp;
  struct gkyl_array *out;
};

// Simulation statistics (allowing for tracking of the number of failed time-steps).
struct sim_stats {
  int nfail;
};

/**
* Initialize the ranges indicating the extents of the skin and ghost regions, for the purpose of applying inter-block boundary conditions.
*
* @param sgr Ranges for the skin and ghost regions.
* @param parent Ranges for the parent regions (of which the skin and ghost regions are subregions).
* @param ghost Number of ghost (and therefore skin) cells.
*/
void skin_ghost_ranges_init(struct skin_ghost_ranges* sgr, const struct gkyl_range* parent, const int* ghost);

/**
* Boundary condition function for applying transmissive boundary conditions for the Euler equations.
*
* @param t Current simulation time.
* @param nc Number of boundary cells to which to apply transmissive boundary conditions.
* @param skin Skin cells in boundary region (from which values are copied).
* @param ghost Ghost cells in boundary region (to which values are copied).
* @param ctx Context to pass to the function.
*/
static void euler_transmissive_bc(double t, int nc, const double* GKYL_RESTRICT skin, double* GKYL_RESTRICT ghost, void* ctx);

/**
* Initialize updaters for both physical (outer-block) and non-physical (inter-block) boundary conditions for the Euler equations.
*
* @param bdata Block-structured data for the Euler equations.
* @param conn Topology/connectivity data for the block hierarchy.
*/
void euler_block_bc_updaters_init(struct euler_block_data* bdata, const struct gkyl_block_connections* conn);

/**
* Release updaters for both physical (outer-block) and non-physical (inter-block) boundary conditions for the Euler equations.
*
* @param bdata Block-structured data for the Euler equations.
*/
void euler_block_bc_updaters_release(struct euler_block_data* bdata);

/**
* Apply both physical (outer-block) and non-physical (inter-block) boundary conditions for the Euler equations.
*
* @param bdata Block-structured data for the Euler equations.
* @param tm Simulation time at which the boundary conditions are applied.
* @param fld Output array.
*/
void euler_block_bc_updaters_apply(const struct euler_block_data* bdata, double tm, struct gkyl_array* fld);

/**
* Synchronize all blocks in the block hierarchy by applying all appropriate physical (outer-block) and non-physical (inter-block) boundary conditions for the Euler equations.
*
* @param btopo Topology/connectivity information for the entire block hierarchy.
* @param bdata Block-structured data for the Euler equations.
* @param fld Output array.
*/
void euler_sync_blocks(const struct gkyl_block_topo* btopo, const struct euler_block_data bdata[], struct gkyl_array* fld[]);

/**
* Write block-structured simulation data for the Euler equations onto disk.
*
* @param file_nm File name schema to use for the simulation output.
* @param bdata Block-structured data for the Euler equations.
*/
void euler_block_data_write(const char* file_nm, const struct euler_block_data* bdata);

/**
* Calculate the maximum stable time-step for the block-structured Euler equations.
*
* @param bdata Block-structured data for the Euler equations.
*/
double euler_block_data_max_dt(const struct euler_block_data* bdata);

/**
* Update the block-structured simulation data for the Euler equations using the thread-based job pool.
*
* @param ctx Context to pass to the function.
*/
void euler_update_block_job_func(void* ctx);

/**
* Update all blocks in the block hierarchy by using the thread-based job pool for the Euler equations.
*
* @param job_pool Job pool for updating block-structured data for the Euler equations using threads.
* @param btopo Topology/connectivity information for the entire block hierarchy.
* @param bdata Block-structured data for the Euler equations.
* @param t_curr Current simulation time.
* @param dt Current stable time-step for the simulation.
*/
struct gkyl_update_status euler_update_all_blocks(const struct gkyl_job_pool* job_pool, const struct gkyl_block_topo* btopo,
  const struct euler_block_data bdata[], double t_curr, double dt);

/**
* Initialize a new job in the thread-based pool for updating the block-structured simulation data for the Euler equations.
*
* @param ctx Context to pass to the function.
*/
void euler_init_job_func(void* ctx);

/**
* Copy an existing job between two thread-based based job pools for updating the block-structured simulation data for the Euler equations.
*
* @param ctx Context to pass to the function.
*/
void euler_copy_job_func(void* ctx);

/**
* Take a single time-step across the entire block hierarchy for the Euler equations.
*
* @param job_pool Job pool for updating block-structured data for the Euler equations using threads.
* @param btopo Topology/connectivity information for the entire block hierarchy.
* @param bdata Block-structured data for the Euler equations.
* @param t_curr Current simulation time.
* @param dt0 Initial guess for the maximum stable time-step.
* @param stats Simulation statistics (allowing for tracking of the number of failed time-steps).
*/
struct gkyl_update_status euler_update(const struct gkyl_job_pool* job_pool, const struct gkyl_block_topo* btopo,
  const struct euler_block_data bdata[], double t_curr, double dt0, struct sim_stats* stats);

/**
* Write the complete simulation output for the entire block hierarchy for the Euler equations onto disk.
*
* @param fbase Base file name schema to use for the simulation output.
* @param num_blocks Number of blocks in the block hierarchy.
* @param bdata Array of block-structured data for the Euler equations.
*/
void euler_write_sol(const char* fbase, int num_blocks, const struct euler_block_data bdata[]);

/**
* Calculate the maximum stable time-step across all blocks in the block hierarchy for the Euler equations.
*
* @param num_blocks Number of blocks in the block hierarchy.
* @param bdata Array of block-structured data for the Euler equations.
*/
double euler_max_dt(int num_blocks, const struct euler_block_data bdata[]);

/**
* Set up the topology/connectivity information for the block hierarchy for a mesh containing a single refinement patch.
*/
struct gkyl_block_topo* create_block_topo();