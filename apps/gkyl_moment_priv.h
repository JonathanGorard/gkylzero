// Private header for use in moment app: do not include in user-facing
// header files!
#pragma once

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include <stc/cstr.h>

#include <gkyl_alloc.h>
#include <gkyl_app_priv.h>
#include <gkyl_array.h>
#include <gkyl_array_ops.h>
#include <gkyl_array_rio.h>
#include <gkyl_dflt.h>
#include <gkyl_dynvec.h>
#include <gkyl_elem_type.h>
#include <gkyl_eval_on_nodes.h>
#include <gkyl_fv_proj.h>
#include <gkyl_moment.h>
#include <gkyl_moment_em_coupling.h>
#include <gkyl_mp_scheme.h>
#include <gkyl_range.h>
#include <gkyl_rect_decomp.h>
#include <gkyl_rect_grid.h>
#include <gkyl_util.h>
#include <gkyl_wave_geom.h>
#include <gkyl_wave_prop.h>
#include <gkyl_wv_apply_bc.h>
#include <gkyl_wv_maxwell.h>
#include <gkyl_wv_ten_moment.h>

// Species data
struct moment_species {
  int ndim;
  char name[128]; // species name
  double charge, mass;
  double k0; // closure parameter (default is 0.0, used by 10 moment)

  int evolve; // evolve species? 1-yes, 0-no

  void *ctx; // context for initial condition init function
  // pointer to initialization function
  void (*init)(double t, const double *xn, double *fout, void *ctx);
    
  struct gkyl_array *app_accel; // array for applied acceleration/forces
  // pointer to projection operator for applied acceleration/forces function
  gkyl_fv_proj *proj_app_accel;
  struct gkyl_array *bc_buffer; // buffer for periodic BCs

  enum gkyl_eqn_type eqn_type; // type ID of equation
  int num_equations; // number of equations in species

  enum gkyl_moment_scheme scheme_type; // scheme to update equations
  // solvers and data to update fluid equations
  union {
    struct {
      gkyl_wave_prop *slvr[3]; // wave-prop solver in each direction
      struct gkyl_array *fdup, *f[4]; // arrays for updates
    };
    struct {
      gkyl_mp_scheme *mp_slvr; // monotonicity-preserving scheme
      struct gkyl_array *f0, *f1, *fnew; // arrays for updates
    };
  };

  // boundary condition type
  enum gkyl_species_bc_type lower_bct[3], upper_bct[3];
  // boundary condition solvers on lower/upper edges in each direction
  gkyl_wv_apply_bc *lower_bc[3], *upper_bc[3];

  gkyl_dynvec integ_q; // integrated conserved quantities
  bool is_first_q_write_call; // flag for dynvec written first time  
};

// Field data
struct moment_field {
  int ndim;
  double epsilon0, mu0;

  int evolve; // evolve species? 1-yes, 0-no
    
  void *ctx; // context for initial condition init function
  // pointer to initialization function
  void (*init)(double t, const double *xn, double *fout, void *ctx);    
    
  struct gkyl_array *app_current; // arrays for applied currents
  // pointer to projection operator for applied current function
  gkyl_fv_proj *proj_app_current;

  bool is_ext_em_static; // flag to indicate if external field is time-independent
  struct gkyl_array *ext_em; // array external fields  
  gkyl_fv_proj *proj_ext_em;   // pointer to projection operator for external fields
  bool was_ext_em_computed; // flag to indicate if we already computed external EM field
  
  struct gkyl_array *bc_buffer; // buffer for periodic BCs

  enum gkyl_moment_scheme scheme_type; // scheme to update equations  
  // solvers and data to update fluid equations
  union {
    struct {
      gkyl_wave_prop *slvr[3]; // wave-prop solver in each direction
      struct gkyl_array *fdup, *f[4]; // arrays for updates
    };
    struct {
      gkyl_mp_scheme *mp_slvr; // monotonicity-preserving scheme
      struct gkyl_array *f0, *f1, *fnew; // arrays for updates
    };
  };

  // boundary condition type
  enum gkyl_field_bc_type lower_bct[3], upper_bct[3];
  // boundary conditions on lower/upper edges in each direction
  gkyl_wv_apply_bc *lower_bc[3], *upper_bc[3];

  gkyl_dynvec integ_energy; // integrated energy components
  bool is_first_energy_write_call; // flag for dynvec written first time
};

// Source data
struct moment_coupling {
  gkyl_moment_em_coupling *slvr; // source solver function
};

// Moment app object: used as opaque pointer in user code
struct gkyl_moment_app {
  char name[128]; // name of app
  int ndim; // space dimensions
  double tcurr; // current time
  double cfl; // CFL number

  int num_periodic_dir; // number of periodic directions
  int periodic_dirs[3]; // list of periodic directions

  int is_dir_skipped[3]; // flags to tell if update in direction are skipped

  struct gkyl_rect_grid grid; // grid
  struct gkyl_range local, local_ext; // local, local-ext ranges

  bool has_mapc2p; // flag to indicate if we have mapc2p
  void *c2p_ctx; // context for mapc2p function
  // pointer to mapc2p function
  void (*mapc2p)(double t, const double *xc, double *xp, void *ctx);

  struct gkyl_wave_geom *geom; // geometry needed for species and field solvers

  struct app_skin_ghost_ranges skin_ghost; // conf-space skin/ghost

  int has_field; // flag to indicate if we have a field
  struct moment_field field; // field data
    
  // species data
  int num_species;
  struct moment_species *species; // species data

  int update_sources; // flag to indicate if sources are to be updated
  struct moment_coupling sources; // sources
    
  struct gkyl_moment_stat stat; // statistics
};

// Function pointer to compute integrated quantities from input
typedef void (*integ_func)(int nc, const double *qin, double *integ_out);

/** Some common functions to species and fields */

// functions for use in integrated quantities calculation
static inline void
integ_unit(int nc, const double *qin, double *integ_out)
{
  for (int i=0; i<nc; ++i) integ_out[i] = qin[i];
}
static inline void
integ_sq(int nc, const double *qin, double *integ_out)
{
  for (int i=0; i<nc; ++i) integ_out[i] = qin[i]*qin[i];
}

// function for copy BC
static inline void
bc_copy(double t, int nc, const double *skin, double * GKYL_RESTRICT ghost, void *ctx)
{
  for (int c=0; c<nc; ++c) ghost[c] = skin[c];
}

// Compute integrated quantities specified by i_func 
void calc_integ_quant(int nc, double vol,
  const struct gkyl_array *q, const struct gkyl_wave_geom *geom,
  struct gkyl_range update_rng, integ_func i_func, double *integ_q);

// Check array "q" for nans
bool check_for_nans(const struct gkyl_array *q, struct gkyl_range update_rng);

// Apply periodic BCs to array "f" in direction "dir"
void moment_apply_periodic_bc(const gkyl_moment_app *app, struct gkyl_array *bc_buffer,
  int dir, struct gkyl_array *f);

// Apply wedge-periodic BCs to array "f"
void moment_apply_wedge_bc(const gkyl_moment_app *app, double tcurr,
  const struct gkyl_range *update_rng, struct gkyl_array *bc_buffer,
  int dir, const struct gkyl_wv_apply_bc *lo, const struct gkyl_wv_apply_bc *up,
  struct gkyl_array *f);

/** moment_species API */

// Initialize the moment species object
void moment_species_init(const struct gkyl_moment *mom,
  const struct gkyl_moment_species *mom_sp, struct gkyl_moment_app *app, struct moment_species *sp);

// Apply BCs to species data "f"
void moment_species_apply_bc(const gkyl_moment_app *app, double tcurr,
  const struct moment_species *sp, struct gkyl_array *f);

// Maximum stable time-step from species
double moment_species_max_dt(const gkyl_moment_app *app, const struct moment_species *sp);

// Advance solution of species by time-step dt to tcurr+dt
struct gkyl_update_status moment_species_update(const gkyl_moment_app *app,
  const struct moment_species *sp, double tcurr, double dt);

// Free memory allocated by species
void moment_species_release(const struct moment_species *sp);

/** moment_field API */

// Initialize EM field
void moment_field_init(const struct gkyl_moment *mom, const struct gkyl_moment_field *mom_fld,
  struct gkyl_moment_app *app, struct moment_field *fld);

// Apply BCs to EM field
void moment_field_apply_bc(const gkyl_moment_app *app, double tcurr,
  const struct moment_field *field, struct gkyl_array *f);


// Maximum stable time-step due to EM fields
double moment_field_max_dt(const gkyl_moment_app *app, const struct moment_field *fld);

// Update EM field from tcurr to tcurr+dt
struct gkyl_update_status moment_field_update(const gkyl_moment_app *app,
  const struct moment_field *fld, double tcurr, double dt);

// Release the EM field object
void moment_field_release(const struct moment_field *fld);

/** moment_coupling API */

// initialize source solver: this should be called after all species
// and fields are initialized
void moment_coupling_init(const struct gkyl_moment_app *app,
  struct moment_coupling *src);

// update sources: 'nstrang' is 0 for the first Strang step and 1 for
// the second step
void moment_coupling_update(gkyl_moment_app *app, struct moment_coupling *src,
  int nstrang, double tcurr, double dt);

// Release coupling sources
void moment_coupling_release(const struct moment_coupling *src);