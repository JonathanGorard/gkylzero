#pragma once

#include <gkyl_block_geom.h>
#include <gkyl_gyrokinetic.h>

typedef struct gkyl_gyrokinetic_multib_app gkyl_gyrokinetic_multib_app;

// Species input per block
struct gkyl_gyrokinetic_multib_species_pb {
  int block_id; // block ID

  // Initial conditions using projection routine.
  struct gkyl_gyrokinetic_projection projection;

  double polarization_density;

  // Source to include.
  struct gkyl_gyrokinetic_source source;
};

// Species input
struct gkyl_gyrokinetic_multib_species {
  char name[128]; // Species name.

  enum gkyl_gkmodel_id gkmodel_id;
  double charge, mass; // Charge and mass.

  double lower[3], upper[3]; // Lower, upper bounds of velocity-space.
  int cells[3]; // Velocity-space cells.
  struct gkyl_mapc2p_inp mapc2p; // velocity mapping


  bool no_by; // Boolean for whether we are using specialized GK kernels with no b_y.
              // These more computationally efficient kernels are for slab or mirror 
              // calculations where there is no toroidal field. 

  int num_diag_moments; // number of diagnostic moments
  char diag_moments[24][24]; // list of diagnostic moments

  // Collisions to include.
  struct gkyl_gyrokinetic_collisions collisions;

  // Diffusion coupling to include.
  struct gkyl_gyrokinetic_diffusion diffusion;

  // Radiation to include.
  struct gkyl_gyrokinetic_radiation radiation;

  // Reactions between plasma species to include.
  struct gkyl_gyrokinetic_react react;
  // Reactions with neutral species to include.
  struct gkyl_gyrokinetic_react react_neut;


  bool duplicate_across_blocks; // set to true if all blocks are identical  
  // species inputs per-block: only one is needed if duplicate_across_blocks = true
  const struct gkyl_gyrokinetic_multib_species_pb *blocks;

  // Physical boundary conditions
  int num_physical_bcs;
  const struct gkyl_block_physical_bcs *bcs;
};

// Neutral species input per block
struct gkyl_gyrokinetic_multib_neut_species_pb {
  int block_id; // block ID

  // Initial conditions using projection routine.
  struct gkyl_gyrokinetic_projection projection;

  // Source to include.
  struct gkyl_gyrokinetic_source source;
};

// Neutral species input
struct gkyl_gyrokinetic_multib_neut_species {
  char name[128]; // Species name.

  double mass; // Mass.
  double lower[3], upper[3]; // Lower, upper bounds of velocity-space.
  int cells[3]; // Velocity-space cells.

  struct gkyl_mapc2p_inp mapc2p;

  bool is_static; // Set to true if neutral species does not change in time.

  int num_diag_moments; // Number of diagnostic moments.
  char diag_moments[16][16]; // List of diagnostic moments.

  // Reactions with plasma species to include.
  struct gkyl_gyrokinetic_react react_neut;

  bool duplicate_across_blocks; // set to true if all blocks are identical  
  // species inputs per-block: only one is needed if duplicate_across_blocks = true
  const struct gkyl_gyrokinetic_multib_neut_species_pb *blocks;

  // Physical boundary conditions
  int num_physical_bcs;
  const struct gkyl_block_physical_bcs *bcs;
};

// Field input per block 
struct gkyl_gyrokinetic_multib_field_pb {
  int block_id; // block ID

  double polarization_bmag; 

  void *phi_wall_lo_ctx; // context for biased wall potential on lower wall
  // pointer to biased wall potential on lower wall function
  void (*phi_wall_lo)(double t, const double *xn, double *phi_wall_lo_out, void *ctx);
  bool phi_wall_lo_evolve; // set to true if biased wall potential on lower wall function is time dependent  

  void *phi_wall_up_ctx; // context for biased wall potential on upper wall
  // pointer to biased wall potential on upper wall function
  void (*phi_wall_up)(double t, const double *xn, double *phi_wall_up_out, void *ctx);
  bool phi_wall_up_evolve; // set to true if biased wall potential on upper wall function is time dependent  
};

// Field input 
struct gkyl_gyrokinetic_multib_field {
  enum gkyl_gkfield_id gkfield_id;
  double kperpSq; // kperp^2 parameter for 1D field equations
  double xLCFS; // radial location of the LCFS.

  // parameters for adiabatic electrons simulations
  double electron_mass, electron_charge, electron_density, electron_temp;

  //enum gkyl_fem_parproj_bc_type fem_parbc;
  //struct gkyl_poisson_bc poisson_bcs;

  bool duplicate_across_blocks; // set to true if all blocks are identical  
  // field inputs per-block: only one is needed if duplicate_across_blocks = true
  const struct gkyl_gyrokinetic_multib_field_pb *blocks;

  // Physical boundary conditions
  int num_physical_bcs;
  const struct gkyl_block_physical_bcs *bcs;
};

// Top-level app parameters: this
struct gkyl_gyrokinetic_multib {
  char name[128]; // name of app

  int cdim, vdim; // conf, velocity space dimensions
  int poly_order; // polynomial order
  enum gkyl_basis_type basis_type; // type of basis functions to use
  bool use_gpu; // Flag to indicate if solver should use GPUs

  // geometry and topology of all blocks in simulation
  struct gkyl_block_geom *block_geom;

  double cfl_frac; // CFL fraction to use (default 1.0)

  bool enforce_positivity; // Positivity enforcement via shift in f.

  int num_species; // number of species
  // species inputs
  struct gkyl_gyrokinetic_multib_species species[GKYL_MAX_SPECIES];

  int num_neut_species; // number of neutral species  
  // neutral species inputs
  struct gkyl_gyrokinetic_multib_neut_species neut_species[GKYL_MAX_SPECIES];

  bool skip_field; // Skip field update -> phi = 0 for all time  
  // field inputs
  struct gkyl_gyrokinetic_multib_field field;

 // communicator to used  
  struct gkyl_comm *comm;  
};


/**
 * Construct a new gk multi-block app.
 *
 * @param mbinp App inputs. See struct docs. All struct params MUST be
 *     initialized
 * @return New multi-block gk app object.
 */
gkyl_gyrokinetic_multib_app* gkyl_gyrokinetic_multib_app_new(const struct gkyl_gyrokinetic_multib *mbinp);

/**
 * Initialize species and field by projecting initial conditions on
 * basis functions.
 *
 * @param app App object.
 * @param t0 Time for initial conditions.
 */
void gkyl_gyrokinetic_multib_app_apply_ic(gkyl_gyrokinetic_multib_app* app, double t0);

/**
 * Initialize species by projecting initial conditions on basis
 * functions. Species index (sidx) is the same index used to specify
 * the species in the gkyl_gk object used to construct app.
 *
 * @param app App object.
 * @param sidx Index of species to initialize.
 * @param t0 Time for initial conditions
 */
void gkyl_gyrokinetic_multib_app_apply_ic_species(gkyl_gyrokinetic_multib_app* app, int sidx, double t0);

/**
 * Initialize neutral species by projecting initial conditions on basis
 * functions. Neutral species index (sidx) is the same index used to specify
 * the neutral species in the gkyl_gk object used to construct app.
 *
 * @param app App object.
 * @param sidx Index of neutral species to initialize.
 * @param t0 Time for initial conditions
 */
void gkyl_gyrokinetic_multib_app_apply_ic_neut_species(gkyl_gyrokinetic_multib_app* app, int sidx, double t0);


/**
 * Initialize field from file
 *
 * @param app App object
 * @param fname file to read
 */
struct gkyl_app_restart_status
gkyl_gyrokinetic_multib_app_from_file_field(gkyl_gyrokinetic_multib_app *app, const char *fname);

/**
 * Initialize gyrokinetic species from file
 *
 * @param app App object
 * @param sidx gk species index
 * @param fname file to read
 */
struct gkyl_app_restart_status 
gkyl_gyrokinetic_multib_app_from_file_species(gkyl_gyrokinetic_multib_app *app, int sidx,
  const char *fname);

/**
 * Initialize neutral species from file
 *
 * @param app App object
 * @param sidx neut species index
 * @param fname file to read
 */
struct gkyl_app_restart_status 
gkyl_gyrokinetic_multib_app_from_file_neut_species(gkyl_gyrokinetic_multib_app *app, int sidx,
  const char *fname);

/**
 * Initialize the gyrokinetic app from a specific frame.
 *
 * @param app App object
 * @param frame frame to read
 */
struct gkyl_app_restart_status
gkyl_gyrokinetic_multib_app_read_from_frame(gkyl_gyrokinetic_multib_app *app, int frame);

/**
 * Initialize field from frame
 *
 * @param app App object
 * @param frame frame to read
 */
struct gkyl_app_restart_status
gkyl_gyrokinetic_multib_app_from_frame_field(gkyl_gyrokinetic_multib_app *app, int frame);

/**
 * Initialize gyrokinetic species from file
 *
 * @param app App object
 * @param sidx gk species index
 * @param frame frame to read
 */
struct gkyl_app_restart_status
gkyl_gyrokinetic_multib_app_from_frame_species(gkyl_gyrokinetic_multib_app *app, int sidx, int frame);

/**
 * Initialize neutral species from file
 *
 * @param app App object
 * @param sidx neut species index
 * @param frame frame to read
 */
struct gkyl_app_restart_status
gkyl_gyrokinetic_multib_app_from_frame_neut_species(gkyl_gyrokinetic_multib_app *app, int sidx, int frame);

/**
 * Write output to console: this is mainly for diagnostic messages the
 * driver code wants to write to console. It accounts for parallel
 * output by not messing up the console with messages from each rank.
 *
 * @param app App object
 * @param fp File pointer for open file for output
 * @param fmt Format string for console output
 * @param argp Objects to write
 */
void gkyl_gyrokinetic_multib_app_cout(const gkyl_gyrokinetic_multib_app* app, FILE *fp, const char *fmt, ...);

/**
 * Write block topology to file.
 * 
 * @param app App object.
 */
void gkyl_gyrokinetic_multib_app_write_topo(const gkyl_gyrokinetic_multib_app* app);

/**
 * Calculate integrated diagnostic moments for plasma species (including sources).
 *
 * @param tm Time at which integrated diagnostics are to be computed
 * @param app App object.
 */
void gkyl_gyrokinetic_multib_app_calc_integrated_mom(gkyl_gyrokinetic_multib_app* app, double tm);

/**
 * Calculate integrated field energy
 *
 * @param tm Time at which integrated diagnostic are to be computed
 * @param app App object.
 */
void gkyl_gyrokinetic_multib_app_calc_field_energy(gkyl_gyrokinetic_multib_app* app, double tm);

/**
 * Write field and species data to file.
 * 
 * @param app App object.
 * @param tm Time-stamp
 * @param frame Frame number
 */
void gkyl_gyrokinetic_multib_app_write(const gkyl_gyrokinetic_multib_app* app, double tm, int frame);

/**
 * Write field data to file.
 * 
 * @param app App object.
 * @param tm Time-stamp
 * @param frame Frame number
 */
void gkyl_gyrokinetic_multib_app_write_field(const gkyl_gyrokinetic_multib_app* app, double tm, int frame);

/**
 * Write species data to file.
 * 
 * @param app App object.
 * @param sidx Index of species to initialize.
 * @param tm Time-stamp
 * @param frame Frame number
 */
void gkyl_gyrokinetic_multib_app_write_species(const gkyl_gyrokinetic_multib_app* app, int sidx, double tm, int frame);


/**
 * Write neutral species data to file.
 * 
 * @param app App object.
 * @param sidx Index of species to initialize.
 * @param tm Time-stamp
 * @param frame Frame number
 */
void gkyl_gyrokinetic_multib_app_write_neut_species(const gkyl_gyrokinetic_multib_app* app, int sidx, double tm, int frame);

/**
 * Write source species data to file.
 * 
 * @param app App object.
 * @param sidx Index of species to initialize.
 * @param tm Time-stamp
 * @param frame Frame number
 */
void gkyl_gyrokinetic_multib_app_write_source_species(gkyl_gyrokinetic_multib_app* app, int sidx, double tm, int frame);

/**
 * Write source neutral species data to file.
 * 
 * @param app App object.
 * @param sidx Index of species to initialize.
 * @param tm Time-stamp
 * @param frame Frame number
 */
void gkyl_gyrokinetic_multib_app_write_source_neut_species(gkyl_gyrokinetic_multib_app* app, int sidx, double tm, int frame);

/**
 * Write collisional moments for species to file.
 * 
 * @param app App object.
 * @param sidx Index of species to initialize.
 * @param tm Time-stamp
 * @param frame Frame number
 */
void gkyl_gyrokinetic_multib_app_write_coll_mom(gkyl_gyrokinetic_multib_app *app, int sidx, double tm, int frame);

/**
 * Write radiation drag coefficients for species to file.
 * 
 * @param app App object.
 * @param sidx Index of species to initialize.
 * @param tm Time-stamp
 * @param frame Frame number
 */
void gkyl_gyrokinetic_multib_app_write_rad_drag(gkyl_gyrokinetic_multib_app *app, int sidx, double tm, int frame);

/**
 * Write radiation emissivity of each species that species sidx collides with
 * 
 * @param app App object.
 * @param sidx Index of species to initialize.
 * @param tm Time-stamp
 * @param frame Frame number
 */
void gkyl_gyrokinetic_multib_app_write_rad_emissivity(gkyl_gyrokinetic_multib_app *app, int sidx, double tm, int frame);

/**
 * Write integrated moments of radiation rhs for radiating species 
 * 
 * @param app App object.
 * @param sidx Index of species from which to write radiation.
 * @param tm Time-stamp
 */
void gkyl_gyrokinetic_multib_app_write_rad_integrated_moms(gkyl_gyrokinetic_multib_app *app, int sidx, double tm);


/**
 * Write iz react rate coefficients for species to file.
 * 
 * @param app App object.
 * @param sidx Index of species to initialize.
 * @param ridx Index of reaction to initialize.
 * @param tm Time-stamp
 * @param frame Frame number
 */
void gkyl_gyrokinetic_multib_app_write_iz_react(gkyl_gyrokinetic_multib_app* app, int sidx, int ridx, double tm, int frame);

/**
 * Write recomb react rate coefficients for species to file.
 * 
 * @param app App object.
 * @param sidx Index of species to initialize.
 * @param ridx Index of reaction to initialize.
 * @param tm Time-stamp
 * @param frame Frame number
 */
void gkyl_gyrokinetic_multib_app_write_recomb_react(gkyl_gyrokinetic_multib_app* app, int sidx, int ridx, double tm, int frame);

/**
 * Write iz react rate coefficients for species to file.
 * 
 * @param app App object.
 * @param sidx Index of species to initialize.
 * @param ridx Index of reaction to initialize.
 * @param tm Time-stamp
 * @param frame Frame number
 */
void gkyl_gyrokinetic_multib_app_write_iz_react_neut(gkyl_gyrokinetic_multib_app* app, int sidx, int ridx, double tm, int frame);

/**
 * Write recomb react rate coefficients for species to file.
 * 
 * @param app App object.
 * @param sidx Index of species to initialize.
 * @param ridx Index of reaction to initialize.
 * @param tm Time-stamp
 * @param frame Frame number
 */
void gkyl_gyrokinetic_multib_app_write_recomb_react_neut(gkyl_gyrokinetic_multib_app* app, int sidx, int ridx, double tm, int frame);

/**
 * Write diagnostic moments for species to file.
 * 
 * @param app App object.
 * @param tm Time-stamp
 * @param frame Frame number
 */
void gkyl_gyrokinetic_multib_app_write_mom(gkyl_gyrokinetic_multib_app *app, double tm, int frame);

/**
 * Write diagnostic moments for species source to file.
 * 
 * @param app App object.
 * @param tm Time-stamp
 * @param frame Frame number
 */
void gkyl_gyrokinetic_multib_app_write_source_mom(gkyl_gyrokinetic_multib_app *app, double tm, int frame);

/**
 * Write integrated diagnostic moments for species to file. Integrated
 * moments are appended to the same file.
 * 
 * @param app App object.
 */
void gkyl_gyrokinetic_multib_app_write_integrated_mom(gkyl_gyrokinetic_multib_app *app);

/**
 * Write integrated diagnostic moments for sources to file. Integrated
 * moments are appended to the same file.
 * 
 * @param app App object.
 */
void gkyl_gyrokinetic_multib_app_write_integrated_source_mom(gkyl_gyrokinetic_multib_app *app);

/**
 * Write field energy to file. Field energy data is appended to the
 * same file.
 * 
 * @param app App object.
 */
void gkyl_gyrokinetic_multib_app_write_field_energy(gkyl_gyrokinetic_multib_app* app);

/**
 * Write integrated correct Maxwellian status of the species distribution function to file. Correct
 * Maxwellian status is appended to the same file.
 * 
 * @param app App object.
 */
void gkyl_gyrokinetic_multib_app_write_max_corr_status(gkyl_gyrokinetic_multib_app *app);

/**
 * Write geometry file.
 *
 * @param app App object.
 */
void gkyl_gyrokinetic_multib_app_write_geometry(gkyl_gyrokinetic_multib_app *app);

/**
 * Read geometry file.
 *
 * @param app App object.
 */
void gkyl_gyrokinetic_multib_app_read_geometry(gkyl_gyrokinetic_multib_app *app);

/**
 * Write stats to file. Data is written in json format.
 *
 * @param app App object.
 */
void gkyl_gyrokinetic_multib_app_stat_write(gkyl_gyrokinetic_multib_app* app);

/**
 * Write output to console: this is mainly for diagnostic messages the
 * driver code wants to write to console. It accounts for parallel
 * output by not messing up the console with messages from each rank.
 *
 * @param app App object
 * @param fp File pointer for open file for output
 * @param fmt Format string for console output
 * @param argp Objects to write
 */
void gkyl_gyrokinetic_multib_app_cout(const gkyl_gyrokinetic_multib_app* app, FILE *fp, const char *fmt, ...);

/**
 * Advance simulation by a suggested time-step 'dt'. The dt may be too
 * large in which case method will attempt to take a smaller time-step
 * and also return it as the 'dt_actual' field of the status
 * object. If the suggested time-step 'dt' is smaller than the largest
 * stable time-step the method will use the smaller value instead,
 * returning the larger time-step in the 'dt_suggested' field of the
 * status object. If the method fails to find any stable time-step
 * then the 'success' flag will be set to 0. At that point the calling
 * code must abort the simulation as this signals a catastrophic
 * failure and the simulation can't be safely continued.
 * 
 * @param app App object.
 * @param dt Suggested time-step to advance simulation
 * @return Status of update.
 */
struct gkyl_update_status gkyl_gyrokinetic_multib_update(gkyl_gyrokinetic_multib_app* app, double dt);

/**
 * Return simulation statistics.
 * 
 * @return Return statistics object.
 */
struct gkyl_gyrokinetic_stat gkyl_gyrokinetic_multib_app_stat(gkyl_gyrokinetic_multib_app* app);

/**
 * Run the RHS for the species update. This is used to compute kernel
 * timers and is not otherwise a useful function for a full
 * simulation.
 *
 * @param app App object.
 * @param update_vol_term Set to 1 to update vol term also, 0 otherwise
 */
void gkyl_gyrokinetic_multib_app_species_ktm_rhs(gkyl_gyrokinetic_multib_app* app, int update_vol_term);

/**
 * Free gk app.
 *
 * @param app App to release.
 */
void gkyl_gyrokinetic_multib_app_release(gkyl_gyrokinetic_multib_app* app);
