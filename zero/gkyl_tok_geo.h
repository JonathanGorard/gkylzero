#pragma once

#include <stdbool.h>

#include <gkyl_math.h>
#include <gkyl_range.h>
#include <gkyl_rect_grid.h>
#include <math.h>
#include <string.h>
#include <gkyl_evalf_def.h>
#include <gkyl_gk_geometry.h>

// Object type
typedef struct gkyl_tok_geo gkyl_tok_geo;


// Some cumulative statistics
struct gkyl_tok_geo_stat {
  long nquad_cont_calls; // num calls from quadrature
  long nroot_cont_calls; // num calls from root-finder
};  

typedef   void (*plate_func)(double s, double* RZ);

struct gkyl_tok_geo {
  struct gkyl_efit* efit;

  struct gkyl_rect_grid rzgrid; // RZ grid on which psi(R,Z) is defined
  struct gkyl_range rzlocal; // local range over which psiRZ is defined
  struct gkyl_range rzlocal_ext; // extended range
  struct gkyl_basis rzbasis; // basis functions for R,Z grid
  int num_rzbasis; // number of basis functions in RZ
  const struct gkyl_array *psiRZ; // psi(R,Z) DG representation
  const struct gkyl_array *psibyrRZ; // psi(R,Z)/R DG representation
  const struct gkyl_array *psibyr2RZ; // psi(R,Z)/R^2 DG representation
                   
  struct gkyl_rect_grid fgrid; // flux grid for fpol
  struct gkyl_range frange; // flux range
  struct gkyl_range frange_ext; // extended range
  struct gkyl_basis fbasis; // psi basis for fpol
  const struct gkyl_array *fpoldg; // fpol(psi) dg rep
  const struct gkyl_array *qdg; // q(psi) dg rep
                                   
  double psisep; // psi of separatrix
  double zmaxis; // z of magnetic axis
  double rleft, rright;
  double rmin, rmax;

  // Flag and functions to specify the plate location/shape in RZ coordinates
  // The functions should specify R(s) and Z(s) on the plate where s is a parameter \in [0,1]
  // For single null, the "lower" plate is the outboard plate and the "upper plate" is the inboard plate
  bool plate_spec;
  plate_func plate_func_lower;
  plate_func plate_func_upper;

  struct { int max_iter; double eps; } root_param;
  struct { int max_level; double eps; } quad_param;

  // pointer to root finder (depends on polyorder)
  struct RdRdZ_sol (*calc_roots)(const double *psi, double psi0, double Z,
    double xc[2], double dx[2], struct gkyl_basis basis);

  struct gkyl_tok_geo_stat stat; 
  struct gkyl_array* mc2p_nodal_fd;
  struct gkyl_range* nrange;
  double* dzc;
};



// Type of flux surface
enum gkyl_tok_geo_type {
  // Full blocks to be used as stand alone simulations
  GKYL_SOL_DN_OUT, // Full Outboard SOL of double-null configuration
  GKYL_SOL_DN_IN, // Full Inboard SOL of double-null configuration
  GKYL_SOL_SN_LO, // Full SOL of a lower single-null configuration
  GKYL_SOL_SN_UP, // Full SOL of an upper single-null configuration -- not yet implemented
  GKYL_CORE, // Full core
  //GKYL_PF_LO, // Full lower PF
  //GKYL_PF_UP, // Full upper PF

  // 12 Blocks to be used for multiblock double null whold device simulations
  GKYL_SOL_DN_OUT_LO,  // Section of outboard SOL below lower xpt
  GKYL_SOL_DN_OUT_MID, // Section of outboard SOL between xpts
  GKYL_SOL_DN_OUT_UP,  // Section of outboard SOL above upper xpt
  GKYL_SOL_DN_IN_LO,   // Section of inboard SOL below lower xpt
  GKYL_SOL_DN_IN_MID,  // Section of inboard SOL between xpts
  GKYL_SOL_DN_IN_UP,   // Section of inboard SOL above upper xpt 

  GKYL_PF_UP_L, // Left half of Private flux region at top (inboard upper plate to upper xpt)
  GKYL_PF_UP_R, // Right half of Private flux region at top (upper xpt to outboard upper plate)
  GKYL_PF_LO_L, // Left half of Private flux region at bottom (lower xpt to inboard lower plate)
  GKYL_PF_LO_R, // Right half of Private flux region at bottom (outboard lower plate to lower xpt)

  GKYL_CORE_L, // Left half of core (lower to upper xpt)
  GKYL_CORE_R // Right half of core (upper to lower xpt)
};  

// Inputs to create a new GK geometry creation object
struct gkyl_tok_geo_efit_inp {
  // Inputs to get psiRZ and related inputs from efit
  char* filepath;
  int rzpoly_order;
  int fluxpoly_order;
  // Specifications for divertor plate
  bool plate_spec;
  plate_func plate_func_lower;
  plate_func plate_func_upper;

  // Parameters for root finder: leave unset to use defaults
  struct {
    int max_iter; // typically 20
    double eps; // typically 1e-10
  } root_param;

  // Parameters for nmumerical quadrature: leave unset to use default
  struct {
    int max_levels; // typically 6-7    
    double eps; // typically 1e-10
  } quad_param;
};

// Inputs to create geometry for a specific computational grid
struct gkyl_tok_geo_grid_inp {
  struct gkyl_rect_grid cgrid;
  struct gkyl_basis cbasis;
  enum gkyl_tok_geo_type ftype; // type of geometry
  
  double rclose; // closest R to discrimate
  double rleft; // closest R to discrimate
  double rright; // closest R to discrimate
  double rmin, rmax;
  double zmin, zmax; // extents of Z for integration
  double zmin_left, zmin_right; // for lower single null and PF cases diff b/t in and outboard side
  double zmax_left, zmax_right; // for upper single null and PF cases diff b/t in and outboard side

  double zxpt_lo; // z of the lower x point
  double zxpt_up; // z of the upper x point

  bool write_node_coord_array; // set to true if nodal coordinates should be written
  const char *node_file_nm; // name of nodal coordinate file
};


/**
 * Create new updater to compute the geometry (mapc2p) needed in GK
 * simulations.
 *
 * @param inp Input parameters
 * @param New GK geometry updater
 */
gkyl_tok_geo *gkyl_tok_geo_new(const struct gkyl_tok_geo_efit_inp *inp);

/**
 * Get R(psi,Z) for a specified psi and Z value. Multiple values may
 * be returned (or none). The R(psi,Z) and dR/dZ are stored in the R
 * and dR arrays which be allocated by the caller.
 *
 * @param geo Geometry object
 * @param psi Psi value
 * @param Z Z value
 * @param nmaxroots Maximum number of roots
 * @param R on output, R(psi,Z)
 * @param dR on output, dR/dZ
 */
int gkyl_tok_geo_R_psiZ(const gkyl_tok_geo *geo, double psi, double Z, int nmaxroots,
  double *R, double *dR);

/**
 * Integrate along a specified psi countour and return its length. The
 * contour must lie completely inside the RZ domain of the psiRZ DG
 * field. The @a rclose parameter is used to select amongst the
 * multiple possible countours with the same psi. Foe example, to
 * select a flux surface on the outboard side of a double-null
 * configuration choose rclose to be Rmax.
 *
 * @param geo Geometry object
 * @param psi Psi value of contour
 * @param zmin Starting z location
 * @param zmax Ending z location
 * @param rclose Value of radial coordinate to discrimate between multiple
 *    contours
 * @return Length of contour
 */
double gkyl_tok_geo_integrate_psi_contour(const gkyl_tok_geo *geo, double psi,
  double zmin, double zmax, double rclose);

/**
 * Compute physical coordinates (mapc2p)  given computational coordinates
 *
 * @param geo Geometry object
 * @param xn computational coordinates
 * @param ret physical coordinates
 */
void gkyl_tok_geo_mapc2p(const gkyl_tok_geo *geo, const struct gkyl_tok_geo_grid_inp *inp,
    const double *xn, double *ret);

/**
 * Compute geometry (mapc2p) on a specified computational grid. The
 * output array must be pre-allocated by the caller.
 *
 * @param geo Geometry object
 * @param ginp Input structure for creating mapc2p
 * @param mapc2p On output, the DG representation of mapc2p
 */
void gkyl_tok_geo_calc(struct gk_geometry* up, struct gkyl_range *nrange, double dzc[3], 
  evalf_t mapc2p_func, void* mapc2p_ctx, evalf_t bmag_func, void *bmag_ctx, 
  struct gkyl_array *mc2p_nodal_fd, struct gkyl_array *mc2p_nodal, struct gkyl_array *mc2p, struct gkyl_array *mc2prz_nodal_fd, struct gkyl_array *mc2prz_nodal, struct gkyl_array *mc2prz, struct gkyl_array *dphidtheta_nodal);

/**
 * Return cumulative statistics from geometry computations
 *
 * @param geo Geometry object
 * @return Cumulative statistics
 */
struct gkyl_tok_geo_stat gkyl_tok_geo_get_stat(const gkyl_tok_geo *geo);

/**
 * Delete updater.
 *
 * @param geo Geometry object to delete
 */
void gkyl_tok_geo_release(gkyl_tok_geo *geo);

struct gkyl_range* gkyl_tok_geo_get_nrange(gkyl_tok_geo* geo);
struct gkyl_array* gkyl_tok_geo_get_mc2p_nodal_fd(gkyl_tok_geo* geo);
double* gkyl_tok_geo_get_dzc(gkyl_tok_geo* geo);
