#pragma once

// Identifiers for various equation systems
enum gkyl_eqn_type {
  GKYL_EQN_EULER, // Euler equations
  GKYL_EQN_SR_EULER, // SR Euler equations
  GKYL_EQN_ISO_EULER, // Isothermal Euler equations
  GKYL_EQN_TEN_MOMENT, // Ten-moment (with pressure tensor)
  GKYL_EQN_MAXWELL, // Maxwell equations
  GKYL_EQN_MHD,  // Ideal MHD equations
  GKYL_EQN_BURGERS, // Burgers equations
  GKYL_EQN_ADVECTION, // Scalar advection equation
  GKYL_EQN_EULER_PKPM, // Euler equations with parallel-kinetic-perpendicular-moment (pkpm) model
};

// Identifiers for specific field object types
enum gkyl_field_id {
  GKYL_FIELD_E_B = 0, // Maxwell (E, B). This is default
  GKYL_FIELD_PHI = 1, // Poisson (only phi)
  GKYL_FIELD_PHI_A = 2, // Poisson with static B = curl(A) (phi, A)
  GKYL_FIELD_NULL = 3, // no field is present
};

// Identifiers for subsidary models
// These are used to distinguish things like special relativistic from non-relativistic
// or the parallel-kinetic-perpendicular-moment model
enum gkyl_model_id {
  GKYL_MODEL_DEFAULT = 0, // No subsidiary model specified
  GKYL_MODEL_SR = 1,
  GKYL_MODEL_GEN_GEO = 2,
  GKYL_MODEL_PKPM = 3,
  GKYL_MODEL_SR_PKPM = 4,
};

// Identifiers for specific diffusion object types
enum gkyl_diffusion_id {
  GKYL_NO_DIFFUSION = 0, // No diffusion. This is default.
  GKYL_ISO_DIFFUSION, // Isotropic diffusion. 
  GKYL_ANISO_DIFFUSION, // Anisotropic diffusion.
  GKYL_EULER_ISO_DIFFUSION, // Diffusion in isothermal Euler equations (momentum equation)
  GKYL_EULER_DIFFUSION, // Diffusion in Euler equations (momentum and energy equations)
};

// Identifiers for specific collision object types
enum gkyl_collision_id {
  GKYL_NO_COLLISIONS = 0, // No collisions. This is default
  GKYL_BGK_COLLISIONS, // BGK Collision operator
  GKYL_LBO_COLLISIONS, // LBO Collision operator
  GKYL_FPO_COLLISIONS, // FPO Collision operator
};

// Identifiers for specific source object types
enum gkyl_source_id {
  GKYL_NO_SOURCE = 0, // No source. This is default
  GKYL_FUNC_SOURCE, // Source given by function
  GKYL_BFLUX_SOURCE // Source which scales to boundary fluxes
};

// type of quadrature to use
enum gkyl_quad_type {
  GKYL_GAUSS_QUAD, // Gauss-Legendre quadrature
  GKYL_GAUSS_LOBATTO_QUAD, // Gauss-Lobatto quadrature
};

/** Flags for indicating acting edge of velocity space */
enum gkyl_vel_edge { 
  GKYL_VX_LOWER, GKYL_VY_LOWER, GKYL_VZ_LOWER, 
  GKYL_VX_UPPER, GKYL_VY_UPPER, GKYL_VZ_UPPER 
};