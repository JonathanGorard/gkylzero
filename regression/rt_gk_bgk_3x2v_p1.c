#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gkyl_alloc.h>
#include <gkyl_const.h>
#include <gkyl_fem_parproj.h>
#include <gkyl_gyrokinetic.h>
#include <gkyl_util.h>

#include <gkyl_null_comm.h>

#ifdef GKYL_HAVE_MPI
#include <mpi.h>
#include <gkyl_mpi_comm.h>
#ifdef GKYL_HAVE_NCCL
#include <gkyl_nccl_comm.h>
#endif
#endif

#include <rt_arg_parse.h>

struct lapd_cart_ctx
{
  // Mathematical constants (dimensionless).
  double pi;

  // Physical constants (using non-normalized physical units).
  double epsilon0; // Permittivity of free space.
  double mass_elc; // Electron mass.
  double charge_elc; // Electron charge.
  double mass_ion; // Proton mass.
  double charge_ion; // Proton charge.

  double Te; // Electron temperature.
  double Ti; // Ion temperature.
  double B0; // Reference magnetic field strength (Tesla).
  double n0; // Reference number density (1 / m^3).

  double nu_frac; // Collision frequency fraction.

  // Derived physical quantities (using non-normalized physical units).
  double log_lambda_elc; // Logarithm of electron wavelength.
  double nu_elc; // Electron collision frequency.
  double log_lambda_ion; // Logarithm of ion wavelength.
  double nu_ion; // Ion collision frequency.

  double c_s; // Sound speed.
  double vte; // Electron thermal velocity.
  double vti; // Ion thermal velocity.
  double omega_ci; // Ion cyclotron frequency.
  double rho_si; // Ion-sound gyroradius.

  double Te_src; // Source electron temperature.
  double r_src; // Source radial extent.
  double L_src; // Source length.
  double S0; // Source reference number density.
  double floor_src; // Minimum source intensity;

  // Simulation parameters.
  int Nx; // Cell count (configuration space: x-direction).
  int Ny; // Cell count (configuration space: y-direction).
  int Nz; // Cell count (configuration space: z-direction).
  int Nv; // Cell count (velocity space: parallel velocity direction).
  int Nmu; // Cell count (velocity space: magnetic moment direction).
  double Lx; // Domain size (configuration space: x-direction).
  double Ly; // Domain size (configuration space: y-direction).
  double Lz; // Domain size (configuration space: z-direction).
  double L_perp; // Perpendicular length of domain.
  double Lv_elc; // Domain size (electron velocity space: parallel velocity direction).
  double Lmu_elc; // Domain size (electron velocity space: magnetic moment direction).
  double Lv_ion; // Domain size (ion velocity space: parallel velocity direction).
  double Lmu_ion; // Domain size (ion velocity space: magnetic moment direction).
  double t_end; // Final simulation time.
  int num_frames; // Number of output frames.
};

struct lapd_cart_ctx
create_ctx(void)
{
  // Mathematical constants (dimensionless).
  double pi = M_PI;

  // Physical constants (using non-normalized physical units).
  double epsilon0 = GKYL_EPSILON0; // Permittivity of free space.
  double mass_elc = GKYL_PROTON_MASS * 3.973 / 400.0; // Electron mass.
  double charge_elc = -GKYL_ELEMENTARY_CHARGE; // Electron charge.
  double mass_ion = 3.973 * GKYL_PROTON_MASS; // Proton mass.
  double charge_ion = GKYL_ELEMENTARY_CHARGE; // Proton charge.

  double Te = 6.0 * GKYL_ELEMENTARY_CHARGE; // Electron temperature.
  double Ti = 1.0 * GKYL_ELEMENTARY_CHARGE; // Ion temperature.
  double B0 = 0.0398; // Reference magnetic field strength (Tesla).
  double n0 = 2.0e18; //  Reference number density (1 / m^3).

  double nu_frac = 0.1; // Collision frequency fraction.

  // Derived physical quantities (using non-normalized physical units).
  double log_lambda_elc = 6.6 - 0.5 * log(n0 / 1.0e20) + 1.5 * log(Te / charge_ion); // Logarithm of electron wavelength.
  double nu_elc = nu_frac * log_lambda_elc * (charge_ion * charge_ion * charge_ion * charge_ion) * n0 /
    (6.0 * sqrt(2.0) * pi * sqrt(pi) * epsilon0 * epsilon0 * sqrt(mass_elc) * (Te * sqrt(Te))); // Electron collision frequency.
  double log_lambda_ion = 6.6 - 0.5 * log(n0 / 1.0e20) + 1.5 * log(Ti / charge_ion); // Logarithm of ion wavelength.
  double nu_ion = nu_frac * log_lambda_ion * (charge_ion * charge_ion * charge_ion * charge_ion) * n0 /
    (12.0 * pi * sqrt(pi) * epsilon0 * epsilon0 * sqrt(mass_ion) * (Ti * sqrt(Ti))); // Ion collision frequency.
  
  double c_s = sqrt(Te / mass_ion); // Sound speed.
  double vte = sqrt(Te / mass_elc); // Electron thermal velocity.
  double vti = sqrt(Ti / mass_ion); // Ion thermal velocity.
  double omega_ci = fabs(charge_ion * B0 / mass_ion); // Ion cyclotron frequency.
  double rho_si = c_s / omega_ci; // Ion-sound gyroradius.

  double Te_src = 2.0 * Te; // Source electron temperature.
  double r_src = 20.0 * rho_si; // Source radial extent.
  double L_src = 0.5 * rho_si; // Source length.
  double S0 = 1.08 * n0 * c_s * (36.0 * 40.0 * rho_si); // Source reference number density.
  double floor_src = 0.01; // Minimum source intensity.

  // Simulation parameters.
  int Nx = 18; // Cell count (configuration space: x-direction).
  int Ny = 18; // Cell count (configuration space: y-direction).
  int Nz = 10; // Cell count (configuration space: z-direction).
  int Nv = 10; // Cell count (velocity space: parallel velocity direction).
  int Nmu = 5; // Cell count (velocity space: magnetic moment direction).
  double Lx = 100.0 * rho_si; // Domain size (configuration space: x-direction).
  double Ly = 100.0 * rho_si; // Domain size (configuration space: y-direction).
  double Lz = 36.0 *  40.0 * rho_si; // Domain size (configuration space: z-direction).
  double L_perp = Lx; // Perpendicular length of domain.
  double Lv_elc = 8.0 * vte; // Domain size (electron velocity space: parallel velocity direction).
  double Lmu_elc = (3.0 / 2.0) * 0.5 * mass_elc * (4.0 * vte) * (4.0 * vte) / (2.0 * B0); // Domain size (electron velocity space: magnetic moment direction).
  double Lv_ion = 8.0 * vti; // Domain size (ion velocity space: parallel velocity direction).
  double Lmu_ion = (3.0 / 2.0) * 0.5 * mass_ion * (4.0 * vti) * (4.0 * vti) / (2.0 * B0); // Domain size (ion velocity space: magnetic moment direction).
  double t_end = 5.0e-7; // Final simulation time.
  int num_frames = 1; // Number of output frames.
  
  struct lapd_cart_ctx ctx = {
    .pi = pi,
    .epsilon0 = epsilon0,
    .mass_elc = mass_elc,
    .charge_elc = charge_elc,
    .mass_ion = mass_ion,
    .charge_ion = charge_ion,
    .Te = Te,
    .Ti = Ti,
    .B0 = B0,
    .n0 = n0,
    .nu_frac = nu_frac,
    .log_lambda_elc = log_lambda_elc,
    .nu_elc = nu_elc,
    .log_lambda_ion = log_lambda_ion,
    .nu_ion = nu_ion,
    .c_s = c_s,
    .vte = vte,
    .vti = vti,
    .omega_ci = omega_ci,
    .rho_si = rho_si,
    .Te_src = Te_src,
    .r_src = r_src,
    .L_src = L_src,
    .S0 = S0,
    .floor_src = floor_src,
    .Nx = Nx,
    .Ny = Ny,
    .Nz = Nz,
    .Nv = Nv,
    .Nmu = Nmu,
    .Lx = Lx,
    .Ly = Ly,
    .Lz = Lz,
    .L_perp = L_perp,
    .Lv_elc = Lv_elc,
    .Lmu_elc = Lmu_elc,
    .Lv_ion = Lv_ion,
    .Lmu_ion = Lmu_ion,
    .t_end = t_end,
    .num_frames = num_frames,
  };

  return ctx;
}

void
evalDensityInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct lapd_cart_ctx *app = ctx;
  double x = xn[0], y = xn[1];

  double n0 = app -> n0;
  double L_perp = app -> L_perp;

  double r = sqrt(x * x + y * y);

  pcg32_random_t rng;
  double perturb = 2.0e-3 * (1.0 - 0.5 * gkyl_pcg32_rand_double(&rng));

  double n = 0.0;

  if (r < 0.5 * L_perp) {
    n = ((1.0 - (1.0 / 20.0)) * pow(1.0 - (r / (0.5 * L_perp)) * (r / (0.5 * L_perp)), 3.0) + (1.0 / 20.0)) * n0 * (1.0 + perturb);
  }
  else {
    n = (1.0 / 20.0) * n0 * (1.0 + perturb);
  }

  // Set number density.
  fout[0] = n;
}

void
evalUparInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  // Set parallel velocity.
  fout[0] = 0.0;
}

void
evalTempElcInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct lapd_cart_ctx *app = ctx;
  double x = xn[0], y = xn[1];

  double Te = app -> Te;
  double L_perp = app -> L_perp;

  double r = sqrt(x * x + y * y);

  double T = 0.0;

  if (r < 0.5 * L_perp) {
    T = ((1.0 - (1.0 / 5.0)) * pow(1.0 - (r / (0.5 * L_perp)) * (r / (0.5 * L_perp)), 3.0) + (1.0 / 5.0)) * Te;
  }
  else {
    T = (1.0 / 5.0) * Te;
  }

  // Set electron temperature.
  fout[0] = T;
}

void
evalTempIonInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct lapd_cart_ctx *app = ctx;

  double Ti = app -> Ti;

  // Set ion temperature.
  fout[0] = Ti;
}

void
evalSourceDensityInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct lapd_cart_ctx *app = ctx;
  double x = xn[0], y = xn[1];

  double r_src = app -> r_src;
  double L_src = app -> L_src;
  double S0 = app -> S0;
  double floor_src = app -> floor_src;

  double r = sqrt(x * x + y * y);

  // Set source number density.
  fout[0] = S0 * (floor_src + (1.0 - floor_src) * 0.5 * (1.0 - tanh((r - r_src) / L_src)));
}

void
evalSourceUparInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  // Set source parallel velocity.
  fout[0] = 0.0;
}

void
evalSourceTempElcInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct lapd_cart_ctx *app = ctx;
  double x = xn[0], y = xn[1];

  double Te_src = app -> Te_src;
  double L_perp = app -> L_perp;

  double r = sqrt(x * x + y * y);

  double T = 0.0;

  if (r < 0.5 * L_perp) {
    T = (1.0 - (1.0 / 2.5) * pow(1.0 - (r / (0.5 * L_perp)) * (r / (0.5 * L_perp)), 3.0) + (1.0 / 2.5)) * Te_src;
  }
  else {
    T = (1.0 / 2.5) * Te_src;
  }

  // Set electron source temperature.
  fout[0] = T;
}

void
evalSourceTempIonInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct lapd_cart_ctx *app = ctx;

  double Ti = app -> Ti;

  // Set ion source temperature.
  fout[0] = Ti;
}

void
evalNuElcInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct lapd_cart_ctx *app = ctx;

  double nu_elc = app -> nu_elc;

  // Set electron collision frequency.
  fout[0] = nu_elc;
}

void
evalNuIonInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct lapd_cart_ctx *app = ctx;

  double nu_ion = app -> nu_ion;

  // Set ion collision frequency.
  fout[0] = nu_ion;
}

static inline void
mapc2p(double t, const double* GKYL_RESTRICT zc, double* GKYL_RESTRICT xp, void* ctx)
{
  // Set physical coordinates (X, Y, Z) from computational coordinates (x, y, z).
  xp[0] = zc[0]; xp[1] = zc[1]; xp[2] = zc[2];
}

void
bmag_func(double t, const double* GKYL_RESTRICT zc, double* GKYL_RESTRICT fout, void* ctx)
{
  struct lapd_cart_ctx *app = ctx;

  double B0 = app -> B0;

  // Set magnetic field strength.
  fout[0] = B0;
}

void
write_data(struct gkyl_tm_trigger* iot, gkyl_gyrokinetic_app* app, double t_curr)
{
  if (gkyl_tm_trigger_check_and_bump(iot, t_curr)) {
    gkyl_gyrokinetic_app_write(app, t_curr, iot -> curr - 1);
    gkyl_gyrokinetic_app_calc_mom(app);
    gkyl_gyrokinetic_app_write_mom(app, t_curr, iot -> curr - 1);
    gkyl_gyrokinetic_app_write_source_mom(app, t_curr, iot -> curr - 1);
  }
}

int
main(int argc, char **argv)
{
  struct gkyl_app_args app_args = parse_app_args(argc, argv);

#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi) {
    MPI_Init(&argc, &argv);
  }
#endif

  if (app_args.trace_mem) {
    gkyl_cu_dev_mem_debug_set(true);
    gkyl_mem_debug_set(true);
  }

  struct lapd_cart_ctx ctx = create_ctx(); // Context for initialization functions.

  int NX = APP_ARGS_CHOOSE(app_args.xcells[0], ctx.Nx);
  int NY = APP_ARGS_CHOOSE(app_args.xcells[1], ctx.Ny);
  int NZ = APP_ARGS_CHOOSE(app_args.xcells[2], ctx.Nz);
  int NV = APP_ARGS_CHOOSE(app_args.vcells[0], ctx.Nv);
  int NMU = APP_ARGS_CHOOSE(app_args.vcells[1], ctx.Nmu);

  int nrank = 1; // Number of processors in simulation.
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi) {
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);
  }
#endif  

  // Create global range.
  int ccells[] = { NX, NY, NZ };
  int cdim = sizeof(ccells) / sizeof(ccells[0]);
  struct gkyl_range cglobal_r;
  gkyl_create_global_range(cdim, ccells, &cglobal_r);

  // Create decomposition.
  int cuts[cdim];
#ifdef GKYL_HAVE_MPI  
  for (int d = 0; d < cdim; d++) {
    if (app_args.use_mpi) {
      cuts[d] = app_args.cuts[d];
    }
    else {
      cuts[d] = 1;
    }
  }
#else
  for (int d = 0; d < cdim; d++) {
    cuts[d] = 1;
  }
#endif  
    
  struct gkyl_rect_decomp *decomp = gkyl_rect_decomp_new_from_cuts(cdim, cuts, &cglobal_r);

  // Construct communicator for use in app.
  struct gkyl_comm *comm;
#ifdef GKYL_HAVE_MPI
  if (app_args.use_gpu && app_args.use_mpi) {
#ifdef GKYL_HAVE_NCCL
    comm = gkyl_nccl_comm_new( &(struct gkyl_nccl_comm_inp) {
        .mpi_comm = MPI_COMM_WORLD,
        .decomp = decomp
      }
    );
#else
    printf(" Using -g and -M together requires NCCL.\n");
    assert(0 == 1);
#endif
  }
  else if (app_args.use_mpi) {
    comm = gkyl_mpi_comm_new( &(struct gkyl_mpi_comm_inp) {
        .mpi_comm = MPI_COMM_WORLD,
        .decomp = decomp
      }
    );
  }
  else {
    comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
        .decomp = decomp,
        .use_gpu = app_args.use_gpu
      }
    );
  }
#else
  comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .decomp = decomp,
      .use_gpu = app_args.use_gpu
    }
  );
#endif

  int my_rank;
  gkyl_comm_get_rank(comm, &my_rank);
  int comm_size;
  gkyl_comm_get_size(comm, &comm_size);

  int ncuts = 1;
  for (int d = 0; d < cdim; d++) {
    ncuts *= cuts[d];
  }

  if (ncuts != comm_size) {
    if (my_rank == 0) {
      fprintf(stderr, "*** Number of ranks, %d, does not match total cuts, %d!\n", comm_size, ncuts);
    }
    goto mpifinalize;
  }

  for (int d = 0; d < cdim - 1; d++) {
    if (cuts[d] > 1) {
      if (my_rank == 0) {
        fprintf(stderr, "*** Parallelization only allowed in z. Number of ranks, %d, in direction %d cannot be > 1!\n", cuts[d], d);
      }
      goto mpifinalize;
    }
  }

  // Electron species.
  struct gkyl_gyrokinetic_species elc = {
    .name = "elc",
    .charge = ctx.charge_elc, .mass = ctx.mass_elc,
    .lower = { -0.5 * ctx.Lv_elc, 0.0},
    .upper = { 0.5 * ctx.Lv_elc, ctx.Lmu_elc},
    .cells = { NV, NMU },
    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .density = evalDensityInit,
      .ctx_density = &ctx,
      .upar = evalUparInit,
      .ctx_upar = &ctx,
      .temp = evalTempElcInit,
      .ctx_temp = &ctx,
    },
    .collisions =  {
      .collision_id = GKYL_BGK_COLLISIONS,
      .self_nu = evalNuElcInit,
      .ctx = &ctx,
      .num_cross_collisions = 1,
      .collide_with = { "ion" },
    },
    .source = {
      .source_id = GKYL_PROJ_SOURCE,
      .write_source = true,
      .num_sources = 1,
      .projection[0] = {
        .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM, 
        .density = evalSourceDensityInit,
        .ctx_density = &ctx,
        .upar = evalSourceUparInit,
        .ctx_upar = &ctx,
        .temp = evalSourceTempElcInit,
        .ctx_temp = &ctx,
      }, 
    },
    
    .bcx = {
      .lower = { .type = GKYL_SPECIES_ZERO_FLUX, },
      .upper = { .type = GKYL_SPECIES_ZERO_FLUX, },
    },
    .bcy = {
      .lower = { .type = GKYL_SPECIES_ZERO_FLUX, },
      .upper = { .type = GKYL_SPECIES_ZERO_FLUX, },
    },
    .bcz = {
      .lower = { .type = GKYL_SPECIES_GK_SHEATH, },
      .upper = { .type = GKYL_SPECIES_GK_SHEATH, },
    },

    .num_diag_moments = 7,
    .diag_moments = { "M0", "M1", "M2", "M2par", "M2perp", "M3par", "M3perp" },
  };

  // Ion species.
  struct gkyl_gyrokinetic_species ion = {
    .name = "ion",
    .charge = ctx.charge_ion, .mass = ctx.mass_ion,
    .lower = { -0.5 * ctx.Lv_ion, 0.0},
    .upper = { 0.5 * ctx.Lv_ion, ctx.Lmu_ion},
    .cells = { NV, NMU },
    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM, 
      .density = evalDensityInit,
      .ctx_density = &ctx,
      .upar = evalUparInit,
      .ctx_upar = &ctx,
      .temp = evalTempIonInit,
      .ctx_temp = &ctx,
    },
    .collisions =  {
      .collision_id = GKYL_BGK_COLLISIONS,
      .self_nu = evalNuIonInit,
      .ctx = &ctx,
      .num_cross_collisions = 1,
      .collide_with = { "elc" },
    },
    .source = {
      .source_id = GKYL_PROJ_SOURCE,
      .write_source = true,
      .num_sources = 1,
      .projection[0] = {
        .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
        .density = evalSourceDensityInit,
        .ctx_density = &ctx,
        .upar = evalSourceUparInit,
        .ctx_upar = &ctx,
        .temp = evalSourceTempIonInit,
        .ctx_temp = &ctx,
      }, 
    },
    
    .bcx = {
      .lower = { .type = GKYL_SPECIES_ZERO_FLUX, },
      .upper = { .type = GKYL_SPECIES_ZERO_FLUX, },
    },
    .bcy = {
      .lower = { .type = GKYL_SPECIES_ZERO_FLUX, },
      .upper = { .type = GKYL_SPECIES_ZERO_FLUX, },
    },
    .bcz = {
      .lower = { .type = GKYL_SPECIES_GK_SHEATH, },
      .upper = { .type = GKYL_SPECIES_GK_SHEATH, },
    },

    .num_diag_moments = 7,
    .diag_moments = { "M0", "M1", "M2", "M2par", "M2perp", "M3par", "M3perp" },
  };

  // Field.
  struct gkyl_gyrokinetic_field field = {
    .bmag_fac = ctx.B0,
    .fem_parbc = GKYL_FEM_PARPROJ_NONE,
    .poisson_bcs = {.lo_type = { GKYL_POISSON_DIRICHLET, GKYL_POISSON_DIRICHLET },
                    .up_type = { GKYL_POISSON_DIRICHLET, GKYL_POISSON_DIRICHLET },
                    .lo_value = { 0.0, 0.0 }, .up_value = { 0.0, 0.0}},
  };

  // GK app.
  struct gkyl_gk app_inp = {
    .name = "gk_bgk_3x2v_p1",

    .cdim = 3, .vdim = 2,
    .lower = { -0.5 * ctx.Lx, -0.5 * ctx.Ly, -0.5 * ctx.Lz },
    .upper = { 0.5 * ctx.Lx, 0.5 * ctx.Ly, 0.5 * ctx.Lz },
    .cells = { NX, NY, NZ },
    .poly_order = 1,
    .basis_type = app_args.basis_type,
    .cfl_frac = 0.1, 

    .geometry = {
      .geometry_id = GKYL_MAPC2P,
      .mapc2p = mapc2p,
      .c2p_ctx = &ctx,
      .bmag_func = bmag_func,
      .bmag_ctx = &ctx
    },

    .num_periodic_dir = 0,
    .periodic_dirs = { },

    .num_species = 2,
    .species = { elc, ion },
    .field = field,

    .use_gpu = app_args.use_gpu,

    .has_low_inp = true,
    .low_inp = {
      .local_range = decomp -> ranges[my_rank],
      .comm = comm
    }
  };

  // Create app object.
  gkyl_gyrokinetic_app *app = gkyl_gyrokinetic_app_new(&app_inp);

  // Initial and final simulation times.
  double t_curr = 0.0, t_end = ctx.t_end;

  // Create trigger for IO.
  int num_frames = ctx.num_frames;
  struct gkyl_tm_trigger io_trig = { .dt = t_end / num_frames };

  // Initialize simulation.
  gkyl_gyrokinetic_app_apply_ic(app, t_curr);
  write_data(&io_trig, app, t_curr);

  gkyl_gyrokinetic_app_calc_field_energy(app, t_curr);
  gkyl_gyrokinetic_app_calc_integrated_mom(app, t_curr);

  // Compute initial guess of maximum stable time-step.
  double dt = t_end - t_curr;

  long step = 1;
  while ((t_curr < t_end) && (step <= app_args.num_steps)) {
    gkyl_gyrokinetic_app_cout(app, stdout, "Taking time-step %ld at t = %g ...", step, t_curr);
    struct gkyl_update_status status = gkyl_gyrokinetic_update(app, dt);
    gkyl_gyrokinetic_app_cout(app, stdout, " dt = %g\n", status.dt_actual);

    gkyl_gyrokinetic_app_calc_field_energy(app, t_curr);
    gkyl_gyrokinetic_app_calc_integrated_mom(app, t_curr);

    if (!status.success) {
      gkyl_gyrokinetic_app_cout(app, stdout, "** Update method failed! Aborting simulation ....\n");
      break;
    }

    t_curr += status.dt_actual;
    dt = status.dt_suggested;

    write_data(&io_trig, app, t_curr);

    step += 1;
  }

  gkyl_gyrokinetic_app_calc_field_energy(app, t_curr);
  gkyl_gyrokinetic_app_calc_integrated_mom(app, t_curr);

  write_data(&io_trig, app, t_curr);
  gkyl_gyrokinetic_app_stat_write(app);
  
  struct gkyl_gyrokinetic_stat stat = gkyl_gyrokinetic_app_stat(app);

  gkyl_gyrokinetic_app_cout(app, stdout, "\n");
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of update calls %ld\n", stat.nup);
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of forward-Euler calls %ld\n", stat.nfeuler);
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of RK stage-2 failures %ld\n", stat.nstage_2_fail);
  if (stat.nstage_2_fail > 0) {
    gkyl_gyrokinetic_app_cout(app, stdout, "  Max rel dt diff for RK stage-2 failures %g\n", stat.stage_2_dt_diff[1]);
    gkyl_gyrokinetic_app_cout(app, stdout, "  Min rel dt diff for RK stage-2 failures %g\n", stat.stage_2_dt_diff[0]);
  }  
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of RK stage-3 failures %ld\n", stat.nstage_3_fail);
  gkyl_gyrokinetic_app_cout(app, stdout, "Species RHS calc took %g secs\n", stat.species_rhs_tm);
  gkyl_gyrokinetic_app_cout(app, stdout, "Species collisions RHS calc took %g secs\n", stat.species_coll_tm);
  gkyl_gyrokinetic_app_cout(app, stdout, "Field RHS calc took %g secs\n", stat.field_rhs_tm);
  gkyl_gyrokinetic_app_cout(app, stdout, "Species collisional moments took %g secs\n", stat.species_coll_mom_tm);
  gkyl_gyrokinetic_app_cout(app, stdout, "Total updates took %g secs\n", stat.total_tm);

  gkyl_gyrokinetic_app_cout(app, stdout, "Number of write calls %ld,\n", stat.nio);
  gkyl_gyrokinetic_app_cout(app, stdout, "IO time took %g secs \n", stat.io_tm);

  // Free resources after simulation completion.
  gkyl_gyrokinetic_app_release(app);
  gkyl_rect_decomp_release(decomp);
  gkyl_comm_release(comm);

  mpifinalize:
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi) {
    MPI_Finalize();
  }
#endif
  
  return 0;
}