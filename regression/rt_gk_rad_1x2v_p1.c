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
#endif

#include <rt_arg_parse.h>

struct rad_ctx
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

  double k_perp_rho_si; // Product of perpendicular wavenumber and ion-sound gyroradius.

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

  double k_perp; // Perpendicular wavenumber (for Poisson solver).

  // Simulation parameters.
  long Nx; // Cell count (configuration space: x-direction).
  long Nv; // Cell count (velocity space: parallel velocity direction).
  long Nmu; // Cell count (velocity space: magnetic moment direction).
  double Lx; // Domain size (configuration space: x-direction).
  double Lv_elc; // Domain size (electron velocity space: parallel velocity direction).
  double Lmu_elc; // Domain size (electron velocity space: magnetic moment direction).
  double Lv_ion; // Domain size (ion velocity space: parallel velocity direction).
  double Lmu_ion; // Domain size (ion velocity space: magnetic moment direction).
  double t_end; // Final simulation time.
};

struct rad_ctx
create_ctx(void)
{
  // Mathematical constants (dimensionless).
  double pi = M_PI;

  // Physical constants (using non-normalized physical units).
  double epsilon0 = GKYL_EPSILON0; // Permittivity of free space.
  double mass_elc = GKYL_ELECTRON_MASS; // Electron mass.
  double charge_elc = -GKYL_ELEMENTARY_CHARGE; // Electron charge.
  double mass_ion = GKYL_PROTON_MASS; // Proton mass.
  double charge_ion = GKYL_ELEMENTARY_CHARGE; // Proton charge.

  double Te = 30.0 * GKYL_ELEMENTARY_CHARGE; // Electron temperature.
  double Ti = 30.0 * GKYL_ELEMENTARY_CHARGE; // Ion temperature.
  double B0 = 1.0; // Reference magnetic field strength (Tesla).
  double n0 = 1.0e19; //  Reference number density (1 / m^3).

  double nu_frac = 0.1; // Collision frequency fraction.

  double k_perp_rho_si = 0.1; // Product of perpendicular wavenumber and ion-sound gyroradius.

  // Derived physical quantities (using normalized code units).
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

  double k_perp = k_perp_rho_si / rho_si; // Perpendicular wavenumber (for Poisson solver).

  // Simulation parameters.
  long Nx = 2; // Cell count (configuration space: x-direction).
  long Nv = 16; // Cell count (velocity space: parallel velocity direction).
  long Nmu = 8; // Cell count (velocity space: magnetic moment direction).
  double Lx = 100.0 * rho_si; // Domain size (configuration space: x-direction).
  double Lv_elc = 8.0 * vte; // Domain size (electron velocity space: parallel velocity direction).
  double Lmu_elc = 0.75 * mass_elc * (4.0 * vte) * (4.0 * vte) / (2.0 * B0); // Domain size (electron velocity space: magnetic moment direction).
  double Lv_ion = 8.0 * vti; // Domain size (ion velocity space: parallel velocity direction).
  double Lmu_ion = 0.75 * mass_ion * (4.0 * vti) * (4.0 * vti) / (2.0 * B0); // Domain size (ion velocity space: magnetic moment direction).
  double t_end = 1.0e-7;
  
  struct rad_ctx ctx = {
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
    .k_perp_rho_si = k_perp_rho_si,
    .log_lambda_elc = log_lambda_elc,
    .nu_elc = nu_elc,
    .log_lambda_ion = log_lambda_ion,
    .nu_ion = nu_ion,
    .c_s = c_s,
    .vte = vte,
    .vti = vti,
    .omega_ci = omega_ci,
    .rho_si = rho_si,
    .k_perp = k_perp,
    .Nx = Nx,
    .Nv = Nv,
    .Nmu = Nmu,
    .Lx = Lx,
    .Lv_elc = Lv_elc,
    .Lmu_elc = Lmu_elc,
    .Lv_ion = Lv_ion,
    .Lmu_ion = Lmu_ion,
    .t_end = t_end,
  };

  return ctx;
}

void
evalDensityInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct rad_ctx *app = ctx;

  double n0 = app -> n0;

  // Set number density.
  fout[0] = n0;
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
  struct rad_ctx *app = ctx;

  double Te = app -> Te;

  // Set electron temperature.
  fout[0] = Te;
}

void
evalTempIonInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct rad_ctx *app = ctx;

  double Ti = app -> Ti;

  // Set ion temperature.
  fout[0] = Ti;
}

void
evalNuElcInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct rad_ctx *app = ctx;

  double nu_elc = app -> nu_elc;

  // Set electron collision frequency.
  fout[0] = nu_elc;
}

void
evalNuIonInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct rad_ctx *app = ctx;

  double nu_ion = app -> nu_ion;

  // Set ion collision frequency.
  fout[0] = app -> nu_ion;
}

static inline void
mapc2p(double t, const double* GKYL_RESTRICT zc, double* GKYL_RESTRICT xp, void* ctx)
{
  // Set physical coordinates (x, v, mu) from computational coordinates (x, v, mu).
  xp[0] = zc[0]; xp[1] = zc[1]; xp[2] = zc[2];
}

void
bmag_func(double t, const double *zc, double* GKYL_RESTRICT fout, void* ctx)
{
  struct rad_ctx *app = ctx;

  double B0 = app -> B0;

  // Set magnetic field strength.
  fout[0] = B0;
}

int
main(int argc, char **argv)
{
  struct gkyl_app_args app_args = parse_app_args(argc, argv);

#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
  {
    MPI_Init(&argc, &argv);
  }
#endif

  if (app_args.trace_mem) 
  {
    gkyl_cu_dev_mem_debug_set(true);
    gkyl_mem_debug_set(true);
  }

  struct rad_ctx ctx = create_ctx(); // Context for initialization functions.

  int NX = APP_ARGS_CHOOSE(app_args.xcells[0], ctx.Nx);
  int NV = APP_ARGS_CHOOSE(app_args.vcells[0], ctx.Nv);
  int NMU = APP_ARGS_CHOOSE(app_args.vcells[1], ctx.Nmu);

  int nrank = 1; // Number of processors in simulation.
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
  {
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);
  }
#endif  

  // Create global range.
  int ccells[] = { NX };
  int cdim = sizeof(ccells) / sizeof(ccells[0]);
  struct gkyl_range cglobal_r;
  gkyl_create_global_range(cdim, ccells, &cglobal_r);

  // Create decomposition.
  int cuts[cdim];
#ifdef GKYL_HAVE_MPI  
  for (int d = 0; d < cdim; d++)
  {
    if (app_args.use_mpi)
    {
      cuts[d] = app_args.cuts[d];
    }
    else
    {
      cuts[d] = 1;
    }
  }
#else
  for (int d = 0; d < cdim; d++)
  {
    cuts[d] = 1;
  }
#endif  
    
  struct gkyl_rect_decomp *decomp = gkyl_rect_decomp_new_from_cuts(cdim, cuts, &cglobal_r);

  // Construct communicator for use in app.
  struct gkyl_comm *comm;
#ifdef GKYL_HAVE_MPI
  if (app_args.use_gpu && app_args.use_mpi)
  {
#ifdef GKYL_HAVE_NCCL
    comm = gkyl_nccl_comm_new( &(struct gkyl_nccl_comm_inp)
      {
        .mpi_comm = MPI_COMM_WORLD,
        .decomp = decomp
      }
    );
#else
    printf(" Using -g and -M together requires NCCL.\n");
    assert(0 == 1);
#endif
  }
  else if (app_args.use_mpi) 
  {
    comm = gkyl_mpi_comm_new( &(struct gkyl_mpi_comm_inp)
      {
        .mpi_comm = MPI_COMM_WORLD,
        .decomp = decomp
      }
    );
  }
  else
  {
    comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp)
      {
        .decomp = decomp,
        .use_gpu = app_args.use_gpu
      }
    );
  }
#else
  comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp)
    {
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
  for (int d = 0; d < cdim; d++)
  {
    ncuts *= cuts[d];
  }

  if (ncuts != comm_size)
  {
    if (my_rank == 0)
    {
      fprintf(stderr, "*** Number of ranks, %d, does not match total cuts, %d!\n", comm_size, ncuts);
    }
    goto mpifinalize;
  }

  for (int d = 0; d < cdim - 1; d++)
  {
    if (cuts[d] > 1)
    {
      if (my_rank == 0)
      {
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
      .proj_id = GKYL_PROJ_MAXWELLIAN, 
      .ctx_density = &ctx,
      .density = evalDensityInit,
      .ctx_upar = &ctx,
      .upar = evalUparInit,
      .ctx_temp = &ctx,
      .temp = evalTempElcInit,
    },
    .collisions =  {
      .collision_id = GKYL_LBO_COLLISIONS,
      .ctx = &ctx,
      .self_nu = evalNuElcInit,
      .num_cross_collisions = 1,
      .collide_with = { "ion" },
    },

    .bcx = {
      .lower = {.type = GKYL_SPECIES_ZERO_FLUX,},
      .upper = {.type = GKYL_SPECIES_ZERO_FLUX,},
    },
    .bcy = {
      .lower = {.type = GKYL_SPECIES_ZERO_FLUX,},
      .upper = {.type = GKYL_SPECIES_ZERO_FLUX,},
    },

    .radiation = {
      .radiation_id = GKYL_GK_RADIATION, 
      .num_cross_collisions = 1, 
      .collide_with = { "ion" },
      .z = 1,
      .charge_state = 0,
      .num_of_densities = 1, // Must be 1 for now.
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
      .proj_id = GKYL_PROJ_MAXWELLIAN, 
      .ctx_density = &ctx,
      .density = evalDensityInit,
      .ctx_upar = &ctx,
      .upar= evalUparInit,
      .ctx_temp = &ctx,
      .temp = evalTempIonInit,      
    },
    .collisions =  {
      .collision_id = GKYL_LBO_COLLISIONS,
      .ctx = &ctx,
      .self_nu = evalNuIonInit,
      .num_cross_collisions = 1,
      .collide_with = { "elc" },
    },

    .bcx = {
      .lower = {.type = GKYL_SPECIES_ZERO_FLUX,},
      .upper = {.type = GKYL_SPECIES_ZERO_FLUX,},
    },
    .bcy = {
      .lower = {.type = GKYL_SPECIES_ZERO_FLUX,},
      .upper = {.type = GKYL_SPECIES_ZERO_FLUX,},
    },
    
    .num_diag_moments = 7,
    .diag_moments = { "M0", "M1", "M2", "M2par", "M2perp", "M3par", "M3perp" },
  };

  // Field.
  struct gkyl_gyrokinetic_field field = {
    .bmag_fac = ctx.B0, 
    .fem_parbc = GKYL_FEM_PARPROJ_PERIODIC, 
    .kperpSq = ctx.k_perp * ctx.k_perp,
  };

  // GK app.
  struct gkyl_gk app_inp = {
    .name = "gk_rad_1x2v_p1",

    .cdim = 1, .vdim = 2,
    .lower = { -0.5 * ctx.Lx },
    .upper = { 0.5 * ctx.Lx },
    .cells = { NX },
    .poly_order = 1,
    .basis_type = app_args.basis_type,

    .geometry = {
      .geometry_id = GKYL_MAPC2P,
      .world = {0.0, 0.0},
      .mapc2p = mapc2p,
      .c2p_ctx = &ctx,
      .bmag_func = bmag_func,
      .bmag_ctx = &ctx
    },

    .num_periodic_dir = 1,
    .periodic_dirs = { 0 },

    .num_species = 2,
    .species = { elc, ion },
    .skip_field = true, 
    .field = field,

    .use_gpu = app_args.use_gpu,
  };
  
  // Create app object.
  gkyl_gyrokinetic_app *app = gkyl_gyrokinetic_app_new(&app_inp);

  // Initial and final simulation times.
  double t_curr = 0.0, t_end = ctx.t_end;

  // Initialize simulation.
  gkyl_gyrokinetic_app_apply_ic(app, t_curr);
  gkyl_gyrokinetic_app_write(app, t_curr, 0);

  gkyl_gyrokinetic_app_calc_field_energy(app, t_curr);
  gkyl_gyrokinetic_app_calc_integrated_mom(app, t_curr);

  // Compute initial guess of maximum stable time-step.
  double dt = t_end - t_curr;

  long step = 1;
  while ((t_curr < t_end) && (step <= app_args.num_steps))
  {
    gkyl_gyrokinetic_app_cout(app, stdout, "Taking time-step %ld at t = %g ...", step, t_curr);
    struct gkyl_update_status status = gkyl_gyrokinetic_update(app, dt);
    gkyl_gyrokinetic_app_cout(app, stdout, " dt = %g\n", status.dt_actual);

    if (step % 10 == 0)
    {
      gkyl_gyrokinetic_app_calc_field_energy(app, t_curr);
      gkyl_gyrokinetic_app_calc_integrated_mom(app, t_curr);
    }

    if (!status.success)
    {
      gkyl_gyrokinetic_app_cout(app, stdout, "** Update method failed! Aborting simulation ....\n");
      break;
    }

    t_curr += status.dt_actual;
    dt = status.dt_suggested;

    step += 1;
  }

  gkyl_gyrokinetic_app_calc_field_energy(app, t_curr);
  gkyl_gyrokinetic_app_calc_integrated_mom(app, t_curr);
  
  gkyl_gyrokinetic_app_write(app, t_curr, 1);
  gkyl_gyrokinetic_app_stat_write(app);
  
  struct gkyl_gyrokinetic_stat stat = gkyl_gyrokinetic_app_stat(app);

  gkyl_gyrokinetic_app_cout(app, stdout, "\n");
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of update calls %ld\n", stat.nup);
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of forward-Euler calls %ld\n", stat.nfeuler);
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of RK stage-2 failures %ld\n", stat.nstage_2_fail);
  if (stat.nstage_2_fail > 0)
  {
    gkyl_gyrokinetic_app_cout(app, stdout, "  Max rel dt diff for RK stage-2 failures %g\n", stat.stage_2_dt_diff[1]);
    gkyl_gyrokinetic_app_cout(app, stdout, "  Min rel dt diff for RK stage-2 failures %g\n", stat.stage_2_dt_diff[0]);
  }  
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of RK stage-3 failures %ld\n", stat.nstage_3_fail);
  gkyl_gyrokinetic_app_cout(app, stdout, "Species RHS calc took %g secs\n", stat.species_rhs_tm);
  gkyl_gyrokinetic_app_cout(app, stdout, "Species collisions RHS calc took %g secs\n", stat.species_coll_tm);
  gkyl_gyrokinetic_app_cout(app, stdout, "Field RHS calc took %g secs\n", stat.field_rhs_tm);
  gkyl_gyrokinetic_app_cout(app, stdout, "Species collisional moments took %g secs\n", stat.species_coll_mom_tm);
  gkyl_gyrokinetic_app_cout(app, stdout, "Total updates took %g secs\n", stat.total_tm);

  // Free resources after simulation completion.
  gkyl_gyrokinetic_app_release(app);

  mpifinalize:
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
  {
    MPI_Finalize();
  }
#endif
  
  return 0;
}