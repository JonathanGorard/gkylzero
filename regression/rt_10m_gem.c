// Geospace Environmental Modeling (GEM) magnetic reconnection test for the 10-moment equations.
// Input parameters match the equilibrium and initial conditions in Section 2, from the article:
// J. Birn et al. (2001), "Geospace Environmental Modeling (GEM) Magnetic Reconnection Challenge",
// Journal of Geophysical Research: Space Physics, Volume 106 (A3): 3715-3719.
// https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999JA900449

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gkyl_alloc.h>
#include <gkyl_moment.h>
#include <gkyl_util.h>
#include <gkyl_wv_ten_moment.h>

#include <gkyl_null_comm.h>

#ifdef GKYL_HAVE_MPI
#include <mpi.h>
#include <gkyl_mpi_comm.h>
#endif

#include <rt_arg_parse.h>

struct gem_ctx
{
  // Mathematical constants (dimensionless).
  double pi;
  
  // Physical constants (using normalized code units).
  double epsilon0; // Permittivity of free space.
  double mu0; // Permeability of free space.
  double mass_ion; // Ion mass.
  double charge_ion; // Ion charge.
  double mass_elc; // Electron mass.
  double charge_elc; // Electron charge.
  double Ti_over_Te; // Ion temperature / electron temperature.
  double lambda; // Wavelength.
  double n0; // Reference number density.
  double nb_over_n0; // Background number density / reference number density.
  double B0; // Reference magnetic field strength.
  double beta; // Plasma beta.
  
  // Derived physical quantities (using normalized code units).
  double psi0; // Reference magnetic scalar potential.

  double Ti_frac; // Fraction of total temperature from ions.
  double Te_frac; // Fraction of total temperature from electrons.
  double T_tot; // Total temperature.

  // Simulation parameters.
  int Nx; // Cell count (x-direction).
  int Ny; // Cell count (y-direction).
  double Lx; // Domain size (x-direction).
  double Ly; // Domain size (y-direction).
  double k0; // Closure parameter.
  double cfl_frac; // CFL coefficient.
  double t_end; // Final simulation time.
};

struct gem_ctx
create_ctx(void)
{
  // Mathematical constants (dimensionless).
  double pi = 3.141592653589793238462643383279502884;

  // Physical constants (using normalized code units).
  double epsilon0 = 1.0; // Permittivity of free space.
  double mu0 = 1.0; // Permeability of free space.
  double mass_ion = 1.0; // Ion mass.
  double charge_ion = 1.0; // Ion charge.
  double mass_elc = 1.0 / 25.0; // Electron mass.
  double charge_elc = -1.0; // Electron charge.
  double Ti_over_Te = 5.0; // Ion temperature / electron temperature.
  double lambda = 0.5; // Wavelength
  double n0 = 1.0; // Reference number density.
  double nb_over_n0 = 0.2; // Background number density / reference number density.
  double B0 = 0.1; // Reference magnetic field strength.
  double beta = 1.0; // Plasma beta.
  
  // Derived physical quantities (using normalized code units).
  double psi0 = 0.1 * B0; // Reference magnetic scalar potential.

  double Ti_frac = Ti_over_Te / (1.0 + Ti_over_Te); // Fraction of total temperature from ions.
  double Te_frac = 1.0 / (1.0 + Ti_over_Te); // Fraction of total temperature from electrons.
  double T_tot = beta * (B0 * B0) / 2.0 / n0; // Total temperature;

  // Simulation parameters.
  int Nx = 128; // Cell count (x-direction).
  int Ny = 64; // Cell count (y-direction).
  double Lx = 25.6; // Domain size (x-direction).
  double Ly = 12.8; // Domain size (y-direction).
  double k0 = 5.0; // Closure parameter.
  double cfl_frac = 1.0; // CFL coefficient.
  double t_end = 250.0; // Final simulation time.

  struct gem_ctx ctx = {
    .pi = pi,
    .epsilon0 = epsilon0,
    .mu0 = mu0,
    .mass_ion = mass_ion,
    .charge_ion = charge_ion,
    .mass_elc = mass_elc,
    .charge_elc = charge_elc,
    .Ti_over_Te = Ti_over_Te,
    .lambda = lambda,
    .n0 = n0,
    .nb_over_n0 = nb_over_n0,
    .B0 = B0,
    .beta = beta,
    .psi0 = psi0,
    .Ti_frac = Ti_frac,
    .Te_frac = Te_frac,
    .T_tot = T_tot,
    .Nx = Nx,
    .Ny = Ny,
    .Lx = Lx,
    .Ly = Ly,
    .k0 = k0,
    .cfl_frac = cfl_frac,
    .t_end = t_end,
  };

  return ctx;
}

void
evalElcInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  double x = xn[0], y = xn[1];
  struct gem_ctx *app = ctx;

  double mass_elc = app -> mass_elc;
  double charge_elc = app -> charge_elc;
  double lambda = app -> lambda;
  double n0 = app -> n0;
  double nb_over_n0 = app -> nb_over_n0;
  double B0 = app -> B0;
  double beta = app -> beta;

  double Te_frac = app -> Te_frac;
  double T_tot = app -> T_tot;

  double sech_sq = (1.0 / cosh(y / lambda)) * (1.0 / cosh(y / lambda)); // Hyperbolic secant squared.

  double n = n0 * (sech_sq + nb_over_n0); // Total number density.
  double Jz = -(B0 / lambda) * sech_sq; // Total current density (z-direction).

  double rhoe = n * mass_elc; // Electron mass density.
  double momze = (mass_elc / charge_elc) * Jz * Te_frac; // Electron momentum density (z-direction).
  double pre = n * T_tot * Te_frac; // Electron pressure (scalar).

  // Set electron mass density.
  fout[0] = rhoe;
  // Set electron momentum density.
  fout[1] = 0.0; fout[2] = 0.0; fout[3] = momze;
  // Set electron pressure tensor.
  fout[4] = pre; fout[5] = 0.0; fout[6] = 0.0;
  fout[7] = pre; fout[8] = 0.0; fout[9] = pre + momze * momze / rhoe;
}

void
evalIonInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  double x = xn[0], y = xn[1];
  struct gem_ctx *app = ctx;

  double mass_ion = app -> mass_ion;
  double charge_ion = app -> charge_ion;
  double lambda = app -> lambda;
  double n0 = app -> n0;
  double nb_over_n0 = app -> nb_over_n0;
  double B0 = app -> B0;
  double beta = app -> beta;

  double Ti_frac = app -> Ti_frac;
  double T_tot = app -> T_tot;

  double sech_sq = (1.0 / cosh(y / lambda)) * (1.0 / cosh(y / lambda)); // Hyperbolic secant squared.

  double n = n0 * (sech_sq + nb_over_n0); // Total number density.
  double Jz = -(B0 / lambda) * sech_sq; // Total current density (z-direction).

  double rhoi = n * mass_ion; // Ion mass density.
  double momzi = (mass_ion / charge_ion) * Jz * Ti_frac; // Ion momentum density (z-direction).
  double pri = n * T_tot * Ti_frac; // Ion pressure (scalar).

  // Set ion mass density.
  fout[0] = rhoi;
  // Set ion momentum density.
  fout[1] = 0.0; fout[2] = 0.0; fout[3] = momzi;
  // Set ion pressure tensor.
  fout[4] = pri; fout[5] = 0.0; fout[6] = 0.0;
  fout[7] = pri; fout[8] = 0.0; fout[9] = pri + momzi * momzi / rhoi;
}

void
evalFieldInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  double x = xn[0], y = xn[1];
  struct gem_ctx *app = ctx;

  double pi = app -> pi;

  double lambda = app -> lambda;
  double B0 = app -> B0;

  double psi0 = app -> psi0;

  double Lx = app -> Lx;
  double Ly = app -> Ly;

  double Bxb = B0 * tanh(y / lambda); // Total magnetic field strength.
  double Bx = Bxb - psi0 * (pi / Ly) * cos(2.0 * pi * x / Lx) * sin(pi * y / Ly); // Total magnetic field (x-direction).
  double By = psi0 * (2.0 * pi / Lx) * sin(2.0 * pi * x / Lx) * cos(pi * y / Ly); // Total magnetic field (y-direction).
  double Bz = 0.0; // Total magnetic field (z-direction).

  // Set electric field.
  fout[0] = 0.0, fout[1] = 0.0; fout[2] = 0.0;
  // Set magnetic field.
  fout[3] = Bx, fout[4] = By; fout[5] = Bz;
  // Set correction potentials.
  fout[6] = 0.0; fout[7] = 0.0;
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

  struct gem_ctx ctx = create_ctx(); // Context for initialization functions.

  int NX = APP_ARGS_CHOOSE(app_args.xcells[0], ctx.Nx);
  int NY = APP_ARGS_CHOOSE(app_args.xcells[1], ctx.Ny);  

  // Electron/ion equations.
  struct gkyl_wv_eqn *elc_ten_moment = gkyl_wv_ten_moment_new(ctx.k0);
  struct gkyl_wv_eqn *ion_ten_moment = gkyl_wv_ten_moment_new(ctx.k0);
  
  struct gkyl_moment_species elc = {
    .name = "elc",
    .charge = ctx.charge_elc, .mass = ctx.mass_elc,
    .equation = elc_ten_moment,
    .evolve = true,
    .init = evalElcInit,
    .ctx = &ctx,

    .bcy = { GKYL_SPECIES_REFLECT, GKYL_SPECIES_REFLECT },
  };
  struct gkyl_moment_species ion = {
    .name = "ion",
    .charge = ctx.charge_ion, .mass = ctx.mass_ion,
    .equation = ion_ten_moment,
    .evolve = true,
    .init = evalIonInit,
    .ctx = &ctx,

    .bcy = { GKYL_SPECIES_REFLECT, GKYL_SPECIES_REFLECT },    
  };

  int nrank = 1; // Number of processes in simulation.
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
  {
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);
  }
#endif

  // Create global range.
  int cells[] = { NX, NY };
  struct gkyl_range globalr;
  gkyl_create_global_range(2, cells, &globalr);

  // Create decomposition.
  int cuts[] = { 1, 1 };
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
  {
    cuts[0] = app_args.cuts[0];
    cuts[1] = app_args.cuts[1];
  }
#endif

  struct gkyl_rect_decomp *decomp = gkyl_rect_decomp_new_from_cuts(2, cuts, &globalr);

  // Construct communicator for use in app.
  struct gkyl_comm *comm;
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
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
  comm = gklyl_null_comm_inew( &(struct gkyl_null_comp_inp)
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

  int ncuts = cuts[0] * cuts[1];
  if (ncuts != comm_size)
  {
    if (my_rank == 0)
    {
      fprintf(stderr, "*** Number of ranks, %d, does not match total cuts, %d!\n", comm_size, ncuts);
    }
    goto mpifinalize;
  }

  // Moment app.
  struct gkyl_moment app_inp = {
    .name = "10m_gem",

    .ndim = 2,
    .lower = { -0.5 * ctx.Lx, -0.5 * ctx.Ly},
    .upper = { 0.5 * ctx.Lx, 0.5 * ctx.Ly},
    .cells = { NX, NY },

    .num_periodic_dir = 1,
    .periodic_dirs = { 0 },
    .cfl_frac = ctx.cfl_frac,

    .num_species = 2,
    .species = { elc, ion },

    .field = {
      .epsilon0 = ctx.epsilon0, .mu0 = ctx.mu0,
      .mag_error_speed_fact = 1.0,
      
      .evolve = true,
      .init = evalFieldInit,
      .ctx = &ctx,
      
      .bcy = { GKYL_FIELD_PEC_WALL, GKYL_FIELD_PEC_WALL },
    },

    .has_low_inp = true,
    .low_inp = {
      .local_range = decomp -> ranges[my_rank],
      .comm = comm
    }
  };

  // Create app object.
  gkyl_moment_app *app = gkyl_moment_app_new(&app_inp);

  // Initial and final simulation times.
  double t_curr = 0.0, t_end = ctx.t_end;

  // Initialize simulation.
  gkyl_moment_app_apply_ic(app, t_curr);
  gkyl_moment_app_write(app, t_curr, 0);

  // Compute estimate of maximum stable time-step.
  double dt = gkyl_moment_app_max_dt(app);

  long step = 1;
  while ((t_curr < t_end) && (step <= app_args.num_steps))
  {
    gkyl_moment_app_cout(app, stdout, "Taking time-step %ld at t = %g ...", step, t_curr);
    struct gkyl_update_status status = gkyl_moment_update(app, dt);
    gkyl_moment_app_cout(app, stdout, " dt = %g\n", status.dt_actual);
    
    if (!status.success)
    {
      gkyl_moment_app_cout(app, stdout, "** Update method failed! Aborting simulation ....\n");
      break;
    }

    t_curr += status.dt_actual;
    dt = status.dt_suggested;

    step += 1;
  }

  gkyl_moment_app_write(app, t_curr, 1);
  gkyl_moment_app_stat_write(app);

  struct gkyl_moment_stat stat = gkyl_moment_app_stat(app);

  gkyl_moment_app_cout(app, stdout, "\n");
  gkyl_moment_app_cout(app, stdout, "Number of update calls %ld\n", stat.nup);
  gkyl_moment_app_cout(app, stdout, "Number of failed time-steps %ld\n", stat.nfail);
  gkyl_moment_app_cout(app, stdout, "Species updates took %g secs\n", stat.species_tm);
  gkyl_moment_app_cout(app, stdout, "Field updates took %g secs\n", stat.field_tm);
  gkyl_moment_app_cout(app, stdout, "Source updates took %g secs\n", stat.sources_tm);
  gkyl_moment_app_cout(app, stdout, "Total updates took %g secs\n", stat.total_tm);

  // Free resources after simulation completion.
  gkyl_wv_eqn_release(elc_ten_moment);
  gkyl_wv_eqn_release(ion_ten_moment);
  gkyl_rect_decomp_release(decomp);
  gkyl_comm_release(comm);
  gkyl_moment_app_release(app);  
  
mpifinalize:
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
  {
    MPI_Finalize();
  }
#endif
  
  return 0;
}