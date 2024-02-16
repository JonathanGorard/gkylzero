// 2D Riemann (quadrant) problem, using the Roe-type approximate Riemann solver, for the 5-moment (Euler) equations.
// Input parameters match the initial conditions in Section 4.3, Case 3, with final time set to t = 0.8 rather than t = 0.3, from the article:
// R. Liska and B. Wendroff (2003), "Comparison of Several Difference Schemes on 1D and 2D Test Problems for the Euler Equations",
// SIAM Journal on Scientific Computing, Volume 25 (3): 995-1017.
// https://epubs.siam.org/doi/10.1137/S1064827502402120

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gkyl_alloc.h>
#include <gkyl_moment.h>
#include <gkyl_util.h>
#include <gkyl_wv_euler.h>

#include <gkyl_null_comm.h>

#ifdef GKYL_HAVE_MPI
#include <mpi.h>
#include <gkyl_mpi_comm.h>
#endif

#include <rt_arg_parse.h>

struct euler_riem_2d_roe_ctx
{
  // Physical constants (using normalized code units).
  double gas_gamma; // Adiabatic index.

  double rho_ul; // Upper left fluid mass density.
  double u_ul; // Upper left fluid x-velocity.
  double v_ul; // Upper left fluid y-velocity.
  double p_ul; // Upper left fluid pressure.

  double rho_ur; // Upper right fluid mass density.
  double u_ur; // Upper right fluid x-velocity.
  double v_ur; // Upper right fluid y-velocity.
  double p_ur; // Upper left fluid pressure.
  
  double rho_ll; // Lower left fluid mass density.
  double u_ll; // Lower left fluid x-velocity.
  double v_ll; // Lower left fluid y-velocity.
  double p_ll; // Lower left fluid pressure.

  double rho_lr; // Lower right fluid mass density.
  double u_lr; // Lower right fluid x-velocity.
  double v_lr; // Lower right fluid y-velocity.
  double p_lr; // Lower right fluid pressure.

  // Simulation parameters.
  int Nx; // Cell count (x-direction).
  int Ny; // Cell count (y-direction).
  double Lx; // Domain size (x-direction).
  double Ly; // Domain size (y-direction).
  double cfl_frac; // CFL coefficient.
  double t_end; // Final simulation time.

  double loc; // Fluid boundaries (both x and y coordinates).
};

struct euler_riem_2d_roe_ctx
create_ctx(void)
{
  // Physical constants (using normalized code units).
  double gas_gamma = 1.4; // Adiabatic index.

  double rho_ul = 0.5323; // Upper-left fluid mass density.
  double u_ul = 1.206; // Upper-left fluid x-velocity.
  double v_ul = 0.0; // Upper-left fluid y-velocity.
  double p_ul = 0.3; // Upper-left fluid pressure.

  double rho_ur = 1.5; // Upper-right fluid mass density.
  double u_ur = 0.0; // Upper-right fluid x-velocity.
  double v_ur = 0.0; // Upper-right fluid y-velocity.
  double p_ur = 1.5; // Upper-right fluid pressure.
  
  double rho_ll = 0.138; // Lower-left fluid mass density.
  double u_ll = 1.206; // Lower-left fluid x-velocity.
  double v_ll = 1.206; // Lower-left fluid y-velocity.
  double p_ll = 0.029; // Lower-left fluid pressure.

  double rho_lr = 0.5323; // Lower-right fluid mass density.
  double u_lr = 0.0; // Lower-right fluid x-velocity.
  double v_lr = 1.206; // Lower-right fluid y-velocity.
  double p_lr = 0.3; // Lower-right fluid pressure.

  // Simulation parameters.
  int Nx = 200; // Cell count (x-direction).
  int Ny = 200; // Cell count (y-direction).
  double Lx = 1.0; // Domain size (x-direction).
  double Ly = 1.0; // Domain size (y-direction).
  double cfl_frac = 0.95; // CFL coefficient.
  double t_end = 0.8; // Final simulation time.

  double loc = 0.8; // Fluid boundaries (both x and y coordinates).

  struct euler_riem_2d_roe_ctx ctx = {
    .gas_gamma = gas_gamma,
    .rho_ul = rho_ul,
    .u_ul = u_ul,
    .v_ul = v_ul,
    .p_ul = p_ul,
    .rho_ur = rho_ur,
    .u_ur = u_ur,
    .v_ur = v_ur,
    .p_ur = p_ur,
    .rho_ll = rho_ll,
    .u_ll = u_ll,
    .v_ll = v_ll,
    .p_ll = p_ll,
    .rho_lr = rho_lr,
    .u_lr = u_lr,
    .v_lr = v_lr,
    .p_lr = p_lr,
    .Nx = Nx,
    .Ny = Ny,
    .Lx = Lx,
    .Ly = Ly,
    .cfl_frac = cfl_frac,
    .t_end = t_end,
    .loc = loc,
  };

  return ctx;
}

void
evalEulerInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  double x = xn[0], y = xn[1];
  struct euler_riem_2d_roe_ctx *app = ctx;

  double gas_gamma = app -> gas_gamma;

  double rho_ul = app -> rho_ul;
  double u_ul = app -> u_ul;
  double v_ul = app -> v_ul;
  double p_ul = app -> p_ul;

  double rho_ur = app -> rho_ur;
  double u_ur = app -> u_ur;
  double v_ur = app -> v_ur;
  double p_ur = app -> p_ur;

  double rho_ll = app -> rho_ll;
  double u_ll = app -> u_ll;
  double v_ll = app -> v_ll;
  double p_ll = app -> p_ll;

  double rho_lr = app -> rho_lr;
  double u_lr = app -> u_lr;
  double v_lr = app -> v_lr;
  double p_lr = app -> p_lr;

  double loc = app -> loc;

  double rho = 0.0;
  double u = 0.0;
  double v = 0.0;
  double p = 0.0;

  if (y > loc)
  {
    if (x < loc)
    {
        rho = rho_ul; // Fluid mass density (upper-left).
        u = u_ul; // Fluid x-velocity (upper-left).
        v = v_ul; // Fluid y-velocity (upper-left).
        p = p_ul; // Fluid pressure (upper-left).
    }
    else
    {
        rho = rho_ur; // Fluid mass density (upper-right).
        u = u_ur; // Fluid x-velocity (upper-right).
        v = v_ur; // Fluid y-velocity (upper-right).
        p = p_ur; // Fluid pressure (upper-right).
    }
  }
  else
  {
    if (x < loc)
    {
        rho = rho_ll; // Fluid mass density (lower-left).
        u = u_ll; // Fluid x-velocity (lower-left).
        v = v_ll; // Fluid y-velocity (lower-left).
        p = p_ll; // Fluid pressure (lower-left).
    }
    else
    {
        rho = rho_lr; // Fluid mass density (lower-right).
        u = u_lr; // Fluid x-velocity (lower-right).
        v = v_lr; // Fluid y-velocity (lower-right).
        p = p_lr; // Fluid pressure (lower-right).
    }
  }
  
  // Set fluid mass density.
  fout[0] = rho;
  // Set fluid momentum density.
  fout[1] = rho * u; fout[2] = rho * v; fout[3] = 0.0;
  // Set fluid total energy density.
  fout[4] = p / (gas_gamma - 1.0) + 0.5 * rho * (u * u + v * v);
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

  struct euler_riem_2d_roe_ctx ctx = create_ctx(); // Context for initialization functions.

  int NX = APP_ARGS_CHOOSE(app_args.xcells[0], ctx.Nx);
  int NY = APP_ARGS_CHOOSE(app_args.xcells[1], ctx.Ny);

  // Fluid equations.
  struct gkyl_wv_eqn *euler = gkyl_wv_euler_inew(
    &(struct gkyl_wv_euler_inp) {
        .gas_gamma = ctx.gas_gamma,
        .rp_type = WV_EULER_RP_ROE,
        .use_gpu = app_args.use_gpu,
    }
  );

  struct gkyl_moment_species fluid = {
    .name = "euler",
    .equation = euler,
    .evolve = true,
    .init = evalEulerInit,
    .ctx = &ctx,
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
    .name = "euler_riem_2d_roe",

    .ndim = 2,
    .lower = { 0.0, 0.0 },
    .upper = { ctx.Lx, ctx.Ly },
    .cells = { NX, NY },

    .scheme_type = GKYL_MOMENT_WAVE_PROP,
    .mp_recon = app_args.mp_recon,

    .cfl_frac = ctx.cfl_frac,

    .num_species = 1,
    .species = { fluid },

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
  gkyl_wv_eqn_release(euler);
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
}