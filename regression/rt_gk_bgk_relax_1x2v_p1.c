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

struct bgk_relax_ctx
{
  // Mathematical constants (dimensionless).
  double pi;

  // Physical constants (using normalized code units).
  double mass; // Top hat/bump mass.
  double charge; // Top hat/bump charge.

  double B0; // Reference magnetic field strength.
  double n0; // Reference number density.
  double u0; // Reference velocity.
  double vt; // Top hat thermal velocity.
  double nu; // Collision frequency.

  double ab; // Bump Maxwellian amplitude.
  double sb; // Bump Maxwellian softening factor, to avoid divergence.
  double vtb; // Bump Maxwellian thermal velocity.

  // Derived physical quantities (using normalized code units).
  double ub; // Bump location (in velocity space).

  // Simulation parameters.
  int Nz; // Cell count (configuration space: z-direction).
  int Nvpar; // Cell count (velocity space: parallel velocity direction).
  int Nmu; // Cell count (velocity space: magnetic moment direction).
  double Lz; // Domain size (configuration space: z-direction).
  double vpar_max; // Domain boundary (velocity space: parallel velocity direction).
  double mu_max; // Domain boundary (velocity space: magnetic moment direction).

  double t_end; // Final simulation time.
  int num_frames; // Number of output frames.
  int int_diag_calc_num; // Number of integrated diagnostics computations (=INT_MAX for every step).
  double dt_failure_tol; // Minimum allowable fraction of initial time-step.
  int num_failures_max; // Maximum allowable number of consecutive small time-steps.
};

struct bgk_relax_ctx
create_ctx(void)
{
  // Mathematical constants (dimensionless).
  double pi = M_PI;

  // Physical constants (using normalized code units).
  double mass = 1.0; // Top hat/bump mass.
  double charge = 1.0; // Top hat/bump charge.

  double B0 = 1.0; // Reference magnetic field strength.
  double n0 = 1.0; // Reference number density.
  double u0 = 0.0; // Reference velocity.
  double vt = 1.0 / 3.0; // Top hat Maxwellian thermal velocity.
  double nu = 0.01; // Collision frequency.

  double ab = sqrt(0.1); // Bump Maxwellian amplitude.
  double sb = 0.12; // Bump Maxwellian softening factor, to avoid divergence.
  double vtb = 1.0; // Bump Maxwellian thermal velocity.

  // Derived physical quantities (using normalized code units).
  double ub = 4.0 * sqrt((pow(3.0 * vt / 2.0, 2.0)) / 3.0); // Bump location (in velocity space).

  // Simulation parameters.
  int Nz = 2; // Cell count (configuration space: z-direction).
  int Nvpar = 16; // Cell count (velocity space: parallel velocity direction).
  int Nmu = 8; // Cell count (velocity space: magnetic moment direction).
  double Lz = 1.0; // Domain size (configuration space: z-direction).
  double vpar_max = 8.0 * vt; // Domain boundary (velocity space: parallel velocity direction).
  double mu_max = 12.0 * (vt * vt) / 2.0 / B0; // Domain boundary (velocity space: magnetic moment direction).

  double t_end = 100.0; // Final simulation time.
  int num_frames = 1; // Number of output frames.
  int int_diag_calc_num = num_frames*100;
  double dt_failure_tol = 1.0e-4; // Minimum allowable fraction of initial time-step.
  int num_failures_max = 20; // Maximum allowable number of consecutive small time-steps.
  
  struct bgk_relax_ctx ctx = {
    .pi = pi,
    .mass = mass,
    .charge = charge,
    .B0 = B0,
    .n0 = n0,
    .u0 = u0,
    .vt = vt,
    .nu = nu,
    .ab = ab,
    .sb = sb,
    .vtb = vtb,
    .ub = ub,
    .Nz = Nz,
    .Nvpar = Nvpar,
    .Nmu = Nmu,
    .Lz = Lz,
    .vpar_max = vpar_max,
    .mu_max = mu_max,
    .t_end = t_end,
    .num_frames = num_frames,
    .int_diag_calc_num = int_diag_calc_num,
    .dt_failure_tol = dt_failure_tol,
    .num_failures_max = num_failures_max,
  };

  return ctx;
}

void
evalTopHatInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct bgk_relax_ctx *app = ctx;
  double v = xn[1];

  double n0 = app->n0;
  double vt = app->vt;

  double v0 = sqrt(3.0) * vt;

  double dist = 0.0;

  if (fabs(v) < v0) {
    dist = n0 / 2.0 / v0;
  }
  else {
    dist = 0.0;
  }

  // Set distribution function.
  fout[0] = dist;
}

void
evalBumpInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct bgk_relax_ctx *app = ctx;
  double v = xn[1], mu = xn[2];

  double pi = app->pi;

  double B0 = app->B0;
  double n0 = app->n0;
  double u0 = app->u0;
  double vt = app->vt;

  double ab = app->ab;
  double sb = app->sb;
  double vtb = app->vtb;

  double ub = app->ub;

  double v_sq = ((v - u0) / (sqrt(2.0) * vt)) * ((v - u0) / (sqrt(2.0) * vt)) + mu * B0;
  double vb_sq = ((v - u0) / (sqrt(2.0) * vtb)) * ((v - u0) / (sqrt(2.0) * vtb)) + mu * B0;

  // Set distribution function.
  fout[0] = (n0 / sqrt(2.0 * pi * vt)) * exp(-v_sq) + (n0 / sqrt(2.0 * pi * vtb)) * exp(-vb_sq) * (ab * ab) / ((v - ub) * (v - ub) + sb * sb);
}

void
evalNuInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct bgk_relax_ctx *app = ctx;

  double nu = app->nu;

  // Set collision frequency.
  fout[0] = nu;
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
  struct bgk_relax_ctx *app = ctx;
  
  double B0 = app->B0;

  // Set magnetic field strength.
  fout[0] = B0;
}

void
calc_integrated_diagnostics(struct gkyl_tm_trigger* iot, gkyl_gyrokinetic_app* app, double t_curr, bool force_calc)
{
  if (gkyl_tm_trigger_check_and_bump(iot, t_curr) || force_calc) {
    gkyl_gyrokinetic_app_calc_field_energy(app, t_curr);
    gkyl_gyrokinetic_app_calc_integrated_mom(app, t_curr);
  }
}

void
write_data(struct gkyl_tm_trigger* iot, gkyl_gyrokinetic_app* app, double t_curr, bool force_write)
{
  if (gkyl_tm_trigger_check_and_bump(iot, t_curr) || force_write) {
    int frame = force_write? iot->curr : iot->curr -1;

    gkyl_gyrokinetic_app_write(app, t_curr, frame);

    gkyl_gyrokinetic_app_calc_mom(app);
    gkyl_gyrokinetic_app_write_mom(app, t_curr, frame);
    gkyl_gyrokinetic_app_write_source_mom(app, t_curr, frame);

    gkyl_gyrokinetic_app_calc_field_energy(app, t_curr);
    gkyl_gyrokinetic_app_write_field_energy(app);

    gkyl_gyrokinetic_app_calc_integrated_mom(app, t_curr);
    gkyl_gyrokinetic_app_write_integrated_mom(app);
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

  struct bgk_relax_ctx ctx = create_ctx(); // Context for initialization functions.

  int NZ = APP_ARGS_CHOOSE(app_args.xcells[0], ctx.Nz);
  int NVPAR = APP_ARGS_CHOOSE(app_args.vcells[0], ctx.Nvpar);
  int NMU = APP_ARGS_CHOOSE(app_args.vcells[1], ctx.Nmu);

  int nrank = 1; // Number of processors in simulation.
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi) {
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);
  }
#endif  

  // Create global range.
  int ccells[] = { NZ };
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

  int my_rank, comm_size;
  gkyl_comm_get_rank(comm, &my_rank);
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

  // Top hat species.
  struct gkyl_gyrokinetic_species square = {
    .name = "square",
    .charge = ctx.charge, .mass = ctx.mass,
    .lower = { -ctx.vpar_max, 0.0 },
    .upper = { ctx.vpar_max, ctx.mu_max }, 
    .cells = { NVPAR, NMU },
    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_FUNC,
      .func = evalTopHatInit,
      .ctx_func = &ctx,
    },
    .collisions =  {
      .collision_id = GKYL_BGK_COLLISIONS,
      .normNu = false,
      .self_nu = evalNuInit,
      .ctx = &ctx,
      .num_cross_collisions = 1,
      .collide_with = { "bump" },
    },
    
    .num_diag_moments = 7,
    .diag_moments = { "M0", "M1", "M2", "M2par", "M2perp", "M3par", "M3perp" },
  };

  // Bump species.
  struct gkyl_gyrokinetic_species bump = {
    .name = "bump",
    .charge = ctx.charge, .mass = ctx.mass,
    .lower = { -ctx.vpar_max, 0.0 },
    .upper = { ctx.vpar_max, ctx.mu_max }, 
    .cells = { NVPAR, NMU },
    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_FUNC,
      .func = evalBumpInit,
      .ctx_func = &ctx,
    },
    .collisions =  {
      .collision_id = GKYL_BGK_COLLISIONS,
      .normNu = false,
      .self_nu = evalNuInit,
      .ctx = &ctx,
      .num_cross_collisions = 1,
      .collide_with = { "square" },
    },

    .num_diag_moments = 7,
    .diag_moments = { "M0", "M1", "M2", "M2par", "M2perp", "M3par", "M3perp" },
  };

  // Field.
  struct gkyl_gyrokinetic_field field = {
    .gkfield_id = GKYL_GK_FIELD_BOLTZMANN,
    .electron_mass = ctx.mass,
    .electron_charge = ctx.charge,
    .electron_temp = ctx.vt,
    .bmag_fac = ctx.B0, 
    .fem_parbc = GKYL_FEM_PARPROJ_NONE, 
  };

  // GK app.
  struct gkyl_gk app_inp = {
    .name = "gk_bgk_relax_1x2v_p1",

    .cdim = 1, .vdim = 2,
    .lower = { 0.0 },
    .upper = { ctx.Lz },
    .cells = { NZ },
    .poly_order = 1,
    .basis_type = app_args.basis_type,

    .geometry = {
      .geometry_id = GKYL_MAPC2P,
      .mapc2p = mapc2p,
      .c2p_ctx = &ctx,
      .bmag_func = bmag_func,
      .bmag_ctx = &ctx,
    },

    .num_periodic_dir = 1,
    .periodic_dirs = { 0 },

    .num_species = 2,
    .species = { square, bump },
    .skip_field = true,
    .field = field,

    .use_gpu = app_args.use_gpu,

    .has_low_inp = true,
    .low_inp = {
      .local_range = decomp->ranges[my_rank],
      .comm = comm
    }
  };

  // Create app object.
  gkyl_gyrokinetic_app *app = gkyl_gyrokinetic_app_new(&app_inp);

  // Initial and final simulation times.
  double t_curr = 0.0, t_end = ctx.t_end;

  // Create triggers for IO.
  int num_frames = ctx.num_frames, num_int_diag_calc = ctx.int_diag_calc_num;
  struct gkyl_tm_trigger io_trig_int_diag = { .dt = t_end/GKYL_MAX2(num_frames, num_int_diag_calc) };
  struct gkyl_tm_trigger io_trig_write = { .dt = t_end/num_frames };

  // Initialize simulation.
  gkyl_gyrokinetic_app_apply_ic(app, t_curr);
  calc_integrated_diagnostics(&io_trig_int_diag, app, t_curr, false);
  write_data(&io_trig_write, app, t_curr, false);

  // Compute initial guess of maximum stable time-step.
  double dt = t_end - t_curr;

  // Initialize small time-step check.
  double dt_init = -1.0, dt_failure_tol = ctx.dt_failure_tol;
  int num_failures = 0, num_failures_max = ctx.num_failures_max;

  long step = 1;
  while ((t_curr < t_end) && (step <= app_args.num_steps)) {
    gkyl_gyrokinetic_app_cout(app, stdout, "Taking time-step %ld at t = %g ...", step, t_curr);
    struct gkyl_update_status status = gkyl_gyrokinetic_update(app, dt);
    gkyl_gyrokinetic_app_cout(app, stdout, " dt = %g\n", status.dt_actual);

    if (!status.success) {
      gkyl_gyrokinetic_app_cout(app, stdout, "** Update method failed! Aborting simulation ....\n");
      break;
    }

    t_curr += status.dt_actual;
    dt = status.dt_suggested;

    calc_integrated_diagnostics(&io_trig_int_diag, app, t_curr, false);
    write_data(&io_trig_write, app, t_curr, false);

    if (dt_init < 0.0) {
      dt_init = status.dt_actual;
    }
    else if (status.dt_actual < dt_failure_tol * dt_init) {
      num_failures += 1;

      gkyl_gyrokinetic_app_cout(app, stdout, "WARNING: Time-step dt = %g", status.dt_actual);
      gkyl_gyrokinetic_app_cout(app, stdout, " is below %g*dt_init ...", dt_failure_tol);
      gkyl_gyrokinetic_app_cout(app, stdout, " num_failures = %d\n", num_failures);
      if (num_failures >= num_failures_max) {
        gkyl_gyrokinetic_app_cout(app, stdout, "ERROR: Time-step was below %g*dt_init ", dt_failure_tol);
        gkyl_gyrokinetic_app_cout(app, stdout, "%d consecutive times. Aborting simulation ....\n", num_failures_max);
        calc_integrated_diagnostics(&io_trig_int_diag, app, t_curr, true);
        write_data(&io_trig_write, app, t_curr, true);
        break;
      }
    }
    else {
      num_failures = 0;
    }

    step += 1;
  }

  calc_integrated_diagnostics(&io_trig_int_diag, app, t_curr, false);
  write_data(&io_trig_write, app, t_curr, false);
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