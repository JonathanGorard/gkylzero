#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gkyl_alloc.h>
#include <gkyl_moment.h>
#include <gkyl_util.h>
#include <gkyl_wv_gr_maxwell_tetrad.h>
#include <gkyl_gr_minkowski.h>
#include <gkyl_gr_blackhole.h>

#include <gkyl_null_comm.h>

#ifdef GKYL_HAVE_MPI
#include <mpi.h>
#include <gkyl_mpi_comm.h>
#endif

#include <rt_arg_parse.h>

struct bz_monopole_slow_ctx
{
  // Mathematical constants (dimensionless).
  double pi;

  // Physical constants (using normalized code units).
  double light_speed; // Speed of light.
  double e_fact; // Factor of speed of light for electric field correction.
  double b_fact; // Factor of speed of light for magnetic field correction.

  double B0; // Reference magnetic field strength.

  // Spacetime parameters (using geometric units).
  double mass; // Mass of the black hole.
  double spin; // Spin of the black hole.

  double pos_x; // Position of the black hole (x-direction).
  double pos_y; // Position of the black hole (y-direction).
  double pos_z; // Position of the black hole (z-direction).

  // Pointer to spacetime metric.
  struct gkyl_gr_spacetime *spacetime;

  // Simulation parameters.
  int Nx; // Cell count (x-direction).
  int Ny; // Cell count (y-direction).
  double Lx; // Domain size (x-direction).
  double Ly; // Domain size (y-direction).
  double cfl_frac; // CFL coefficient.

  double t_end; // Final simulation time.
  int num_frames; // Number of output frames.
  double dt_failure_tol; // Minimum allowable fraction of initial time-step.
  int num_failures_max; // Maximum allowable number of consecutive small time-steps.
};

struct bz_monopole_slow_ctx
create_ctx(void)
{
  // Mathematical constants (dimensionless).
  double pi = M_PI;

  // Physical constants (using normalized code units).
  double light_speed = 1.0; // Speed of light.
  double e_fact = 0.0; // Factor of speed of light for electric field correction.
  double b_fact = 0.0; // Factor of speed of light for magnetic field correction.

  double B0 = 1.0; // Reference magnetic field strength.

  // Spacetime parameters (using geometric units).
  double mass = 1.0; // Mass of the black hole.
  double spin = -0.1; // Spin of the black hole.

  double pos_x = 0.0; // Position of the black hole (x-direction).
  double pos_y = 0.0; // Position of the black hole (y-direction).
  double pos_z = 0.0; // Position of the black hole (z-direction).

  // Pointer to spacetime metric.
  struct gkyl_gr_spacetime *spacetime = gkyl_gr_blackhole_new(false, mass, spin, pos_x, pos_y, pos_z);

  // Simulation parameters.
  int Nx = 256; // Cell count (x-direction).
  int Ny = 256; // Cell count (y-direction).
  double Lx = 10.0; // Domain size (x-direction).
  double Ly = 10.0; // Domain size (y-direction).
  double cfl_frac = 0.95; // CFL coefficient.

  double t_end = 50.0; // Final simulation time.
  int num_frames = 1; // Number of output frames.
  double dt_failure_tol = 1.0e-4; // Minimum allowable fraction of initial time-step.
  int num_failures_max = 20; // Maximum allowable number of consecutive small time-steps.

  struct bz_monopole_slow_ctx ctx = {
    .pi = pi,
    .light_speed = light_speed,
    .e_fact = e_fact,
    .b_fact = b_fact,
    .B0 = B0,
    .mass = mass,
    .spin = spin,
    .pos_x = pos_x,
    .pos_y = pos_y,
    .pos_z = pos_z,
    .spacetime = spacetime,
    .Nx = Nx,
    .Ny = Ny,
    .Lx = Lx,
    .Ly = Ly,
    .cfl_frac = cfl_frac,
    .t_end = t_end,
    .num_frames = num_frames,
    .dt_failure_tol = dt_failure_tol,
    .num_failures_max = num_failures_max,
  };

  return ctx;
}

void
evalGRMaxwellInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  double x = xn[0], y = xn[1];
  struct bz_monopole_slow_ctx *app = ctx;

  struct gkyl_gr_spacetime *spacetime = app->spacetime;

  double pi = app->pi;
  double B0 = app->B0;

  double r = sqrt((x * x) + (y * y));
  double phi = 0.5 * pi;

  double theta = 0.0;
  if (fabs(y) < pow(10.0, -6.0)) {
    theta = 0.5 * pi;
  }
  else {
    theta = atan(x / y);
  }

  double Br = B0 * sin(theta);
  
  double spatial_det, lapse;
  double *shift = gkyl_malloc(sizeof(double[3]));
  bool in_excision_region;

  double **spatial_metric = gkyl_malloc(sizeof(double*[3]));
  for (int i = 0; i < 3; i++) {
    spatial_metric[i] = gkyl_malloc(sizeof(double[3]));
  }

  spacetime->spatial_metric_det_func(spacetime, 0.0, x, y, 0.0, &spatial_det);
  spacetime->lapse_function_func(spacetime, 0.0, x, y, 0.0, &lapse);
  spacetime->shift_vector_func(spacetime, 0.0, x, y, 0.0, &shift);
  spacetime->excision_region_func(spacetime, 0.0, x, y, 0.0, &in_excision_region);
  
  spacetime->spatial_metric_tensor_func(spacetime, 0.0, x, y, 0.0, &spatial_metric);
  
  double B_r = B0 * sin(theta) / sqrt(spatial_det);

  double Bx = sin(theta) * sin(phi) * B_r;
  double By = sin(theta) * cos(phi) * B_r;
  double Bz = cos(theta) * B_r;
  
  // Set electric field.
  fout[0] = 0.0; fout[1] = 0.0; fout[2] = 0.0;
  // Set magnetic field.
  fout[3] = Bx; fout[4] = By; fout[5] = Bz;
  // Set correction potentials.
  fout[6] = 0.0; fout[7] = 0.0;

  // Set lapse gauge variable.
  fout[8] = lapse;
  // Set shift gauge variables.
  fout[9] = shift[0]; fout[10] = shift[1]; fout[11] = shift[2];

  // Set spatial metric tensor.
  fout[12] = spatial_metric[0][0]; fout[13] = spatial_metric[0][1]; fout[14] = spatial_metric[0][2];
  fout[15] = spatial_metric[1][0]; fout[16] = spatial_metric[1][1]; fout[17] = spatial_metric[1][2];
  fout[18] = spatial_metric[2][0]; fout[19] = spatial_metric[2][1]; fout[20] = spatial_metric[2][2];

  // Set excision boundary conditions.
  if (in_excision_region) {
    for (int i = 0; i < 21; i++) {
      fout[i] = 0.0;
    }

    fout[21] = -1.0;
  }
  else {
    fout[21] = 1.0;
  }

  // Free all tensorial quantities.
  for (int i = 0; i < 3; i++) {
    gkyl_free(spatial_metric[i]);
  }
  gkyl_free(spatial_metric);
  gkyl_free(shift);
}

void
write_data(struct gkyl_tm_trigger* iot, gkyl_moment_app* app, double t_curr, bool force_write)
{
  if (gkyl_tm_trigger_check_and_bump(iot, t_curr)) {
    int frame = iot->curr - 1;
    if (force_write) {
      frame = iot->curr;
    }

    gkyl_moment_app_write(app, t_curr, frame);
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

  struct bz_monopole_slow_ctx ctx = create_ctx(); // Context for initialization functions.

  int NX = APP_ARGS_CHOOSE(app_args.xcells[0], ctx.Nx);
  int NY = APP_ARGS_CHOOSE(app_args.xcells[1], ctx.Ny);

  // Field
  struct gkyl_wv_eqn *gr_maxwell_tetrad = gkyl_wv_gr_maxwell_tetrad_new(ctx.light_speed, ctx.e_fact, ctx.b_fact, ctx.spacetime, app_args.use_gpu);

  struct gkyl_moment_species field = {
    .name = "field",
    .equation = gr_maxwell_tetrad,
    .evolve = true,
    .init = evalGRMaxwellInit,
    .force_low_order_flux = true, // Use Lax fluxes.
    .ctx = &ctx,

    .bcx = { GKYL_SPECIES_COPY, GKYL_SPECIES_COPY },
    .bcy = { GKYL_SPECIES_COPY, GKYL_SPECIES_COPY },
  };

  int nrank = 1; // Number of processes in simulation.
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi) {
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);
  }
#endif

  // Create global range.
  int cells[] = { NX, NY };
  int dim = sizeof(cells) / sizeof(cells[0]);
  struct gkyl_range global_r;
  gkyl_create_global_range(dim, cells, &global_r);

  // Create decomposition.
  int cuts[dim];
#ifdef GKYL_HAVE_MPI
  for (int d = 0; d < dim; d++) {
    if (app_args.use_mpi) {
      cuts[d] = app_args.cuts[d];
    }
    else {
      cuts[d] = 1;
    }
  }
#else
  for (int d = 0; d < dim; d++) {
    cuts[d] = 1;
  }
#endif

  struct gkyl_rect_decomp *decomp = gkyl_rect_decomp_new_from_cuts(dim, cuts, &global_r);

  // Construct communicator for use in app.
  struct gkyl_comm *comm;
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi) {
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
  for (int d = 0; d < dim; d++) {
    ncuts *= cuts[d];
  }

  if (ncuts != comm_size) {
    if (my_rank == 0) {
      fprintf(stderr, "*** Number of ranks, %d, does not match total cuts, %d!\n", comm_size, ncuts);
    }
    goto mpifinalize;
  }

  // Moment app.
  struct gkyl_moment app_inp = {
    .name = "gr_bz_monopole_slow_tetrad",

    .ndim = 2,
    .lower = { -0.5 * ctx.Lx, -0.5 * ctx.Ly },
    .upper = { 0.5 * ctx.Lx, 0.5 * ctx.Ly },
    .cells = { NX, NY },

    .cfl_frac = ctx.cfl_frac,

    .num_species = 1,
    .species = { field },

    .has_low_inp = true,
    .low_inp = {
      .local_range = decomp->ranges[my_rank],
      .comm = comm
    }
  };

  // Create app object.
  gkyl_moment_app *app = gkyl_moment_app_new(&app_inp);

  // Initial and final simulation times.
  double t_curr = 0.0, t_end = ctx.t_end;

  // Create trigger for IO.
  int num_frames = ctx.num_frames;
  struct gkyl_tm_trigger io_trig = { .dt = t_end / num_frames };

  // Initialize simulation.
  gkyl_moment_app_apply_ic(app, t_curr);
  write_data(&io_trig, app, t_curr, false);

  // Compute estimate of maximum stable time-step.
  double dt = gkyl_moment_app_max_dt(app);

  // Initialize small time-step check.
  double dt_init = -1.0, dt_failure_tol = ctx.dt_failure_tol;
  int num_failures = 0, num_failures_max = ctx.num_failures_max;

  long step = 1;
  while ((t_curr < t_end) && (step <= app_args.num_steps)) {
    gkyl_moment_app_cout(app, stdout, "Taking time-step %ld at t = %g ...", step, t_curr);
    struct gkyl_update_status status = gkyl_moment_update(app, dt);
    gkyl_moment_app_cout(app, stdout, " dt = %g\n", status.dt_actual);
    
    if (!status.success) {
      gkyl_moment_app_cout(app, stdout, "** Update method failed! Aborting simulation ....\n");
      break;
    }

    t_curr += status.dt_actual;
    dt = status.dt_suggested;

    write_data(&io_trig, app, t_curr, false);

    if (dt_init < 0.0) {
      dt_init = status.dt_actual;
    }
    else if (status.dt_actual < dt_failure_tol * dt_init) {
      num_failures += 1;

      gkyl_moment_app_cout(app, stdout, "WARNING: Time-step dt = %g", status.dt_actual);
      gkyl_moment_app_cout(app, stdout, " is below %g*dt_init ...", dt_failure_tol);
      gkyl_moment_app_cout(app, stdout, " num_failures = %d\n", num_failures);
      if (num_failures >= num_failures_max) {
        gkyl_moment_app_cout(app, stdout, "ERROR: Time-step was below %g*dt_init ", dt_failure_tol);
        gkyl_moment_app_cout(app, stdout, "%d consecutive times. Aborting simulation ....\n", num_failures_max);
        break;
      }
    }
    else {
      num_failures = 0;
    }

    step += 1;
  }

  write_data(&io_trig, app, t_curr, false);
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
  gkyl_wv_eqn_release(gr_maxwell_tetrad);
  gkyl_gr_spacetime_release(ctx.spacetime);
  gkyl_rect_decomp_release(decomp);
  gkyl_comm_release(comm);
  gkyl_moment_app_release(app);  
  
mpifinalize:
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi) {
    MPI_Finalize();
  }
#endif
  
  return 0;
}