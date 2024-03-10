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

struct sheath_ctx
{
  // Mathematical constants (dimensionless).
  double pi;

  // Physical constants (using non-normalized physical units).
  double epsilon0; // Permittivity of free space.
  double mass_elc; // Electron mass.
  double charge_elc; // Electron charge.
  double mass_ion; // Proton mass.
  double charge_ion; // Proton charge.

  double Tpare;  // Parallel electron temperature.
  double Tpari;  // Parallel ion temperature.
  double Tperpe; // Perp electron temperature.
  double Tperpi; // Perp ion temperature.

  double Te; // Electron temperature.
  double Ti; // Ion temperature.
  double n0; // Reference number density (1 / m^3).

  double B_axis; // Magnetic field axis (simple toroidal coordinates).
  double R0; // Major radius (simple toroidal coordinates).
  double a0; // Minor axis (simple toroidal coordinates).

  double nu_frac; // Collision frequency fraction.

  // Derived physical quantities (using non-normalized physical units).
  double R; // Radial coordinate (simple toroidal coordinates).
  double B0; // Reference magnetic field strength (Tesla).

  double log_lambda_elc; // Logarithm of electron wavelength.
  double nu_elc; // Electron collision frequency.
  double log_lambda_ion; // Logarithm of ion wavelength.
  double nu_ion; // Ion collision frequency.

  double vte; // Electron thermal velocity.
  double vti; // Ion thermal velocity.
  double omega_ci; // Ion cyclotron frequency.

  // Simulation parameters.
  int Nz; // Cell count (configuration space: z-direction).
  int Nvpar; // Cell count (velocity space: parallel velocity direction).
  int Nmu; // Cell count (velocity space: magnetic moment direction).
  double Lz; // Domain size (configuration space: z-direction).
  double Lvpar_elc; // Domain size (electron velocity space: parallel velocity direction).
  double Lmu_elc; // Domain size (electron velocity space: magnetic moment direction).
  double Lvpar_ion; // Domain size (ion velocity space: parallel velocity direction).
  double Lmu_ion; // Domain size (ion velocity space: magnetic moment direction).
  double t_end; // Final simulation time.
  int num_frames; // Number of output frames.
};

struct sheath_ctx
create_ctx(void)
{
  // Mathematical constants (dimensionless).
  double pi = M_PI;

  // Physical constants (using non-normalized physical units).
  double epsilon0 = GKYL_EPSILON0; // Permittivity of free space.
  double eV = GKYL_ELEMENTARY_CHARGE; // Elementary charge.
  double mass_elc = GKYL_ELECTRON_MASS; // Electron mass.
  double mass_ion = 2.014 * GKYL_PROTON_MASS; // Proton mass.
  double charge_elc = -eV; // Electron charge.
  double charge_ion =  eV; // Proton charge.

  double Tpare = 90.0 * eV; // Parallel electron temperature.
  double Tpari = 90.0 * eV; // Parallel ion temperature.

  double Tperpe = 15.0 * eV; // Perpendicular electron temperature.
  double Tperpi = 15.0 * eV; // Perpendicular ion temperature.

  double n0 = 7.0e18; //  Reference number density (1 / m^3).

  double B_axis = 0.5; // Magnetic field axis (simple toroidal coordinates).
  double R0 = 0.85; // Major radius (simple toroidal coordinates).
  double a0 = 0.15; // Minor axis (simple toroidal coordinates).

  double nu_frac = 0.1; // Collision frequency fraction.

  // Derived physical quantities (using non-normalized physical units).

  double Te = (Tpare+2.*Tperpe)/3.; // Electron temperature.
  double Ti = (Tpari+2.*Tperpi)/3.; // Ion temperature.
                         //
  double R = R0 + a0; // Radial coordinate (simple toroidal coordinates).
  double B0 = B_axis * (R0 / R); // Reference magnetic field strength (Tesla).

  // Coulomb logarithms.
  double log_lambda_elc = 6.6 - 0.5*log(n0/1.0e20) + 1.5*log(Te/charge_ion);
  double log_lambda_ion = 6.6 - 0.5*log(n0/1.0e20) + 1.5*log(Ti/charge_ion);

  // Collision frequencies.
  double nu_elc = nu_frac*log_lambda_elc*pow(charge_elc,4)*n0 /
    (6.0*sqrt(2.)*pow(pi,3./2.)*pow(epsilon0,2)*sqrt(mass_elc)*pow(Te,3./2.));
  double nu_ion = nu_frac*log_lambda_ion*pow(charge_ion,4)*n0 /
    (12.0*pow(pi,3./2.)*pow(epsilon0,2)*sqrt(mass_ion)*pow(Ti,3./2.));
  
  double c_s = sqrt(Te / mass_ion); // Sound speed.
  double vte = sqrt(Te / mass_elc); // Electron thermal velocity.
  double vti = sqrt(Ti / mass_ion); // Ion thermal velocity.
  double omega_ci = fabs(charge_ion * B0 / mass_ion); // Ion cyclotron frequency.

  // Simulation parameters.
  int Nz = 4; // Cell count (configuration space: z-direction).
  int Nvpar = 16; // Cell count (velocity space: parallel velocity direction).
  int Nmu = 8; // Cell count (velocity space: magnetic moment direction).
  double Lz = 4.0; // Domain size (configuration space: z-direction).
  double Lvpar_elc = 8.0 * vte; // Domain size (electron velocity space: parallel velocity direction).
  double Lvpar_ion = 8.0 * vti; // Domain size (ion velocity space: parallel velocity direction).
  double Lmu_elc = mass_elc * pow(4.*vte,2)/(2.*B0); // Domain size (electron velocity space: magnetic moment direction).
  double Lmu_ion = mass_ion * pow(4.*vti,2)/(2.*B0); // Domain size (ion velocity space: magnetic moment direction).

  double t_end = 1.0/nu_elc; // Final simulation time.
  int num_frames = 20; // Number of output frames.
  
  struct sheath_ctx ctx = {
    .pi = pi,
    .epsilon0 = epsilon0,
    .mass_elc = mass_elc,
    .charge_elc = charge_elc,
    .mass_ion = mass_ion,
    .charge_ion = charge_ion,
    .Tpare = Tpare,
    .Tperpe = Tperpe,
    .Tpari = Tpari,
    .Tperpi = Tperpi,
    .Te = Te,
    .Ti = Ti,
    .n0 = n0,
    .B_axis = B_axis,
    .R0 = R0,
    .a0 = a0,
    .nu_frac = nu_frac,
    .R = R,
    .B0 = B0,
    .log_lambda_elc = log_lambda_elc,
    .nu_elc = nu_elc,
    .log_lambda_ion = log_lambda_ion,
    .nu_ion = nu_ion,
    .vte = vte,
    .vti = vti,
    .omega_ci = omega_ci,
    .Nz = Nz,
    .Nvpar = Nvpar,
    .Nmu = Nmu,
    .Lz = Lz,
    .Lvpar_elc = Lvpar_elc,
    .Lmu_elc = Lmu_elc,
    .Lvpar_ion = Lvpar_ion,
    .Lmu_ion = Lmu_ion,
    .t_end = t_end,
    .num_frames = num_frames,
  };

  return ctx;
}

void
eval_init_density(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct sheath_ctx *app = ctx;
  double n0 = app->n0;

  // Set number density.
  fout[0] = n0;
}

void
eval_init_upar(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  // Set parallel velocity.
  fout[0] = 0.0;
}

void
eval_init_tpar_elc(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct sheath_ctx *app = ctx;
  double Tpar = app->Tpare;
  fout[0] = Tpar;
}

void
eval_init_tperp_elc(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct sheath_ctx *app = ctx;
  double Tperp = app->Tperpe;
  fout[0] = Tperp;
}

void
eval_init_temp_elc(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  double Tpar[1], Tperp[1];
  eval_init_tpar_elc(t, xn, Tpar, ctx);
  eval_init_tperp_elc(t, xn, Tperp, ctx);

  fout[0] = (Tpar[0] + 2.*Tperp[0])/3.;
}

void
eval_init_tpar_ion(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct sheath_ctx *app = ctx;
  double Tpar = app->Tpari;
  fout[0] = Tpar;
}

void
eval_init_tperp_ion(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct sheath_ctx *app = ctx;
  double Tperp = app->Tperpi;
  fout[0] = Tperp;
}

void
eval_init_temp_ion(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  double Tpar[1], Tperp[1];
  eval_init_tpar_ion(t, xn, Tpar, ctx);
  eval_init_tperp_ion(t, xn, Tperp, ctx);

  fout[0] = (Tpar[0] + 2.*Tperp[0])/3.;
}


void
evalNuElcInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct sheath_ctx *app = ctx;

  double nu_elc = app->nu_elc;

  // Set electron collision frequency.
  fout[0] = nu_elc;
}

void
evalNuIonInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct sheath_ctx *app = ctx;

  double nu_ion = app->nu_ion;

  // Set ion collision frequency.
  fout[0] = nu_ion;
}

static inline void
mapc2p(double t, const double* GKYL_RESTRICT zc, double* GKYL_RESTRICT xp, void* ctx)
{
  // Set physical coordinates (X, Y, Z) from computational coordinates (x, y, z).
  xp[0] = zc[0]; xp[1] = zc[1]; xp[2] = zc[2];
}

void mapc2p_vel_elc(double t, const double *vc, double* GKYL_RESTRICT vp, void *ctx)
{
  struct sheath_ctx *app = ctx;
  double Lvpar = app->Lvpar_elc;
  double Lmu = app->Lmu_elc;

  double cvpar = vc[0], cmu = vc[1];
//  vp[0] = cvpar;
  if (cvpar < 0.)
    vp[0] = -(Lvpar/2.)*pow(cvpar,2);
  else
    vp[0] =  (Lvpar/2.)*pow(cvpar,2);

//  vp[1] = cmu;
  vp[1] = Lmu*pow(cmu,2);
}

void
bmag_func(double t, const double *xc, double* GKYL_RESTRICT fout, void *ctx)
{
  struct sheath_ctx *app = ctx;

  double B0 = app->B0;

  // Set magnetic field strength.
  fout[0] = B0;
}

void
write_data(struct gkyl_tm_trigger* iot, gkyl_gyrokinetic_app* app, double t_curr)
{
  if (gkyl_tm_trigger_check_and_bump(iot, t_curr)) {
    gkyl_gyrokinetic_app_write(app, t_curr, iot->curr - 1);
    gkyl_gyrokinetic_app_calc_mom(app);
    gkyl_gyrokinetic_app_write_mom(app, t_curr, iot->curr - 1);
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

  struct sheath_ctx ctx = create_ctx(); // Context for initialization functions.

  int NZ = APP_ARGS_CHOOSE(app_args.xcells[0], ctx.Nz);
  int NV = APP_ARGS_CHOOSE(app_args.vcells[0], ctx.Nvpar);
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
//    .lower = { -ctx.Lvpar_elc/2, 0.0 },
//    .upper = {  ctx.Lvpar_elc/2, ctx.Lmu_elc },
    .lower = { -1.0, 0.0 },
    .upper = {  1.0, 1.0 },
    .cells = { NV, NMU },
    .polarization_density = ctx.n0,

    .mapc2p = {
      .is_mapped = true,
      .mapping = mapc2p_vel_elc,
      .ctx = &ctx,
    },

    .projection = {
      .proj_id = GKYL_PROJ_BIMAXWELLIAN,
      .density = eval_init_density,
      .upar = eval_init_upar,
      .temppar = eval_init_tpar_elc,
      .tempperp = eval_init_tperp_elc,
      .ctx_density = &ctx,
      .ctx_upar = &ctx,
      .ctx_temppar = &ctx,
      .ctx_tempperp = &ctx,
    },
    .collisions =  {
      .collision_id = GKYL_BGK_COLLISIONS,
      .self_nu = evalNuElcInit,
      .ctx = &ctx,
//      .num_cross_collisions = 1,
//      .collide_with = { "ion" },
    },
    
    .num_diag_moments = 5,
    .diag_moments = { "M0", "M1", "M2", "M2par", "M2perp" },
  };

  // Ion species.
  struct gkyl_gyrokinetic_species ion = {
    .name = "ion",
    .charge = ctx.charge_ion, .mass = ctx.mass_ion,
    .lower = { -ctx.Lvpar_ion/2, 0.0 },
    .upper = {  ctx.Lvpar_ion/2, ctx.Lmu_ion },
    .cells = { NV, NMU },
    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_BIMAXWELLIAN,
      .density = eval_init_density,
      .upar = eval_init_upar,
      .temppar = eval_init_tpar_ion,
      .tempperp = eval_init_tperp_ion,
      .ctx_density = &ctx,
      .ctx_upar = &ctx,
      .ctx_temppar = &ctx,
      .ctx_tempperp = &ctx,
    },

    .collisions =  {
      .collision_id = GKYL_BGK_COLLISIONS,
      .self_nu = evalNuIonInit,
      .ctx = &ctx,
//      .num_cross_collisions = 1,
//      .collide_with = { "elc" },
    },

    .num_diag_moments = 5,
    .diag_moments = { "M0", "M1", "M2", "M2par", "M2perp" },
  };

  // Field.
  struct gkyl_gyrokinetic_field field = {
    .gkfield_id = GKYL_GK_FIELD_BOLTZMANN,
    .electron_mass = ctx.mass_elc,
    .electron_charge = ctx.charge_elc,
    .electron_temp = ctx.Te,
    .bmag_fac = ctx.B0, 
    .fem_parbc = GKYL_FEM_PARPROJ_NONE, 
  };

  // GK app.
  struct gkyl_gk app_inp = {
    .name = "gk_bgk_relax_bimaxwellian_nonuniformv_1x2v_p1",

    .cdim = 1, .vdim = 2,
    .lower = { -ctx.Lz/2},
    .upper = {  ctx.Lz/2},
    .cells = { NZ },
    .poly_order = 1,
    .basis_type = app_args.basis_type,

    .geometry = {
      .geometry_id = GKYL_MAPC2P,
      .world = { 0.0, 0.0 },
      .mapc2p = mapc2p,
      .c2p_ctx = &ctx,
      .bmag_func = bmag_func,
      .bmag_ctx = &ctx
    },

    .num_periodic_dir = 1,
    .periodic_dirs = { 0 },
    .skip_field = true,

    .num_species = 2,
    .species = { elc, ion },
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
