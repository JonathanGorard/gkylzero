#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <gkyl_alloc.h>
#include <gkyl_vlasov.h>
#include <rt_arg_parse.h>

struct pkpm_ot_ctx {
  double epsilon0;
  double mu0;
  double chargeElc; // electron charge
  double massElc; // electron mass
  double chargeIon; // ion charge
  double massIon; // ion mass
  double Te_Ti; // electron to ion temperature ratio
  double n0;
  double vAe;
  double B0;
  double beta;
  double vtElc;
  double vtIon;
  double nuElc;
  double nuIon;
  double delta_u0;
  double delta_B0;
  double L;
  double tend;
  double min_dt;
  bool use_gpu;
};

static inline double
maxwellian(double n, double v, double vth)
{
  double v2 = v*v;
  return n/sqrt(2*M_PI*vth*vth)*exp(-v2/(2*vth*vth));
}

void
evalDistFuncElc(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct pkpm_ot_ctx *app = ctx;
  
  double x = xn[0], y = xn[1], vx = xn[2];

  double qe = app->chargeElc;
  double qi = app->chargeIon;
  double Lx = app->L;
  double Ly = app->L;
  double u0x = app->delta_u0;
  double u0y = app->delta_u0;
  double B0x = app->delta_B0;
  double B0y = app->delta_B0;
  
  double fv = maxwellian(app->n0, vx, app->vtElc);
    
  fout[0] = fv;
  fout[1] = app->vtElc*app->vtElc*fv;
}
void
evalDistFuncIon(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct pkpm_ot_ctx *app = ctx;
  
  double x = xn[0], y = xn[1], vx = xn[2];

  double qe = app->chargeElc;
  double qi = app->chargeIon;
  double Lx = app->L;
  double Ly = app->L;
  double u0x = app->delta_u0;
  double u0y = app->delta_u0;
  double B0x = app->delta_B0;
  double B0y = app->delta_B0;

  double fv = maxwellian(app->n0, vx, app->vtIon);
    
  fout[0] = fv;
  fout[1] = app->vtIon*app->vtIon*fv;
}

void
evalFluidElc(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct pkpm_ot_ctx *app = ctx;
  
  double x = xn[0], y = xn[1];

  double qe = app->chargeElc;
  double qi = app->chargeIon;
  double me = app->massElc;
  double mi = app->massIon;
  double Lx = app->L;
  double Ly = app->L;
  double u0x = app->delta_u0;
  double u0y = app->delta_u0;
  double B0x = app->delta_B0;
  double B0y = app->delta_B0;

  double Jz = (B0y*(4.0*M_PI/Lx)*cos(4.0*M_PI*x/Lx) + B0x*(2.0*M_PI/Ly)*cos(2.0*M_PI*y/Ly)) / app->mu0;

  double vdrift_x = -u0x*sin(2.0*M_PI*y/Ly);
  double vdrift_y = u0y*sin(2.0*M_PI*x/Lx);
  double vdrift_z = -Jz / qi;

  fout[0] = me*vdrift_x;
  fout[1] = me*vdrift_y;
  fout[2] = me*vdrift_z;
}

void
evalFluidIon(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct pkpm_ot_ctx *app = ctx;
  
  double x = xn[0], y = xn[1];

  double qe = app->chargeElc;
  double qi = app->chargeIon;
  double me = app->massElc;
  double mi = app->massIon;
  double Lx = app->L;
  double Ly = app->L;
  double u0x = app->delta_u0;
  double u0y = app->delta_u0;
  double B0x = app->delta_B0;
  double B0y = app->delta_B0;

  double vdrift_x = -u0x*sin(2.0*M_PI*y/Ly);
  double vdrift_y = u0y*sin(2.0*M_PI*x/Lx);
  double vdrift_z = 0.0;

  fout[0] = mi*vdrift_x;
  fout[1] = mi*vdrift_y;
  fout[2] = 0.0;
}

void
evalFieldFunc(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct pkpm_ot_ctx *app = ctx;

  double x = xn[0], y = xn[1];

  double qe = app->chargeElc;
  double qi = app->chargeIon;
  double Lx = app->L;
  double Ly = app->L;
  double u0x = app->delta_u0;
  double u0y = app->delta_u0;
  double B0x = app->delta_B0;
  double B0y = app->delta_B0;

  double Jz = (B0y*(4.0*M_PI/Lx)*cos(4.0*M_PI*x/Lx) + B0x*(2.0*M_PI/Ly)*cos(2.0*M_PI*y/Ly)) / app->mu0;

  double B_x = -B0x*sin(2.0*M_PI*y/Ly);
  double B_y = B0y*sin(4.0*M_PI*x/Lx);
  double B_z = app->B0;

  // Assumes qi = abs(qe)
  double u_xe = -u0x*sin(2.0*M_PI*y/Ly);
  double u_ye = u0y*sin(2.0*M_PI*x/Lx);
  double u_ze = -Jz / qi;

  // E = - v_e x B ~  (J - u) x B
  double E_x = - (u_ye*B_z - u_ze*B_y);
  double E_y = - (u_ze*B_x - u_xe*B_z);
  double E_z = - (u_xe*B_y - u_ye*B_x);
  
  fout[0] = E_x; fout[1] = E_y, fout[2] = E_z;
  fout[3] = B_x; fout[4] = B_y; fout[5] = B_z;
  fout[6] = 0.0; fout[7] = 0.0;
}

void
evalNuElc(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct pkpm_ot_ctx *app = ctx;
  fout[0] = app->nuElc;
}

void
evalNuIon(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct pkpm_ot_ctx *app = ctx;
  fout[0] = app->nuIon;
}

struct pkpm_ot_ctx
create_ctx(void)
{
  double epsilon0 = 1.0; // permittivity of free space
  double mu0 = 1.0; // pemiability of free space

  double massElc = 1.0; // electron mass
  double chargeElc = -1.0; // electron charge
  double massIon = 25.0; // ion mass
  double chargeIon = 1.0; // ion charge

  double Te_Ti = 1.0; // ratio of electron to ion temperature
  double n0 = 1.0; // initial number density

  double vAe = 0.5;
  double B0 = vAe*sqrt(mu0*n0*massElc);
  double beta = 0.08;
  double vtElc = vAe*sqrt(beta/2.0);

  // ion velocities
  double vAi = vAe/sqrt(massIon);
  double vtIon = vtElc/sqrt(massIon); //Ti/Te = 1.0

  // ion cyclotron frequency and gyroradius
  double omegaCi = chargeIon*B0/massIon;
  double di = vAi/omegaCi;

  // collision frequencies
  double nuElc = 0.01*omegaCi;
  double nuIon = 0.01*omegaCi/sqrt(massIon);

  // OT initial conditions
  double delta_u0 = 0.2*vAi;
  double delta_B0 = 0.2*B0;

  // domain size and simulation time
  double L = 20.48*di;
  double tend = 75.0/omegaCi;
  
  struct pkpm_ot_ctx ctx = {
    .epsilon0 = epsilon0,
    .mu0 = mu0,
    .chargeElc = chargeElc,
    .massElc = massElc,
    .chargeIon = chargeIon,
    .massIon = massIon,
    .Te_Ti = Te_Ti,
    .n0 = n0,
    .vAe = vAe,
    .B0 = B0,
    .beta = beta,
    .vtElc = vtElc,
    .vtIon = vtIon,
    .nuElc = nuElc,
    .nuIon = nuIon,
    .delta_u0 = delta_u0,
    .delta_B0 = delta_B0,
    .L = L,
    .tend = tend,
    .min_dt = 1.0e-2, 
  };
  return ctx;
}

void
write_data(struct gkyl_tm_trigger *iot, gkyl_vlasov_app *app, double tcurr)
{
  if (gkyl_tm_trigger_check_and_bump(iot, tcurr)) {
    gkyl_vlasov_app_write(app, tcurr, iot->curr-1);
    gkyl_vlasov_app_calc_mom(app); gkyl_vlasov_app_write_mom(app, tcurr, iot->curr-1);
  }
}

int
main(int argc, char **argv)
{
  struct gkyl_app_args app_args = parse_app_args(argc, argv);

  int NX = APP_ARGS_CHOOSE(app_args.xcells[0], 112);
  int VX = APP_ARGS_CHOOSE(app_args.vcells[0], 16);

  if (app_args.trace_mem) {
    gkyl_cu_dev_mem_debug_set(true);
    gkyl_mem_debug_set(true);
  }
     
  struct pkpm_ot_ctx ctx = create_ctx(); // context for init functions

  // electron momentum                                                                                              
  struct gkyl_vlasov_fluid_species fluid_elc = {
    .name = "fluid_elc",
    .num_eqn = 3,
    .pkpm_species = "elc",
    .ctx = &ctx,
    .init = evalFluidElc,
    .nuHyp = 1.0e-4,
  };  
  
  // electrons
  struct gkyl_vlasov_species elc = {
    .name = "elc",
    .model_id = GKYL_MODEL_PKPM,
    .pkpm_fluid_species = "fluid_elc",
    .charge = ctx.chargeElc, .mass = ctx.massElc,
    .lower = { -6.0 * ctx.vtElc},
    .upper = { 6.0 * ctx.vtElc}, 
    .cells = { VX },

    .ctx = &ctx,
    .init = evalDistFuncElc,

    .collisions = {
      .collision_id = GKYL_LBO_COLLISIONS,

      .ctx = &ctx,
      .self_nu = evalNuElc,
    },    

    .num_diag_moments = 0,
  };

  // ion momentum                                                                                              
  struct gkyl_vlasov_fluid_species fluid_ion = {
    .name = "fluid_ion",
    .num_eqn = 3,
    .pkpm_species = "ion",
    .ctx = &ctx,
    .init = evalFluidIon,
    .nuHyp = 1.0e-4, 
  };  
  
  // ions
  struct gkyl_vlasov_species ion = {
    .name = "ion",
    .model_id = GKYL_MODEL_PKPM,
    .pkpm_fluid_species = "fluid_ion",
    .charge = ctx.chargeIon, .mass = ctx.massIon,
    .lower = { -6.0 * ctx.vtIon},
    .upper = { 6.0 * ctx.vtIon}, 
    .cells = { VX },

    .ctx = &ctx,
    .init = evalDistFuncIon,

    .collisions = {
      .collision_id = GKYL_LBO_COLLISIONS,

      .ctx = &ctx,
      .self_nu = evalNuIon,
    },    

    .num_diag_moments = 0,
  };

  // field
  struct gkyl_vlasov_field field = {
    .epsilon0 = 1.0, .mu0 = 1.0,
    .elcErrorSpeedFactor = 0.0,
    .mgnErrorSpeedFactor = 0.0,

    .ctx = &ctx,
    .init = evalFieldFunc
  };

  // VM app
  struct gkyl_vm vm = {
    .name = "pkpm_ot_p1",

    .cdim = 2, .vdim = 1,
    .lower = { 0.0, 0.0 },
    .upper = { ctx.L, ctx.L },
    .cells = { NX, NX },
    .poly_order = 1,
    .basis_type = app_args.basis_type,
    //.cfl_frac = 0.8,
    
    .num_periodic_dir = 2,
    .periodic_dirs = { 0, 1 },

    .num_species = 2,
    .species = { elc, ion },
    .num_fluid_species = 2,
    .fluid_species = { fluid_elc, fluid_ion },
    .field = field,

    .use_gpu = app_args.use_gpu,
  };

  // create app object
  gkyl_vlasov_app *app = gkyl_vlasov_app_new(&vm);

  // start, end and initial time-step
  double tcurr = 0.0, tend = ctx.tend;
  double dt = tend-tcurr;
  int nframe = 75;
  // create trigger for IO
  struct gkyl_tm_trigger io_trig = { .dt = tend/nframe };

  // initialize simulation
  gkyl_vlasov_app_apply_ic(app, tcurr);
  write_data(&io_trig, app, tcurr);

  long step = 1, num_steps = app_args.num_steps;
  while ((tcurr < tend) && (step <= num_steps)) {
    printf("Taking time-step at t = %g ...", tcurr);
    struct gkyl_update_status status = gkyl_vlasov_update(app, dt);
    printf(" dt = %g\n", status.dt_actual);
    
    if (!status.success) {
      printf("** Update method failed! Aborting simulation ....\n");
      break;
    }
    if (status.dt_actual < ctx.min_dt) {
      printf("** Time step crashing! Aborting simulation and writing out last output ....\n");
      gkyl_vlasov_app_write(app, tcurr, 1000);
      gkyl_vlasov_app_calc_mom(app); gkyl_vlasov_app_write_mom(app, tcurr, 1000);
      break;
    }
    tcurr += status.dt_actual;
    dt = status.dt_suggested;

    write_data(&io_trig, app, tcurr);

    step += 1;
  }

  gkyl_vlasov_app_stat_write(app);

  // fetch simulation statistics
  struct gkyl_vlasov_stat stat = gkyl_vlasov_app_stat(app);

  // simulation complete, free app
  gkyl_vlasov_app_release(app);

  printf("\n");
  printf("Number of update calls %ld\n", stat.nup);
  printf("Number of forward-Euler calls %ld\n", stat.nfeuler);
  printf("Number of RK stage-2 failures %ld\n", stat.nstage_2_fail);
  if (stat.nstage_2_fail > 0) {
    printf("Max rel dt diff for RK stage-2 failures %g\n", stat.stage_2_dt_diff[1]);
    printf("Min rel dt diff for RK stage-2 failures %g\n", stat.stage_2_dt_diff[0]);
  }  
  printf("Number of RK stage-3 failures %ld\n", stat.nstage_3_fail);
  printf("Species RHS calc took %g secs\n", stat.species_rhs_tm);
  printf("Species collisions took %g secs\n", stat.species_coll_mom_tm);
  printf("Species collisions took %g secs\n", stat.species_coll_tm);
  printf("Field RHS calc took %g secs\n", stat.field_rhs_tm);
  printf("Current evaluation and accumulate took %g secs\n", stat.current_tm);
  printf("Updates took %g secs\n", stat.total_tm);
  
  return 0;
}