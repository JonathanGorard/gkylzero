#include <math.h>
#include <stdio.h>
#include <time.h>

#include <gkyl_alloc.h>
#include <gkyl_const.h>
#include <gkyl_eqn_type.h>
#include <gkyl_fem_parproj.h>
#include <gkyl_fem_poisson_bctype.h>
#include <gkyl_gyrokinetic.h>
#include <gkyl_math.h>

#include <rt_arg_parse.h>

// Define the context of the simulation. This is basically all the globals
struct gk_mirror_ctx
{
  int cdim, vdim; // Dimensionality.
  // Plasma parameters
  double mi;
  double qi;
  double me;
  double qe;
  double Te0;
  double n0;
  double B_p;
  double beta;
  double tau;
  double Ti0;
  // Parameters controlling initial conditions.
  double alim;
  double alphaIC0;
  double alphaIC1;
  double nuFrac;
  // Ion-ion collision freq.
  double logLambdaIon;
  double nuIon;
  // Thermal speeds.
  double vti;
  double vte;
  double c_s;
  // Gyrofrequencies and gyroradii.
  double omega_ci;
  double rho_s;
  double RatZeq0; // Radius of the field line at Z=0.
  // Axial coordinate Z extents. Endure that Z=0 is not on
  double Z_min;
  double Z_max;
  double z_min;
  double z_max;
  double psi_eval;
  double psi_in;
  double z_in;
  // Magnetic equilibrium model.
  double mcB;
  double gamma;
  double Z_m;
  // Bananna tip info. Hardcoad to avoid dependency on ctx
  double B_bt;
  double R_bt;
  double Z_bt;
  double z_bt;
  double R_m;
  double B_m;
  double z_m;
  // Physics parameters at mirror throat
  double n_m;
  double Ti_m;
  double Ti_perp0;
  double Ti_par0;
  double Ti_perp_m;
  double Ti_par_m;
  double cs_m;
  // Source parameters
  double NSrcIon;
  double lineLengthSrcIon;
  double sigSrcIon;
  double NSrcFloorIon;
  double TSrc0Ion;
  double TSrcFloorIon;
  // Grid parameters
  double vpar_max_ion;
  double mu_max_ion;
  int Nz;
  int Nvpar;
  int Nmu;
  int cells[GKYL_MAX_DIM]; // Number of cells in all directions.
  int poly_order;
  double t_end;
  int num_frames;
  int int_diag_calc_num; // Number of integrated diagnostics computations (=INT_MAX for every step).
  double dt_failure_tol; // Minimum allowable fraction of initial time-step.
  int num_failures_max; // Maximum allowable number of consecutive small time-steps.
  // For non-uniform mapping
  double diff_dz;
  double psi_in_diff;
  int mapping_order;
  double mapping_frac;
};

double
psi_RZ(double RIn, double ZIn, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  double mcB = app->mcB;
  double gamma = app->gamma;
  double Z_m = app->Z_m;
  double psi = 0.5 * pow(RIn, 2.) * mcB *
               (1. / (M_PI * gamma * (1. + pow((ZIn - Z_m) / gamma, 2.))) +
                1. / (M_PI * gamma * (1. + pow((ZIn + Z_m) / gamma, 2.))));
  return psi;
}

double
R_psiZ(double psiIn, double ZIn, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  double Rout = sqrt(2.0 * psiIn / (app->mcB * 
    (1.0 / (M_PI * app->gamma * (1.0 + pow((ZIn - app->Z_m) / app->gamma, 2.))) +
     1.0 / (M_PI * app->gamma * (1.0 + pow((ZIn + app->Z_m) / app->gamma, 2.))))));
  return Rout;
}

void
Bfield_psiZ(double psiIn, double ZIn, void *ctx, double *BRad, double *BZ, double *Bmag)
{
  struct gk_mirror_ctx *app = ctx;
  double Rcoord = R_psiZ(psiIn, ZIn, ctx);
  double mcB = app->mcB;
  double gamma = app->gamma;
  double Z_m = app->Z_m;
  *BRad = -(1.0 / 2.0) * Rcoord * mcB *
          (-2.0 * (ZIn - Z_m) / (M_PI * pow(gamma, 3.) * (pow(1.0 + pow((ZIn - Z_m) / gamma, 2.), 2.))) -
            2.0 * (ZIn + Z_m) / (M_PI * pow(gamma, 3.) * (pow(1.0 + pow((ZIn + Z_m) / gamma, 2.), 2.))));
  *BZ = mcB *
        (1.0 / (M_PI * gamma * (1.0 + pow((ZIn - Z_m) / gamma, 2.))) +
         1.0 / (M_PI * gamma * (1.0 + pow((ZIn + Z_m) / gamma, 2.))));
  *Bmag = sqrt(pow(*BRad, 2) + pow(*BZ, 2));
}

double
integrand_z_psiZ(double ZIn, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  double psi = app->psi_in;
  double BRad, BZ, Bmag;
  Bfield_psiZ(psi, ZIn, ctx, &BRad, &BZ, &Bmag);
  return Bmag / BZ;
}

double
z_psiZ(double psiIn, double ZIn, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  app->psi_in = psiIn;
  double eps = 0.0;
  struct gkyl_qr_res integral;
  if (eps <= ZIn)
  {
    integral = gkyl_dbl_exp(integrand_z_psiZ, ctx, eps, ZIn, 7, 1e-14);
  }
  else
  {
    integral = gkyl_dbl_exp(integrand_z_psiZ, ctx, ZIn, eps, 7, 1e-14); 
    integral.res = -integral.res;
  }
  return integral.res;
}

// Invert z(Z) via root-finding.
double
root_Z_psiz(double Z, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  return app->z_in - z_psiZ(app->psi_in, Z, ctx);
}

double
Z_psiz(double psiIn, double zIn, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  double maxL = app->Z_max - app->Z_min;
  double eps = maxL / app->Nz;   // Interestingly using a smaller eps yields larger errors in some geo quantities.
  app->psi_in = psiIn;
  app->z_in = zIn;
  struct gkyl_qr_res Zout;
  if (zIn >= 0.0)
  {
    double fl = root_Z_psiz(-eps, ctx);
    double fr = root_Z_psiz(app->Z_max + eps, ctx);
    Zout = gkyl_ridders(root_Z_psiz, ctx, -eps, app->Z_max + eps, fl, fr, 1000, 1e-14);
  }
  else
  {
    double fl = root_Z_psiz(app->Z_min - eps, ctx);
    double fr = root_Z_psiz(eps, ctx);
    Zout = gkyl_ridders(root_Z_psiz, ctx, app->Z_min - eps, eps, fl, fr, 1000, 1e-14);
  }
  return Zout.res;
}

// Non-uniform grid mapping
double
dBdz(double z, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  double dz = app->diff_dz;
  double psi = app->psi_in_diff;
  double Zp = Z_psiz(psi, z + dz, ctx);
  double Zm = Z_psiz(psi, z - dz, ctx);
  double B_rad, B_Z, Bmag_p, Bmag_m;
  Bfield_psiZ(psi, Zp, ctx, &B_rad, &B_Z, &Bmag_p);
  Bfield_psiZ(psi, Zm, ctx, &B_rad, &B_Z, &Bmag_m);
  double dBdz = (Bmag_p - Bmag_m) / (2 * dz);
  return fabs(dBdz);
}

// double
// z_xi_dBdz_inverse(double chi, double psi, void *ctx)
// {
//   struct gk_mirror_ctx *app = ctx;
//   double map_strength = app->map_strength; // 1 is full strength, 0 is no mapping
//   if (map_strength == 0.0)
//   {
//     return chi;
//   }
//   else
//   {
//     double psi = app->psi_in_diff;
//     double z_min = app->z_min;
//     double z_max = app->z_max;
//     ;
//     if (chi <= z_min || chi >= z_max)
//     {
//       return chi;
//     }
//     else
//     {
//       struct gkyl_qr_res integral = gkyl_dbl_exp(dBdz, ctx, z_min, chi, 7, 1e-14);
//       double coord = (integral.res / app->map_integral_total * (z_max - z_min) + z_min) * map_strength + (1 - map_strength) * chi;
//       return coord;
//     }
//   }
// }

double
z_xi(double xi, double psi, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  double z_min = app->z_min;
  double z_max = app->z_max;
  double z_m = app->z_m;
  int n = app->mapping_order;
  double frac = app->mapping_frac; // 1 is full mapping, 0 is no mapping
  double z, left, right;
  if (xi >= z_min && xi <= z_max)
  {
    if (xi <= -z_m)
    {
      left = -z_m;
      right = z_min;
    }
    else if (xi <= 0.0)
    {
      left = -z_m;
      right = 0.0;
    }
    else if (xi <= z_m)
    {
      left = z_m;
      right = 0.0;
    }
    else
    {
      left = z_m;
      right = z_max;
    }
    z = (pow(right - left, 1 - n) * pow(xi - left, n) + left) * frac + xi * (1 - frac);
  }
  else
  {
    z = xi;
  }
  return z;
}

void
eval_density_ion_source(double t, const double *GKYL_RESTRICT xn, double *GKYL_RESTRICT fout, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  double psi = psi_RZ(app->RatZeq0, 0.0, ctx); // Magnetic flux function psi of field line.
  double z = z_xi(xn[0], psi, ctx);
  double Z = Z_psiz(psi, z, ctx); // Cylindrical axial coordinate.
  double NSrc = app->NSrcIon;
  double zSrc = app->lineLengthSrcIon;
  double sigSrc = app->sigSrcIon;
  double NSrcFloor = app->NSrcFloorIon;
  if (fabs(Z) <= app->Z_m)
  {
    fout[0] = fmax(NSrcFloor, (NSrc / sqrt(2.0 * M_PI * pow(sigSrc, 2))) *
                                  exp(-1 * pow((z - zSrc), 2) / (2.0 * pow(sigSrc, 2))));
  }
  else
  {
    fout[0] = 1e-16;
  }
}

void
eval_upar_ion_source(double t, const double *GKYL_RESTRICT xn, double *GKYL_RESTRICT fout, void *ctx)
{
  fout[0] = 0.0;
}

void
eval_temp_ion_source(double t, const double *GKYL_RESTRICT xn, double *GKYL_RESTRICT fout, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  double psi = psi_RZ(app->RatZeq0, 0.0, ctx); // Magnetic flux function psi of field line.
  double z = z_xi(xn[0], psi, ctx);
  double sigSrc = app->sigSrcIon;
  double TSrc0 = app->TSrc0Ion;
  double Tfloor = app->TSrcFloorIon;
  if (fabs(z) <= 2.0 * sigSrc)
  {
    fout[0] = TSrc0;
  }
  else
  {
    fout[0] = Tfloor;
  }
}

// Ion initial conditions
void
eval_density_ion(double t, const double *GKYL_RESTRICT xn, double *GKYL_RESTRICT fout, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  double psi = psi_RZ(app->RatZeq0, 0.0, ctx); // Magnetic flux function psi of field line.
  double z = z_xi(xn[0], psi, ctx);
  double Z = Z_psiz(psi, z, ctx); // Cylindrical axial coordinate.
  double R = R_psiZ(psi, Z, ctx); // Cylindrical radial coordinate.
  double BRad, BZ, Bmag;
  Bfield_psiZ(psi, Z, ctx, &BRad, &BZ, &Bmag);
  if (fabs(Z) <= app->Z_bt)
  {
    fout[0] = app->n0 * pow(1.0 - pow((R - app->R_bt) / app->alim, 2), app->alphaIC0 / 2);
  }
  else if (fabs(Z) <= app->Z_m)
  {
    fout[0] = app->n0 * pow(1.0 - pow((R - app->R_bt) / app->alim, 2), app->alphaIC1 / 2);
  }
  else
  {
    fout[0] = app->n_m * sqrt(Bmag / app->B_m);
  }
}

void
eval_upar_ion(double t, const double *GKYL_RESTRICT xn, double *GKYL_RESTRICT fout, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  double psi = psi_RZ(app->RatZeq0, 0.0, ctx); // Magnetic flux function psi of field line.
  double z = z_xi(xn[0], psi, ctx);
  if (fabs(z) <= app->z_m)
  {
    fout[0] = 0.0;
  }
  else if (z > app->z_m)
  {
    fout[0] = app->cs_m * (z - app->z_m); //* (z -  / app->z_m);
  }
  else
  {
    fout[0] = app->cs_m * (z + app->z_m); //* (z + app->z_m) / app->z_m;
  }
}

void
eval_temp_par_ion(double t, const double *GKYL_RESTRICT xn, double *GKYL_RESTRICT fout, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  double psi = psi_RZ(app->RatZeq0, 0.0, ctx); // Magnetic flux function psi of field line.
  double z = z_xi(xn[0], psi, ctx);
  double z_m = app->z_m;
  double z_max = app->z_max;
  if (fabs(z) <= z_m)
  {
    fout[0] = (app->Ti_par0 - app->Ti_par_m) * tanh(3 * z_m * fabs(z_m - fabs(z))) + app->Ti_par_m;
  }
  else
  {
    fout[0] = app->Ti_par_m;
  }
}

void
eval_temp_perp_ion(double t, const double *GKYL_RESTRICT xn, double *GKYL_RESTRICT fout, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  double psi = psi_RZ(app->RatZeq0, 0.0, ctx); // Magnetic flux function psi of field line.
  double z = z_xi(xn[0], psi, ctx);
  double z_m = app->z_m;
  double z_max = app->z_max;
  if (fabs(z) <= z_m)
  {
    fout[0] = (app->Ti_perp_m - app->Ti_perp0) * ((tanh((fabs(z) - z_m * 0.8) * 10 * z_m)) / 2 + 0.5) + app->Ti_perp0;
  }
  else
  {
    fout[0] = app->Ti_perp_m * exp(-5 * (fabs(z_m - fabs(z))));
  }
}

void
eval_temp_ion(double t, const double *GKYL_RESTRICT xn, double *GKYL_RESTRICT fout, void *ctx)
{
  eval_temp_par_ion(t, xn, fout, ctx);
  double Tpar = fout[0];
  eval_temp_perp_ion(t, xn, fout, ctx);
  double Tperp = fout[0];
  fout[0] = (Tpar + 2 * Tperp) / 3;
}

void
evalNuIon(double t, const double *GKYL_RESTRICT xn, double *GKYL_RESTRICT fout, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  fout[0] = app->nuIon;
}

// Geometry evaluation functions for the gk app
void
mapc2p(double t, const double *xc, double *GKYL_RESTRICT xp, void *ctx)
{
  double psi = xc[0];
  double theta = xc[1];
  double z = z_xi(xc[2], psi, ctx);

  double Z = Z_psiz(psi, z, ctx);
  double R = R_psiZ(psi, Z, ctx);

  // Cartesian coordinates on plane perpendicular to Z axis.
  double x = R * cos(theta);
  double y = R * sin(theta);
  xp[0] = x;
  xp[1] = y;
  xp[2] = Z;
}

void
bmag_func(double t, const double *xc, double *GKYL_RESTRICT fout, void *ctx)
{
  struct gk_mirror_ctx *app = ctx;
  double z = z_xi(xc[2], xc[0], ctx);
  double psi = psi_RZ(app->RatZeq0, 0.0, ctx); // Magnetic flux function psi of field line.
  double Z = Z_psiz(psi, z, ctx);
  double BRad, BZ, Bmag;
  Bfield_psiZ(psi, Z, ctx, &BRad, &BZ, &Bmag);
  fout[0] = Bmag;
}

struct gk_mirror_ctx
create_ctx(void)
{
  int cdim = 1, vdim = 2; // Dimensionality.

  // Universal constant parameters.
  double eps0 = GKYL_EPSILON0;
  double mu0 = GKYL_MU0; // Not sure if this is right
  double eV = GKYL_ELEMENTARY_CHARGE;
  double mp = GKYL_PROTON_MASS; // ion mass
  double me = GKYL_ELECTRON_MASS;
  double qi = eV;  // ion charge
  double qe = -eV; // electron charge

  // Plasma parameters.
  double mi = 2.014 * mp;
  double Te0 = 940 * eV;
  double n0 = 3e19;
  double B_p = 0.53;
  double beta = 0.4;
  double tau = pow(B_p, 2.) * beta / (2.0 * mu0 * n0 * Te0) - 1.;
  double Ti0 = tau * Te0;

  // Parameters controlling initial conditions.
  double alim = 0.125;
  double alphaIC0 = 2;
  double alphaIC1 = 10;

  double nuFrac = 1.0;
  // Ion-ion collision freq.
  double logLambdaIon = 6.6 - 0.5 * log(n0 / 1e20) + 1.5 * log(Ti0 / eV);
  double nuIon = nuFrac * logLambdaIon * pow(eV, 4.) * n0 /
                 (12 * pow(M_PI, 3. / 2.) * pow(eps0, 2.) * sqrt(mi) * pow(Ti0, 3. / 2.));

  // Thermal speeds.
  double vti = sqrt(Ti0 / mi);
  double vte = sqrt(Te0 / me);
  double c_s = sqrt(Te0 / mi);

  // Gyrofrequencies and gyroradii.
  double omega_ci = eV * B_p / mi;
  double rho_s = c_s / omega_ci;

  // Perpendicular wavenumber in SI units:
  // Geometry parameters.
  double RatZeq0 = 0.10; // Radius of the field line at Z=0.
  // Axial coordinate Z extents. Endure that Z=0 is not on
  // the boundary of a cell (due to AD errors).
  double Z_min = -2.5;
  double Z_max = 2.5;
  double z_min = -2.515312;
  double z_max = 2.515312;
  double psi_eval = 0.0026530898059565;

  // Parameters controlling the magnetic equilibrium model.
  double mcB = 6.51292;
  double gamma = 0.124904;
  double Z_m = 0.98;

  // Source parameters
  double NSrcIon = 3.1715e23 / 8.0;
  double lineLengthSrcIon = 0.0;
  double sigSrcIon = Z_m / 4.0;
  double NSrcFloorIon = 0.05 * NSrcIon;
  double TSrc0Ion = Ti0 * 1.25;
  double TSrcFloorIon = TSrc0Ion / 8.0;

  // Grid parameters
  double vpar_max_ion = 5 * vti;
  double mu_max_ion = mi * pow(3. * vti, 2.) / (2. * B_p);
  int Nz = 32;
  int Nvpar = 32; // Number of cells in the paralell velocity direction 96
  int Nmu = 48;  // Number of cells in the mu direction 192
  int poly_order = 1;

  double t_end = 6.0e-8;
  int num_frames = 1;
  int int_diag_calc_num = num_frames*100;
  double dt_failure_tol = 1.0e-4; // Minimum allowable fraction of initial time-step.
  int num_failures_max = 20; // Maximum allowable number of consecutive small time-steps.

  // Bananna tip info. Hardcoad to avoid dependency on ctx
  double B_bt = 1.058278;
  double R_bt = 0.071022;
  double Z_bt = 0.467101;
  double z_bt = 0.468243;
  double R_m = 0.017845;
  double B_m = 16.662396;
  double z_m = 0.982544;

  // Physics parameters at mirror throat
  double n_m = 1.105617e19;
  double Ti_m = 3081.437703 * eV;
  double cs_m = 4.037740e5;

  // Initial conditions parameters
  double Ti_perp0 = 10000 * eV;
  double Ti_par0 = 7500 * eV;
  double Ti_perp_m = 15000 * eV;
  double Ti_par_m = 1000 * eV;

  // Non-uniform z mapping
  int mapping_order = 20;  // Order of the polynomial to fit through points for mapc2p
  double mapping_frac = 0.0;//0.72; // 1 is full mapping, 0 is no mapping

  struct gk_mirror_ctx ctx = {
    .cdim = cdim,
    .vdim = vdim,
    .mi = mi,
    .qi = qi,
    .me = me,
    .qe = qe,
    .Te0 = Te0,
    .n0 = n0,
    .B_p = B_p,
    .beta = beta,
    .tau = tau,
    .Ti0 = Ti0,
    .alim = alim,
    .alphaIC0 = alphaIC0,
    .alphaIC1 = alphaIC1,
    .nuFrac = nuFrac,
    .logLambdaIon = logLambdaIon,
    .nuIon = nuIon,
    .vti = vti,
    .vte = vte,
    .c_s = c_s,
    .omega_ci = omega_ci,
    .rho_s = rho_s,
    .RatZeq0 = RatZeq0,
    .Z_min = Z_min,
    .Z_max = Z_max,
    .z_min = z_min,
    .z_max = z_max,
    .psi_eval = psi_eval,
    .mcB = mcB,
    .gamma = gamma,
    .Z_m = Z_m,
    .B_bt = B_bt,
    .R_bt = R_bt,
    .Z_bt = Z_bt,
    .z_bt = z_bt,
    .R_m = R_m,
    .B_m = B_m,
    .z_m = z_m,
    .n_m = n_m,
    .Ti_m = Ti_m,
    .Ti_perp0 = Ti_perp0,
    .Ti_par0 = Ti_par0,
    .Ti_perp_m = Ti_perp_m,
    .Ti_par_m = Ti_par_m,
    .cs_m = cs_m,
    .NSrcIon = NSrcIon,
    .lineLengthSrcIon = lineLengthSrcIon,
    .sigSrcIon = sigSrcIon,
    .NSrcFloorIon = NSrcFloorIon,
    .TSrc0Ion = TSrc0Ion,
    .TSrcFloorIon = TSrcFloorIon,
    .vpar_max_ion = vpar_max_ion,
    .mu_max_ion = mu_max_ion,
    .Nz = Nz,
    .Nvpar = Nvpar,
    .Nmu = Nmu,
    .cells = {Nz, Nvpar, Nmu},
    .poly_order = poly_order,
    .t_end = t_end,
    .num_frames = num_frames,
    .int_diag_calc_num = int_diag_calc_num,
    .dt_failure_tol = dt_failure_tol,
    .num_failures_max = num_failures_max,
    .mapping_order = mapping_order,  // Order of the polynomial to fit through points for mapc2p
    .mapping_frac = mapping_frac, // 1 is full mapping, 0 is no mapping
  };
  // Printing
  double dxi = (ctx.z_max - ctx.z_min) / ctx.Nz;
  double diff_z_max = z_xi(ctx.z_m + dxi/2, ctx.psi_eval, &ctx) - z_xi(ctx.z_m - dxi/2, ctx.psi_eval, &ctx);
  double diff_z_p75 = z_xi(ctx.z_m * .75 + dxi/2, ctx.psi_eval, &ctx) - z_xi(ctx.z_m * .75 - dxi/2, ctx.psi_eval, &ctx);
  double diff_z_p50 = z_xi(ctx.z_m * .5  + dxi/2, ctx.psi_eval, &ctx) - z_xi(ctx.z_m * .5  - dxi/2, ctx.psi_eval, &ctx);
  double diff_z_p25 = z_xi(ctx.z_m * .25 + dxi/2, ctx.psi_eval, &ctx) - z_xi(ctx.z_m * .25 - dxi/2, ctx.psi_eval, &ctx);
  double diff_z_min = z_xi(dxi/2, ctx.psi_eval, &ctx) - z_xi(-dxi/2, ctx.psi_eval, &ctx);
  if (ctx.mapping_frac == 0.0)
  {
    printf("Uniform cell spacing in z: %g m\n", dxi);
  } else {
    printf("Non-uniform cell spacings:\n");
    printf("Total number of cells in z   : %d\n", ctx.Nz);
    printf("Polynomials order %i with mapping fraction %g\n", ctx.mapping_order, ctx.mapping_frac);
    printf("Uniform computational spacing: %g m\n", dxi);
    printf("Maximum cell spacing at z_m  : %g m\n", diff_z_max);
    printf("Cell spacing at z_m * 0.75   : %g m\n", diff_z_p75);
    printf("Cell spacing at z_m * 0.50   : %g m\n", diff_z_p50);
    printf("Cell spacing at z_m * 0.25   : %g m\n", diff_z_p25);
    printf("Minimum cell spacing at 0    : %g m\n", diff_z_min);
  }

  // Looking at calculating dB/dz in each cell
  // xi is uniformly spaced computational coordinate
  double dB_values[ctx.Nz];
  double mean = 0.0;
  double max_dB = 0.0;
  double loc_max_dB = 0.0;
  for (int iz = 0; iz < ctx.Nz; iz++)
  {
    double left_xi = ctx.z_min + iz * dxi;
    double right_xi = ctx.z_min + (iz + 1) * dxi;
    double psi = ctx.psi_eval;
    double left_z = z_xi(left_xi, psi, &ctx);
    double right_z = z_xi(right_xi, psi, &ctx);
    double B_rad, B_Z, Bmag_left, Bmag_right;
    Bfield_psiZ(psi, left_z, &ctx, &B_rad, &B_Z, &Bmag_left);
    Bfield_psiZ(psi, right_z, &ctx, &B_rad, &B_Z, &Bmag_right);
    double dB = (Bmag_right - Bmag_left);
    dB_values[iz] = fabs(dB);
    mean += fabs(dB);
    if (fabs(dB) > max_dB)
    {
      max_dB = fabs(dB);
      loc_max_dB = (left_z + right_z) / 2;
    }
  }
  // Calculate mean and standard deviation of dBdz values
  mean /= ctx.Nz;
  double std = 0.0;
  for (int iz = 0; iz < ctx.Nz; iz++)
  {
    std += pow(dB_values[iz] - mean, 2);
  }
  std = sqrt(std / ctx.Nz);
  printf("Mean dB: %g\n", mean);
  printf("Std dB : %g\n", std);
  printf("Max dB : %g\n", max_dB);
  printf("Max dB location: %g\n", loc_max_dB);
  return ctx;
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
  bool trig_now = gkyl_tm_trigger_check_and_bump(iot, t_curr);
  if (trig_now || force_write) {
    int frame = (!trig_now) && force_write? iot->curr : iot->curr-1;

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

int main(int argc, char **argv)
{
  struct gkyl_app_args app_args = parse_app_args(argc, argv);

#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
    MPI_Init(&argc, &argv);
#endif

  if (app_args.trace_mem)
  {
    gkyl_cu_dev_mem_debug_set(true);
    gkyl_mem_debug_set(true);
  }

  struct gk_mirror_ctx ctx = create_ctx();

  int cells_x[ctx.cdim], cells_v[ctx.vdim];
  for (int d=0; d<ctx.cdim; d++)
    cells_x[d] = APP_ARGS_CHOOSE(app_args.xcells[d], ctx.cells[d]);
  for (int d=0; d<ctx.vdim; d++)
    cells_v[d] = APP_ARGS_CHOOSE(app_args.vcells[d], ctx.cells[ctx.cdim+d]);

  // Create decomposition.
  struct gkyl_rect_decomp *decomp = gkyl_gyrokinetic_comms_decomp_new(ctx.cdim, cells_x, app_args.cuts, app_args.use_mpi, stderr);

  // Construct communicator for use in app.
  struct gkyl_comm *comm = gkyl_gyrokinetic_comms_new(app_args.use_mpi, app_args.use_gpu, decomp, stderr);

  int my_rank = 0;
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
    gkyl_comm_get_rank(comm, &my_rank);
#endif

  struct gkyl_gyrokinetic_species ion = {
    .name = "ion",
    .charge = ctx.qi,
    .mass = ctx.mi,
    .lower = {-ctx.vpar_max_ion, 0.0},
    .upper = { ctx.vpar_max_ion, ctx.mu_max_ion},
    .cells = { cells_v[0], cells_v[1] },
    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_BIMAXWELLIAN, 
      .ctx_density = &ctx,
      .density = eval_density_ion,
      .ctx_upar = &ctx,
      .upar= eval_upar_ion,
      .ctx_temppar = &ctx,
      .temppar = eval_temp_par_ion,      
      .ctx_tempperp = &ctx,
      .tempperp = eval_temp_perp_ion,   
    },

    .collisions = {
      .collision_id = GKYL_LBO_COLLISIONS,
      .ctx = &ctx,
      .self_nu = evalNuIon,
    },

    .source = {
      .source_id = GKYL_PROJ_SOURCE,
      .write_source = true,
      .num_sources = 1,
      .projection[0] = {
        .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM, 
        .ctx_density = &ctx,
        .density = eval_density_ion_source,
        .ctx_upar = &ctx,
        .upar= eval_upar_ion_source,
        .ctx_temp = &ctx,
        .temp = eval_temp_ion_source,      
      }, 
    },

    .bcx = {
      .lower={.type = GKYL_SPECIES_GK_SHEATH,},
      .upper={.type = GKYL_SPECIES_GK_SHEATH,},
    },

    .num_diag_moments = 7,
    .diag_moments = {"M0", "M1", "M2", "M2par", "M2perp", "M3par", "M3perp"},
  };

  struct gkyl_gyrokinetic_field field = {
    .gkfield_id = GKYL_GK_FIELD_BOLTZMANN,
    .electron_mass = ctx.me,
    .electron_charge = ctx.qe,
    .electron_temp = ctx.Te0,
    .polarization_bmag = ctx.B_p, // Issue here. B0 from soloviev, so not sure what to do. Ours is not constant
    .fem_parbc = GKYL_FEM_PARPROJ_NONE,
  };

  struct gkyl_gk app_inp = {  // GK app
    .name = "gk_mirror_boltz_elc_1x2v_p1",
    .cdim = 1,
    .vdim = 2,
    .lower = {ctx.z_min},
    .upper = {ctx.z_max},
    .cells = { cells_x[0] },
    .poly_order = ctx.poly_order,
    .basis_type = app_args.basis_type,

    .geometry = {
      .geometry_id = GKYL_MAPC2P,
      .world = {ctx.psi_eval, 0.0},
      .mapc2p = mapc2p, // mapping of computational to physical space
      .c2p_ctx = &ctx,
      .bmag_func = bmag_func, // magnetic field magnitude
      .bmag_ctx = &ctx
    },

    .num_periodic_dir = 0,
    .periodic_dirs = {},

    .num_species = 1,
    .species = {ion},

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
  int frame_curr = 0;
  double t_curr = 0.0, t_end = ctx.t_end;
  // Initialize simulation.
  if (app_args.is_restart) {
    struct gkyl_app_restart_status status = gkyl_gyrokinetic_app_read_from_frame(app, app_args.restart_frame);

    if (status.io_status != GKYL_ARRAY_RIO_SUCCESS) {
      gkyl_gyrokinetic_app_cout(app, stderr, "*** Failed to read restart file! (%s)\n",
        gkyl_array_rio_status_msg(status.io_status));
      goto freeresources;
    }

    frame_curr = status.frame;
    t_curr = status.stime;

    gkyl_gyrokinetic_app_cout(app, stdout, "Restarting from frame %d", frame_curr);
    gkyl_gyrokinetic_app_cout(app, stdout, " at time = %g\n", t_curr);
  }
  else {
    gkyl_gyrokinetic_app_apply_ic(app, t_curr);
  }  

  // Create triggers for IO.
  int num_frames = ctx.num_frames, num_int_diag_calc = ctx.int_diag_calc_num;
  struct gkyl_tm_trigger trig_write = { .dt = t_end/num_frames, .tcurr = t_curr, .curr = frame_curr };
  struct gkyl_tm_trigger trig_calc_intdiag = { .dt = t_end/GKYL_MAX2(num_frames, num_int_diag_calc),
    .tcurr = t_curr, .curr = frame_curr };

  // Write out ICs (if restart, it overwrites the restart frame).
  calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, false);
  write_data(&trig_write, app, t_curr, false);

  // initial time step
  double dt = t_end - t_curr;
  // Initialize small time-step check.
  double dt_init = -1.0, dt_failure_tol = ctx.dt_failure_tol;
  int num_failures = 0, num_failures_max = ctx.num_failures_max;

  printf("Starting main loop ...\n");
  long step = 1, num_steps = app_args.num_steps;
  while ((t_curr < t_end) && (step <= num_steps))
  {
    gkyl_gyrokinetic_app_cout(app, stdout, "Taking time-step at t = %g ...", t_curr);
    struct gkyl_update_status status = gkyl_gyrokinetic_update(app, dt);
    gkyl_gyrokinetic_app_cout(app, stdout, " dt = %g\n", status.dt_actual);

    if (!status.success)
    {
      gkyl_gyrokinetic_app_cout(app, stdout, "** Update method failed! Aborting simulation ....\n");
      break;
    }
    t_curr += status.dt_actual;
    dt = status.dt_suggested;

    calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, t_curr > t_end);
    write_data(&trig_write, app, t_curr, t_curr > t_end);

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
        calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, true);
        write_data(&trig_write, app, t_curr, true);
        break;
      }
    }
    else {
      num_failures = 0;
    }

    step += 1;
  }
  printf(" ... finished\n");

  gkyl_gyrokinetic_app_stat_write(app);

  struct gkyl_gyrokinetic_stat stat = gkyl_gyrokinetic_app_stat(app); // fetch simulation statistics
  gkyl_gyrokinetic_app_cout(app, stdout, "\n");
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of update calls %ld\n", stat.nup);
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of forward-Euler calls %ld\n", stat.nfeuler);
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of RK stage-2 failures %ld\n", stat.nstage_2_fail);
  if (stat.nstage_2_fail > 0)
  {
    gkyl_gyrokinetic_app_cout(app, stdout, "Max rel dt diff for RK stage-2 failures %g\n", stat.stage_2_dt_diff[1]);
    gkyl_gyrokinetic_app_cout(app, stdout, "Min rel dt diff for RK stage-2 failures %g\n", stat.stage_2_dt_diff[0]);
  }
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of RK stage-3 failures %ld\n", stat.nstage_3_fail);
  gkyl_gyrokinetic_app_cout(app, stdout, "Species RHS calc took %g secs\n", stat.species_rhs_tm);
  gkyl_gyrokinetic_app_cout(app, stdout, "Species collisions RHS calc took %g secs\n", stat.species_coll_tm);
  gkyl_gyrokinetic_app_cout(app, stdout, "Field RHS calc took %g secs\n", stat.field_rhs_tm);
  gkyl_gyrokinetic_app_cout(app, stdout, "Species collisional moments took %g secs\n", stat.species_coll_mom_tm);
  gkyl_gyrokinetic_app_cout(app, stdout, "Updates took %g secs\n", stat.total_tm);
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of write calls %ld,\n", stat.nio);
  gkyl_gyrokinetic_app_cout(app, stdout, "IO time took %g secs \n", stat.io_tm);

  freeresources:
  // simulation complete, free app
  gkyl_gyrokinetic_app_release(app);
  gkyl_gyrokinetic_comms_release(decomp, comm);
  
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
    MPI_Finalize();
#endif
  return 0;
}