#pragma once

// Private header, not for direct use in user code

#include <math.h>
#include <gkyl_array.h>
#include <gkyl_wv_eqn.h>
#include <gkyl_eqn_type.h>
#include <gkyl_range.h>
#include <gkyl_util.h>

struct wv_iso_euler {
  struct gkyl_wv_eqn eqn; // base object
  double vt; // thermal velocity
};

/**
 * Free isothermal euler eqn object.
 *
 * @param ref Reference counter for isothermal euler eqn
 */
void gkyl_iso_euler_free(const struct gkyl_ref_count *ref);

GKYL_CU_D
static inline void
rot_to_local(const double *tau1, const double *tau2, const double *norm,
  const double *GKYL_RESTRICT qglobal, double *GKYL_RESTRICT qlocal)
{
  qlocal[0] = qglobal[0];
  qlocal[1] = qglobal[1]*norm[0] + qglobal[2]*norm[1] + qglobal[3]*norm[2];
  qlocal[2] = qglobal[1]*tau1[0] + qglobal[2]*tau1[1] + qglobal[3]*tau1[2];
  qlocal[3] = qglobal[1]*tau2[0] + qglobal[2]*tau2[1] + qglobal[3]*tau2[2];
}

GKYL_CU_D
static inline void
rot_to_global(const double *tau1, const double *tau2, const double *norm,
  const double *GKYL_RESTRICT qlocal, double *GKYL_RESTRICT qglobal)
{
  qglobal[0] = qlocal[0];
  qglobal[1] = qlocal[1]*norm[0] + qlocal[2]*tau1[0] + qlocal[3]*tau2[0];
  qglobal[2] = qlocal[1]*norm[1] + qlocal[2]*tau1[1] + qlocal[3]*tau2[1];
  qglobal[3] = qlocal[1]*norm[2] + qlocal[2]*tau1[2] + qlocal[3]*tau2[2];
}

// Waves and speeds using Roe averaging
GKYL_CU_D
static double
wave_roe(const struct gkyl_wv_eqn *eqn, 
  const double *delta, const double *ql, const double *qr, double *waves, double *s)
{
  const struct wv_iso_euler *iso_euler = container_of(eqn, struct wv_iso_euler, eqn);
  double vt = iso_euler->vt;

  double rhol = ql[0], rhor = qr[0];

  // Roe averages: see Roe's original 1981 paper or LeVeque book
  double srrhol = sqrt(rhol), srrhor = sqrt(rhor);
  double ravgl1 = 1/srrhol, ravgr1 = 1/srrhor;
  double ravg2 = 1/(srrhol+srrhor);
  double u = (ql[1]*ravgl1 + qr[1]*ravgr1)*ravg2;
  double v = (ql[2]*ravgl1 + qr[2]*ravgr1)*ravg2;
  double w = (ql[3]*ravgl1 + qr[3]*ravgr1)*ravg2;

  // Compute projections of jump
  double a0 = delta[0]*(vt+u)/vt/2.0-delta[1]/vt/2.0;
  double a1 = delta[2]-delta[0]*v;
  double a2 = delta[3]-delta[0]*w;
  double a3 = delta[0]*(vt-u)/vt/2.0+delta[1]/vt/2.0;

  double *wv;
  // Wave 1: eigenvalue is u-vt
  wv = &waves[0];
  wv[0] = a0;
  wv[1] = a0*(u-vt);
  wv[2] = a0*v;
  wv[3] = a0*w;
  s[0] = u-vt;

  // Wave 2: eigenvalue is u & u, two waves are lumped into one
  wv = &waves[4];
  wv[0] = 0.0;
  wv[1] = 0.0;
  wv[2] = a1;
  wv[3] = a2;
  s[1] = u;

  // Wave 3: eigenvalue is u+vt
  wv = &waves[8];
  wv[0] = a3;
  wv[1] = a3*(u+vt);
  wv[2] = a3*v;
  wv[3] = a3*w;
  s[2] = u+vt;
  
  return fabs(u)+vt;
}

GKYL_CU_D
static void
qfluct_roe(const struct gkyl_wv_eqn *eqn, 
  const double *ql, const double *qr, const double *waves, const double *s,
  double *amdq, double *apdq)
{
  const double *w0 = &waves[0], *w1 = &waves[4], *w2 = &waves[8];
  double s0m = fmin(0.0, s[0]), s1m = fmin(0.0, s[1]), s2m = fmin(0.0, s[2]);
  double s0p = fmax(0.0, s[0]), s1p = fmax(0.0, s[1]), s2p = fmax(0.0, s[2]);

  for (int i=0; i<4; ++i) {
    amdq[i] = s0m*w0[i] + s1m*w1[i] + s2m*w2[i];
    apdq[i] = s0p*w0[i] + s1p*w1[i] + s2p*w2[i];
  }
}

GKYL_CU_D
static double
max_speed(const struct gkyl_wv_eqn *eqn, const double *q)
{
  const struct wv_iso_euler *iso_euler = container_of(eqn, struct wv_iso_euler, eqn);
  double u = q[1]/q[0];
  return fabs(u) + iso_euler->vt;
}

/**
 * Compute flux. Assumes rotation to local coordinate system.
 * 
 * @param vt Thermal velocity
 * @param Conserved variables
 * @param flux On output, the flux in direction 'dir'
 */
GKYL_CU_D
static void
gkyl_iso_euler_flux(double vt, const double q[4], double flux[4])
{
  double u = q[1]/q[0];
  flux[0] = q[1]; // rho*u
  flux[1] = q[1]*u + q[0]*vt*vt; // rho*(u*u + vt*vt)
  flux[2] = q[2]*u; // rho*v*u
  flux[3] = q[3]*u; // rho*w*u
}