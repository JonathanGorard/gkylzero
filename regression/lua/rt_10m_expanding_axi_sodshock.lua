local Moments = G0.Moments
local TenMoment = G0.Moments.Eq.TenMoment

-- Mathematical constants (dimensionless).
pi = math.pi

-- Physical constants (using normalized code units).
gas_gamma = 5.0 / 3.0 -- Adiabatic index.
U0 = 1.0 -- (Initial) comoving plasma velocity.
R0 = 1.0 -- (Initial) radial distance from expansion/contraction center.

rhol = 3.0 -- Left/inner fluid mass density.
ul = 0.0 -- Left/inner fluid velocity.
pl = 3.0 -- Left/inner fluid pressure.

rhor = 1.0 -- Right/outer fluid mass density.
ur = 0.0 -- Right/outer fluid velocity.
pr = 1.0 -- Right/outer fluid pressure.

-- Simulation parameters.
Nr = 128 -- Cell count (radial direction).
Ntheta = 128 * 6 -- Cell count (angular direction).
Lr = 1.0 -- Domain size (radial direction).
Ltheta = 2.0 * pi -- Domain size (angular direction).
k0 = 0.1 -- Closure parameter.
cfl_frac = 0.9 -- CFL coefficient.

t_end = 0.2 -- Final simulation time.
num_frames = 1 -- Number of output frames.
rloc = 0.5 * (0.25 + 1.25) -- Fluid boundary (radial coordinate).

momentApp = Moments.App.new {
  tEnd = t_end,
  nFrame = num_frames,
  lower = { 0.25, 0.0 },
  upper = { 0.25 + Lr, Ltheta },
  cells = { Nr, Ntheta },
  cflFrac = cfl_frac,
    
  -- Boundary conditions for configuration space.
  periodicDirs = { }, -- Periodic directions (none).

  -- Computational coordinates (r, theta) to physical coordinates (x, y).
  mapc2p = function (t, zc)
    local r, theta = zc[1], zc[2]

    local xp = { }
    xp[1] = r * math.cos(theta)
    xp[2] = r * math.sin(theta)

    return xp[1], xp[2]
  end,
  
  -- Fluid.
  fluid = Moments.Species.new {
    equation = TenMoment.new { k0 = k0 },

    hasVolumeSources = true,
    volumeGasGamma = gas_gamma,
    volumeU0 = U0,
    volumeR0 = R0,
    
    -- Initial conditions function.
    init = function (t, zc)
      local r, theta = zc[1], zc[2]

      local rho = 0.0
      local u = 0.0
      local p = 0.0

      if r < rloc then
        rho = rhol -- Fluid mass density (left/inner).
        u = ul -- Fluid velocity (left/inner).
        p = pl -- Fluid pressure (left/inner).
      else
        rho = rhor -- Fluid mass density (right/outer).
        u = ur -- Fluid velocity (right/outer).
        p = pr -- Fluid pressure (right/outer).
      end
  
      local p_xx = p + (0.5 * rho * u * u)-- Fluid pressure tensor (xx-component).
      local p_xy = 0.0 -- Fluid pressure tensor (xy-component).
      local p_xz = 0.0 -- Fluid pressure tensor (xz-component).
      local p_yy = p -- Fluid pressure tensor (yy-component).
      local p_yz = 0.0 -- Fluid pressure tensor (yz-component).
      local p_zz = p -- Fluid pressure tensor (zz-component).
      
      return rho, mom_x, mom_y, mom_z, p_xx, p_xy, p_xz, p_yy, p_yz, p_zz
    end,
  
    evolve = true, -- Evolve species?
    bcx = { G0.SpeciesBc.bcCopy, G0.SpeciesBc.bcCopy }, -- Copy boundary conditions (x-direction).
    bcy = { G0.SpeciesBc.bcCopy, G0.SpeciesBc.bcCopy } -- Copy boundary conditions (y-direction).
  },
    
  -- Field.
  field = Moments.Field.new {
    epsilon0 = 1.0, mu0 = 1.0,

    -- Initial conditions function.
    init = function (t, xn)
      return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    end,

    evolve = false, -- Evolve field?
    bcx = { G0.FieldBc.bcCopy, G0.FieldBc.bcCopy }, -- Copy boundary conditions (x-direction).
    bcy = { G0.FieldBc.bcCopy, G0.FieldBc.bcCopy } -- Copy boundary conditions (y-direction).
  }
}

-- Run application.
momentApp:run()