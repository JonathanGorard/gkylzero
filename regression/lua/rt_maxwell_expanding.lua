local Moments = G0.Moments
local Euler = G0.Moments.Eq.Euler

-- Mathematical constants (dimensionless).
pi = math.pi

-- Physical constants (using normalized code units).
epsilon0 = 1.0 -- Permittivity of free space.
mu0 = 1.0 -- Permeability of free space.

gas_gamma = 5.0 / 3.0 -- Adiabatic index.
U0 = 1.0 -- (Initial) comoving plasma velocity.
R0 = 1.0 -- (Initial) radial distance from expansion/contraction center.

E0 = 1.0 / math.sqrt(2.0) -- Reference electric field strength.
k_wave_x = 2.0 -- Wave number (x-direction).

-- Derived physical quantities (using normalized code units).
k_norm = math.sqrt(k_wave_x * k_wave_x) -- Wave number normalization factor.
k_xn = k_wave_x / k_norm -- Normalized wave number (x-direction).

-- Simulation parameters.
Nx = 512 -- Cell count (x-direction).
Lx = 1.0 -- Domain size (x-direction).
cfl_frac = 0.95 -- CFL coefficient.

t_end = 2.0 -- Final simulation time.
num_frames = 1 -- Number of output frames.

momentApp = Moments.App.new {
  
  tEnd = t_end,
  nFrame = num_frames,
  lower = { 0.0 },
  upper = { Lx },
  cells = { Nx },
  cflFrac = cfl_frac,
      
  -- Boundary conditions for configuration space.
  periodicDirs = { 1 }, -- Periodic directions (x-direction only).
    
  -- Fluid.
  fluid = Moments.Species.new {
    equation = Euler.new { gasGamma = gas_gamma },
  
    hasVolumeSources = true,
    volumeGasGamma = gas_gamma,
    volumeU0 = U0,
    volumeR0 = R0,
      
    -- Initial conditions function.
    init = function (t, xn)
      return 1.0, 0.0, 0.0, 0.0, 1.0
    end,
    
    evolve = true -- Evolve species?
  },
      
  -- Field.
  field = Moments.Field.new {
    epsilon0 = epsilon0, mu0 = mu0,

    -- Initial conditions function.
    init = function (t, xn)
      local x = xn[1]

      local phi = ((2.0 * pi) / Lx) * (k_wave_x * x)

      local Ex = 0.0 -- Total electric field (x-direction).
      local Ey = E0 * math.cos(phi) -- Total electric field (y-direction).
      local Ez = E0 * math.cos(phi) -- Total electric field (z-direction).

      local Bx = 0.0 -- Total magnetic field (x-direction).
      local By = -E0 * math.cos(phi) * k_xn -- Total magnetic field (y-direction).
      local Bz = E0 * math.cos(phi) * k_xn -- Total magnetic field (z-direction).

      return Ex, Ey, Ez, Bx, By, Bz, 0.0, 0.0
    end,

    evolve = true -- Evolve field?
  }
}
  
-- Run application.
momentApp:run()
  