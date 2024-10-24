local Vlasov = G0.Vlasov

-- Mathematical constants (dimensionless).
pi = math.pi

-- Physical constants (using normalized code units).
epsilon0 = 1.0 -- Permittivity of free space.
mu0 = 1.0 -- Permeability of free space.
mass_elc = 1.0 -- Electron mass.
charge_elc = -1.0 -- Electron charge.
mass_ion = 1836.153 -- Ion mass.
charge_ion = 1.0 -- Ion charge.

Te_over_Ti = 4.0 -- Electron temperature / ion temperature.

-- Derived physical quantities (using normalized code units).
vte = 1.0 -- Electron thermal velocity.
vti = vte / math.sqrt(Te_over_Ti * mass_ion) -- Ion thermal velocity.
cs = vte / math.sqrt(mass_ion) -- Sound speed.

Vx_drift = 2.0 * cs -- Drift velocity (x-direction).

nu_elc = 1.0e-4 -- Electron collision frequency.
nu_ion = (nu_elc / math.sqrt(mass_ion)) * (Te_over_Ti * math.sqrt(Te_over_Ti)) -- Ion collision frequency.

-- Simulation parameters.
Nx = 32 -- Cell count (configuration space: x-direction).
Nvx = 12 -- Cell count (velocity space: vx-direction).
Nvy = 12 -- Cell count (velocity space: vy-direction).
Nvz = 12 -- Cell count (velocity space: vz-direction).
Lx = 256.0 -- Domain size (configuration space: x-direction).
vx_max_elc = 6.0 * vte -- Domain boundary (electron velocity space: vx-direction).
vy_max_elc = 6.0 * vte -- Domain boundary (electron velocity space: vy-direction).
vz_max_elc = 6.0 * vte -- Domain boundary (electron velocity space: vz-direction).
vx_max_ion = 16.0 * vti -- Domain boundary (ion velocity space: vx-direction).
vy_max_ion = 16.0 * vti -- Domain boundary (ion velocity space: vy-direction).
vz_max_ion = 16.0 * vti -- Domain boundary (ion velocity space: vz-direction).
poly_order = 2 -- Polynomial order.
basis_type = "serendipity" -- Basis function set.
time_stepper = "rk3" -- Time integrator.
cfl_frac = 1.0 -- CFL coefficient.

t_end = 20.0 -- Final simulation time.
num_frames = 1 -- Number of output frames.

vlasovApp = Vlasov.App.new {

  tEnd = t_end,
  nFrame = num_frames,
  lower = { -0.5 * Lx },
  upper = { 0.5 * Lx },
  cells = { Nx },
  cflFrac = cfl_frac,

  basis = basis_type,
  polyOrder = poly_order,
  timeStepper = time_stepper,

  -- Decomposition for configuration space.
  decompCuts = { 1 }, -- Cuts in each coodinate direction (x-direction only).

  -- Boundary conditions for configuration space.
  periodicDirs = { }, -- Periodic directions (none).

  -- Electrons.
  elc = Vlasov.Species.new {
    modelID = "default",
    charge = charge_elc, mass = mass_elc,
    
    -- Velocity space grid.
    lower = { -vx_max_elc, -vy_max_elc, -vz_max_elc },
    upper = { vx_max_elc, vy_max_elc, vz_max_elc },
    cells = { Nvx, Nvy, Nvz },

    -- Initial conditions.
    numInit = 1,
    projections = {
      {
        projectionID = "func",

        init = function (t, xn)
          local x, vx, vy, vz = xn[1], xn[2], xn[3], xn[4]

          local v_sq_m = ((vx - Vx_drift) * (vx - Vx_drift)) + (vy * vy) + (vz * vz)
          local v_sq_p = ((vx + Vx_drift) * (vx + Vx_drift)) + (vy * vy) + (vz * vz)

          local n = 0.0
          if x < 0.0 then
            n = (1.0 / math.pow(math.sqrt(2.0 * pi * vte * vte), 3.0)) * (math.exp(-v_sq_m / (2.0 * vte * vte))) -- Total number density (left).
          else
            n = (1.0 / math.pow(math.sqrt(2.0 * pi * vte * vte), 3.0)) * (math.exp(-v_sq_p / (2.0 * vte * vte))) -- Total number density (right).
          end

          return n
        end
      },
    },

    collisions = {
      collisionID = "LBO",

      selfNu = function (t, xn)
        return nu_elc -- Collision frequency.
      end
    },

    evolve = true, -- Evolve species?
    diagnostics = { "M0", "M1i", "M2" }
  },

  -- Ions.
  ion = Vlasov.Species.new {
    modelID = "default",
    charge = charge_ion, mass = mass_ion,

    -- Velocity space grid.
    lower = { -vx_max_ion, -vy_max_ion, -vz_max_ion },
    upper = { vx_max_ion, vy_max_ion, vz_max_ion },
    cells = { Nvx, Nvy, Nvz },
  
    -- Initial conditions.
    numInit = 1,
    projections = {
      {
        projectionID = "func",
  
        init = function (t, xn)
          local x, vx, vy, vz = xn[1], xn[2], xn[3], xn[4]
  
          local v_sq_m = ((vx - Vx_drift) * (vx - Vx_drift)) + (vy * vy) + (vz * vz)
          local v_sq_p = ((vx + Vx_drift) * (vx + Vx_drift)) + (vy * vy) + (vz * vz)
  
          local n = 0.0
          if x < 0.0 then
            n = (1.0 / math.pow(math.sqrt(2.0 * pi * vti * vti), 3.0)) * (math.exp(-v_sq_m / (2.0 * vti * vti))) -- Total number density (left).
          else
            n = (1.0 / math.pow(math.sqrt(2.0 * pi * vti * vti), 3.0)) * (math.exp(-v_sq_p / (2.0 * vti * vti))) -- Total number density (right).
          end

          return n
        end
      },
    },

    collisions = {
      collisionID = "LBO",

      selfNu = function (t, xn)
        return nu_ion -- Collision frequency.
      end
    },
  
    evolve = true, -- Evolve species?
    diagnostics = { "M0", "M1i", "M2" }
  },

  -- Field.
  field = Vlasov.Field.new {
    epsilon0 = epsilon0, mu0 = mu0,

    -- Initial conditions function.
    init = function (t, xn)
      local Ex = 0.0 -- Total electric field (x-direction).
      local Ey = 0.0 -- Total electric field (y-direction).
      local Ez = 0.0 -- Total electric field (z-direction).

      local Bx = 0.0 -- Total magnetic field (x-direction).
      local By = 0.0 -- Total magnetic field (y-direction).
      local Bz = 0.0 -- Total magnetic field (z-direction).

      return Ex, Ey, Ez, Bx, By, Bz, 0.0, 0.0
    end,

    evolve = true, -- Evolve field?
    elcErrorSpeedFactor = 0.0,
    mgnErrorSpeedFactor = 0.0
  }
}

vlasovApp:run()