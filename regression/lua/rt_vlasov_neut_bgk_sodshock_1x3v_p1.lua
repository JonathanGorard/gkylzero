local Vlasov = G0.Vlasov

-- Physical constants (using normalized code units).
mass = 1.0 -- Neutral mass.
charge = 0.0 -- Neutral charge.

nl = 1.0 -- Left number density.
Tl = 1.0 -- Left temperature.

nr = 0.125 -- Right number density.
Tr = math.sqrt(0.1 / 0.125) -- Right temperature.

vt = 1.0 -- Thermal velocity.
Vx_drift = 0.0 -- Drift velocity (x-direction).
Vy_drift = 0.0 -- Drift velocity (y-direction).
Vz_drift = 0.0 -- Drift velocity (z-direction).
nu = 100.0 -- Collision frequency.

-- Simulation parameters.
Nx = 32 -- Cell count (configuration space: x-direction).
Nvx = 8 -- Cell count (velocity space: vx-direction).
Nvy = 8 -- Cell count (velocity space: vy-direction).
Nvz = 8 -- Cell count (velocity space: vz-direction).
Lx = 1.0 -- Domain size (configuration space: x-direction).
vx_max = 8.0 * vt -- Domain boundary (velocity space: vx-direction).
vy_max = 8.0 * vt -- Domain boundary (velocity space: vy-direction).
vz_max = 8.0 * vt -- Domain boundary (velocity space: vz-direction).
poly_order = 1 -- Polynomial order.
basis_type = "serendipity" -- Basis function set.
time_stepper = "rk3" -- Time integrator.
cfl_frac = 1.0 -- CFL coefficient.

t_end = 0.1 -- Final simulation time.
num_frames = 1 -- Number of output frames.

vlasovApp = Vlasov.App.new {

  tEnd = t_end,
  nFrame = num_frames,
  lower = { 0.0 },
  upper = { Lx },
  cells = { Nx },
  cflFrac = cfl_frac,

  basis = basis_type,
  polyOrder = poly_order,
  timeStepper = time_stepper,

  -- Decomposition for configuration space.
  decompCuts = { 1 }, -- Cuts in each coodinate direction (x-direction only).

  -- Boundary conditions for configuration space.
  periodicDirs = { }, -- Periodic directions (none).

  -- Neutral species.
  neut = Vlasov.Species.new {
    modelID = "default",
    charge = charge_neut, mass = mass_neut,
    
    -- Velocity space grid.
    lower = { -vx_max, -vy_max, -vz_max },
    upper = { vx_max, vy_max, vz_max },
    cells = { Nvx, Nvy, Nvz },

    -- Initial conditions.
    numInit = 1,
    projections = {
      {
        projectionID = "LTE",

        densityInit = function (t, xn)
          local x = xn[1]

          local n = 0.0
          if x < 0.5 then
            n = nl -- Total number density (left).
          else
            n = nr -- Total number density (right).
          end

          return n
        end,
        temperatureInit = function (t, xn)
          local x = xn[1]

          local T = 0.0
          if x < 0.5 then
            T = Tl -- Total temperature (left).
          else
            T = Tr -- Total temperature (right).
          end

          return T
        end,
        driftVelocityInit = function (t, xn)
          return Vx_drift, Vy_drift, Vz_drift -- Total drift velocity.
        end,

        correctAllMoments = true
      }
    },

    collisions = {
      collisionID = "BGK",

      selfNu = function (t, xn)
        return nu -- Collision frequency.
      end,
      
      correctAllMoments = true
    },

    evolve = true, -- Evolve species?
    diagnostics = { "M0", "M1i", "LTEMoments" }
  },

  skipField = true,

  -- Field.
  field = Vlasov.Field.new {
    epsilon0 = 1.0, mu0 = 1.0,

    -- Initial conditions function.
    init = function (t, xn)
      return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    end,

    evolve = false, -- Evolve field?
    elcErrorSpeedFactor = 0.0,
    mgnErrorSpeedFactor = 0.0,

    isStatic = true
  }
}

vlasovApp:run()