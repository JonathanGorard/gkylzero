local Vlasov = G0.Vlasov

-- Mathematical constants (dimensionless).
pi = math.pi

-- Physical constants (using normalized code units).
mass_neut = 1.0 -- Top hat/bump mass.
charge_neut = 0.0 -- Top hat/bump charge.

n0 = 1.0 -- Reference number density.
ux0 = 0.0 -- Reference velocity (x-direction).
uy0 = 0.0 -- Reference velocity (y-direction).
vt = 1.0 / 3.0 -- Top hat Maxwellian thermal velocity.
nu = 0.01 -- Collision frequency.

ab = math.sqrt(0.1) -- Bump Maxwellian amplitude.
sb = 0.12 -- Bump Maxwellian softening factor, to avoid divergence.
ubx = 4.0 * math.sqrt(0.25 / 3.0) -- Bump location (x-direction in velocity space).
uby = 0.0 -- Bump location (y-direction, in velocity space).
vtb = 1.0 -- Bump Maxwellian thermal velocity.

-- Simulation parameters.
Nx = 2 -- Cell count (configuration space: x-direction).
Nvx = 16 -- Cell count (velocity space: vx-direction).
Nvy = 16 -- Cell count (velocity space: vy-direction).
Lx = 1.0 -- Domain size (configuration space: x-direction).
vx_max = 8.0 * vt -- Domain boundary (velocity space: vx-direction).
vy_max = 8.0 * vt -- Domain boundary (velocity space: vy-direction).
poly_order = 1 -- Polynomial order.
basis_type = "serendipity" -- Basis function set.
time_stepper = "rk3" -- Time integrator.
cfl_frac = 1.0 -- CFL coefficient.

t_end = 50.0 -- Final simulation time.
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
  periodicDirs = { 1 }, -- Periodic directions (x-direction only).

  -- Top hat species.
  square = Vlasov.Species.new {
    modelID = "default",
    charge = charge_neut, mass = mass_neut,
    
    -- Velocity space grid.
    lower = { -vx_max, -vy_max },
    upper = { vx_max, vy_max },
    cells = { Nvx, Nvy },

    -- Initial conditions.
    numInit = 1,
    projections = {
      {
        projectionID = "func",

        init = function (t, xn)
          local vx, vy = xn[2], xn[3]

          local n = 0.0
          if math.abs(vx) < 1.0 and math.abs(vy) < 1.0 then
            n = 0.5 * n0 -- Total number density (low velocity).
          else
            n = 0.0 -- Total number density (high velocity).
          end

          return n
        end
      }
    },

    collisions = {
      collisionID = "LBO",

      selfNu = function (t, xn)
        return nu -- Collision frequency.
      end,
      
      correctAllMoments = true
    },

    evolve = true, -- Evolve species?
    diagnostics = { "M0", "M1i", "M2" }
  },

  -- Bump species.
  bump = Vlasov.Species.new {
    modelID = "default",
    charge = charge_neut, mass = mass_neut,
    
    -- Velocity space grid.
    lower = { -vx_max, -vy_max },
    upper = { vx_max, vy_max },
    cells = { Nvx, Nvy },

    -- Initial conditions.
    numInit = 1,
    projections = {
      {
        projectionID = "func",

        init = function (t, xn)
          local vx, vy = xn[2], xn[3]

          local v_sq = ((vx - ux0) * (vx - ux0)) + ((vy - uy0) * (vy - uy0))
          local vb_sq = ((vx - ubx) * (vx - ubx)) + ((vy - uby) * (vy - uby))
          
          local n = (n0 / math.sqrt(2.0 * pi * vt * vt)) * math.exp(-v_sq / (2.0 * vt * vt)) + (n0 / math.sqrt(2.0 * pi * vtb * vtb)) *
            math.exp(-vb_sq / (2.0 * vtb * vtb)) * (ab * ab) / (vb_sq + (sb * sb)) -- Total number density.

          return n
        end
      }
    },

    collisions = {
      collisionID = "LBO",

      selfNu = function (t, xn)
        return nu -- Collision frequency.
      end,
      
      correctAllMoments = true
    },

    evolve = true, -- Evolve species?
    diagnostics = { "M0", "M1i", "M2" }
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