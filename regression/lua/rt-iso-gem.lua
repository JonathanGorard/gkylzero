local Moments = G0.Moments
local IsoEuler = G0.Moments.Eq.IsoEuler

-- physical parameters
gasGamma = 5./3.
elcCharge = -1.0
ionCharge = 1.0
ionMass = 1.0
elcMass = ionMass/25.0
lightSpeed = 1.0
epsilon0 = 1.0
mu0 = 1.0

n0 = 1.0
VAe = 0.5
plasmaBeta = 1.0
lambdaOverDi0 = 0.5
TiOverTe = 5.0
nbOverN0 = 0.2
pert = 0.1
Valf = VAe*math.sqrt(elcMass/ionMass)
B0 = Valf*math.sqrt(n0*ionMass)
OmegaCi0 = ionCharge*B0/ionMass
psi0 = pert*B0

OmegaPe0 = math.sqrt(n0 * elcCharge^2 / (epsilon0 * elcMass))
de0 = lightSpeed / OmegaPe0
OmegaPi0 = math.sqrt(n0 * ionCharge^2 / (epsilon0 * ionMass))
di0 = lightSpeed / OmegaPi0
lambda = lambdaOverDi0 * di0

Ti_frac = TiOverTe / (1.0 + TiOverTe)
Te_frac = 1.0 / (1.0 + TiOverTe)
T_tot = plasmaBeta*(B0*B0)/2.0/n0
vti = math.sqrt(T_tot * Ti_frac / ionMass)
vte = math.sqrt(T_tot * Te_frac / elcMass)

-- domain size
Lx = 25.6 * di0
Ly = 12.8 * di0

momentApp = Moments.App.new {

   tEnd = 25.0/OmegaCi0,
   nFrame = 1,
   lower = {-Lx/2, -Ly/2},
   upper = {Lx/2, Ly/2},
   cells = {2*64, 2*32},
   cflFrac = 1.0,

   -- boundary conditions for configuration space
   periodicDirs = {1}, -- periodic directions

   -- electrons
   elc = Moments.Species.new {
      charge = elcCharge, mass = elcMass,

      equation = IsoEuler.new { vThermal = vte  },
      -- initial conditions
      init = function (t, xn)
	 local x, y = xn[1], xn[2]

	 local TeFrac = 1.0/(1.0 + TiOverTe)
	 local sech2 = (1.0/math.cosh(y/lambda))^2
	 local n = n0*(sech2 + nbOverN0)
	 local Jz = -(B0/lambda)*sech2
	 local Ttotal = plasmaBeta*(B0*B0)/2.0/n0

	 local rhoe = n*elcMass
	 local ezmom = (elcMass/elcCharge)*Jz*TeFrac
	 local ere = n*Ttotal*TeFrac/(gasGamma-1) + 0.5*ezmom*ezmom/rhoe
	 
	 return rhoe, 0.0, 0.0, ezmom
      end,
      evolve = true, -- evolve species?
      bcy = { G0.SpeciesBc.bcWall, G0.SpeciesBc.bcWall },
   },

   -- ions
   ion = Moments.Species.new {
      charge = ionCharge, mass = ionMass,

      equation = IsoEuler.new { vThermal = vti },
      -- initial conditions
      init = function (t, xn)
	 local x, y = xn[1], xn[2]

	 local TiFrac = TiOverTe/(1.0 + TiOverTe)
	 local sech2 = (1.0/math.cosh(y/lambda))^2
	 local n = n0*(sech2 + nbOverN0)
	 local Jz = -(B0/lambda)*sech2
	 local Ttotal = plasmaBeta*(B0*B0)/2.0/n0

	 local rhoi = n*ionMass
	 local izmom = (ionMass/ionCharge)*Jz*TiFrac
	 local eri = n*Ttotal*TiFrac/(gasGamma-1) + 0.5*izmom*izmom/rhoi
	 
	 return rhoi, 0.0, 0.0, izmom
      end,
      evolve = true, -- evolve species?
      bcy = { G0.SpeciesBc.bcWall, G0.SpeciesBc.bcWall },
   },

   field = Moments.Field.new {
      epsilon0 = 1.0, mu0 = 1.0,
      init = function (t, xn)
	 local x, y = xn[1], xn[2]

	 local Bxb = B0 * math.tanh(y / lambda) 
	 local Bx = Bxb - psi0 *(math.pi / Ly) * math.cos(2 * math.pi * x / Lx) * math.sin(math.pi * y / Ly) 
	 local By = psi0 * (2 * math.pi / Lx) * math.sin(2 * math.pi * x / Lx) * math.cos(math.pi * y / Ly)
	 local Bz = 0.0

	 return 0.0, 0.0, 0.0, Bx, By, Bz
      end,
      evolve = true, -- evolve field?
      bcy = { G0.FieldBc.bcWall, G0.FieldBc.bcWall },
   },

}
-- run application
momentApp:run()
