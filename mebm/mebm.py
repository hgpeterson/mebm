################################################################################
# This file contains the class for a diffusive moist energy balance model.
#
# Henry G. Peterson with William R. Boos, 2019
################################################################################

import numpy as np
import scipy as sp
import scipy.integrate, scipy.sparse, scipy.optimize, scipy.interpolate
import climt
import sympl
from time import clock
import matplotlib.pyplot as plt
from matplotlib import rc
import os

# Constants 
ps = 98000     # kg/m/s2
cp = 1005      # J/kg/K
g = 9.81       # m/s2
D = 1.06e6     # m2/s
Re = 6.371e6   # m
RH = 0.8       # 0-1
S0 = 1365      # J/m2/s
R = 287.058    # J/kg/K
Lv = 2257000   # J/kg
sig = 5.67e-8  # J/s/m2/K4

# Styles:
package_path = os.path.abspath(os.path.dirname(__file__))
plt.style.use(package_path + "/plot_styles.mplstyle")

class MoistEnergyBalanceModel():
    """
    Diffusive moist energy balance model.
    """
    def __init__(self, N_pts=401, max_iters=1000, tol=1e-8, diffusivity="constant", control_file="default"):
        # Setup grid
        self.N_pts = N_pts
        self.dx = 2 / (N_pts - 1)
        self.sin_lats = np.linspace(-1.0, 1.0, N_pts)
        self.cos_lats = np.sqrt(1 - self.sin_lats**2)
        self.lats = np.arcsin(self.sin_lats)

        # Setup Diffusivity
        self.diffusivity = diffusivity
        if diffusivity == "constant":
            D_f = lambda x: 0*x + ps/g*D/Re**2
        elif diffusivity == "cesm2":
            cesm2_data = np.load(package_path + "/data/D_cesm2.npz")
            sin_lats_cesm2 = cesm2_data["sin_lats"]
            D_cesm2 = cesm2_data["D"]
            # D_cesm2 = cesm2_data["D_filtered"]
            x = np.arcsin(sin_lats_cesm2)
            y = D_cesm2
            D_f = sp.interpolate.interp1d(x, y, kind="quadratic")
        elif diffusivity == "D1":
            def D_f(lats):
                D_avg = ps / g * D / Re**2
                lat0 = 15
                L_trop = 2 * np.sin(np.deg2rad(lat0))
                L_extrop = 2 * (1 - np.sin(np.deg2rad(lat0)))
                D_trop = 4.5e-4
                D_extrop = (2*D_avg - D_trop*L_trop)/L_extrop
                Diff =  D_trop * np.ones(lats.shape)
                Diff[np.where(np.logical_or(np.rad2deg(lats) <= -lat0, np.rad2deg(lats) > lat0))] = D_extrop
                return Diff
        elif diffusivity == "D2":
            def D_f(lats):
                D_avg = ps / g * D / Re**2
                lat0 = 15
                L_trop = 2 * np.sin(np.deg2rad(lat0))
                L_extrop = 2 * (1 - np.sin(np.deg2rad(lat0)))
                D_trop = 0.5e-4
                D_extrop = (2*D_avg - D_trop*L_trop)/L_extrop
                Diff =  D_trop * np.ones(lats.shape)
                Diff[np.where(np.logical_or(np.rad2deg(lats) <= -lat0, np.rad2deg(lats) > lat0))] = D_extrop
                return Diff
        else:
            os.sys.exit("Unsupported D type.")

        self.D = D_f(self.lats)

        self.sin_lats_mids = (self.sin_lats - self.dx/2)[1:]
        self.lats_mids = np.arcsin(self.sin_lats_mids)
        self.D_mids = D_f(self.lats_mids)
        
        # Tolerance for error
        self.tol = tol

        # Max iterations
        self.max_iters = max_iters

        # Datasets for numpy search sorted
        self.T_dataset = np.arange(100, 400, 1e-3)
        self.q_dataset = self._humidsat(self.T_dataset, ps/100)[1]
        self.E_dataset = cp*self.T_dataset + RH*self.q_dataset*Lv

        # Boolean options
        self.plot_transports = False

        # Control data
        self.control_file = control_file
        if self.control_file == "default":
            self.control_file = package_path + "/data/ctrl.npz"
        self.ctrl_data = np.load(self.control_file)


    def _E_to_T(self, E):
        """
        Return temp given MSE.

        INPUTS
            E: array of MSE

        OUTPUTS
            array of temp
        """
        return self.T_dataset[np.searchsorted(self.E_dataset, E)]
        

    def _T_to_E(self, T):
        """
        Return MSE given temp.

        INPUTS
            T: array of temp

        OUTPUTS
            array of mse
        """
        return self.E_dataset[np.searchsorted(self.T_dataset, T)]


    def _humidsat(self, t, p):
        """
        FROM BOOS:
        % function [esat,qsat,rsat]=_humidsat(t,p)
        %  computes saturation vapor pressure (esat), saturation specific humidity (qsat),
        %  and saturation mixing ratio (rsat) given inputs temperature (t) in K and
        %  pressure (p) in hPa.
        %
        %  these are all computed using the modified Tetens-like formulae given by
        %  Buck (1981, J. Appl. Meteorol.)
        %  for vapor pressure over liquid water at temperatures over 0 C, and for
        %  vapor pressure over ice at temperatures below -23 C, and a quadratic
        %  polynomial interpolation for intermediate temperatures.
        """
        tc=t-273.16;
        tice=-23;
        t0=0;
        Rd=287.04;
        Rv=461.5;
        epsilon=Rd/Rv;
    
        # first compute saturation vapor pressure over water
        ewat=(1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*tc/(240.97+tc))
        eice=(1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*tc/(272.55+tc))
        # alternatively don"t use enhancement factor for non-ideal gas correction
        #ewat=6.1121.*exp(17.502.*tc./(240.97+tc));
        #eice=6.1115.*exp(22.452.*tc./(272.55+tc));
        eint=eice+(ewat-eice)*((tc-tice)/(t0-tice))**2
    
        esat=eint
        esat[np.where(tc<tice)]=eice[np.where(tc<tice)]
        esat[np.where(tc>t0)]=ewat[np.where(tc>t0)]
    
        # now convert vapor pressure to specific humidity and mixing ratio
        rsat=epsilon*esat/(p-esat);
        qsat=epsilon*esat/(p-esat*(1-epsilon));
        return esat, qsat, rsat


    def set_init_temp(self, init_temp_type, low=None, high=None):
        """
        Set the initial temperature distribution.

        INPUTS
            init_temp_type   : "triangle"  -> triangle in temp with max at eq
                               "legendre"  -> using first two legendre polys
                               "load_data" -> load T_array.npz data if it is in the folder
            low: lowest temp
            high: highest temp

        OUTPUTS
            Creates array init_temp saved to class.
        """
        self.init_temp_type = init_temp_type
        if init_temp_type == "triangle":
            self.init_temp = high - (high - low) * np.abs(self.sin_lats)
        elif init_temp_type == "legendre":
            self.init_temp = 2/3*high + 1/3*low - 2/3 * (high-low) * 1/2 * (3 * self.sin_lats**2 - 1)
        elif init_temp_type == "load_data":
            self.init_temp = np.load("simulation_data.npz")["T"][-1, :]
        else:
            os.sys.exit("Unsupported initial temperature type.")


    def set_insol(self, insol_type, perturb_center=None, perturb_spread=None, perturb_intensity=None):
        """
        Set the incoming shortwave radiation.

        INPUTS
            insol_type: "perturbation" -> as in Clark et al. 2018, a gaussian 
            perturb_center: degrees lat -> center of gaussian 
            perturb_spread: degrees lat -> spread of gaussian 
            perturb_intensity: W/m^2 -> M 

        OUTPUTS
            Creates array S saved to class.
        """
        self.insol_type = insol_type
        if insol_type == "perturbation":
            self.perturb_center = perturb_center
            self.perturb_spread = perturb_spread
            self.perturb_intensity = perturb_intensity

            S_basic = S0 / np.pi * self.cos_lats

            func = lambda y: 0.5 * np.exp(-(y - np.deg2rad(perturb_center))**2 / (2*np.deg2rad(perturb_spread)**2)) * np.cos(y)
            perturb_normalizer, er = sp.integrate.quadrature(func, -np.pi/2, np.pi/2, tol=1e-16, rtol=1e-16, maxiter=1000)

            self.dS = -perturb_intensity/perturb_normalizer * np.exp(-(self.lats - np.deg2rad(perturb_center))**2 / (2*np.deg2rad(perturb_spread)**2))
        else:
            os.sys.exit("Unsupported insolation type.")

        self.S = S_basic + self.dS


    def set_albedo(self, al_feedback=False, alb_ice=None, alb_water=None):
        """
        Set surface albedo.

        INPUTS
            al_feedback: boolean -> run model with changing albedo or not
            alb_ice: [0, 1] -> albedo of ice
            alb_water: [0, 1] -> albedo of water

        OUTPUTS
            Creates arrays alb, init_alb, and function reset_alb saved to class.
        """
        self.al_feedback = al_feedback
        self.alb_ice = alb_ice
        self.alb_water = alb_water
        if al_feedback == True:
            def reset_alb(T):
                alb = np.ones(self.N_pts)
                alb[:] = self.alb_water
                alb[np.where(T <= 273.16 - 10)] = alb_ice
                return alb

            self.init_alb = reset_alb(self.init_temp)
            self.reset_alb = reset_alb
        else:
            self.init_alb = self.ctrl_data["alb"]

        self.alb = self.init_alb


    def set_olr(self, olr_type, emissivity=None, A=None, B=None):
        """
        Set outgoing longwave radiation.

        INPUTS
            olr_type: "planck" -> L = epsillon * sigma * T^4
                      "linear" -> L = A + B * T 
                      "full_radiation" -> CliMT radiation scheme
                      "full_radiation_2xCO2" -> CliMT radiation scheme with doubled CO2 
                      "full_radiation_no_wv" -> CliMT radiation scheme with prescribed WV
                      "full_radiation_no_lr" -> CliMT radiation scheme with prescribed LR
            emissivity: [0, 1] -> epsillon for "planck" option
            A: float -> A for "linear" option
            B: float -> B for "linear" option

        OUTPUTS
            Creates function L(T) saved within the class.
        """
        self.olr_type = olr_type
        if olr_type == "planck":
            # PLANCK RADIATION
            L = lambda T: emissivity * sig * T**4 
        elif olr_type == "linear":
            # LINEAR FIT
            self.A = A
            self.B = B
            L = lambda T: self.A + self.B * T
        elif "full_radiation" in olr_type:
            # FULL BLOWN
            if "no_wv" in olr_type:
                wv_feedback = False
            else:
                wv_feedback = True

            if "no_lr" in olr_type:
                lr_feedback = False
            else:
                lr_feedback = True

            if "_rh" in olr_type:
                rh_feedback = True
            else:
                rh_feedback = False

            # Use CliMT radiation scheme along with MetPy"s moist adiabat calculator
            self.N_levels = 30 
            self.longwave_radiation = climt.RRTMGLongwave(cloud_overlap_method="clear_only")  
            grid = climt.get_grid(nx=1, ny=self.N_pts, nz=self.N_levels)
            grid["latitude"].values[:] = np.rad2deg(self.lats).reshape((self.N_pts, 1))  
            self.state = climt.get_default_state([self.longwave_radiation], grid_state=grid)  
            pressures = self.state["air_pressure"].values[:, 0, 0]

            if "2xCO2" in olr_type:
                self.state["mole_fraction_of_carbon_dioxide_in_air"].values[:] = 2 * self.state["mole_fraction_of_carbon_dioxide_in_air"].values[:]

            # RH profile
            gaussian = lambda mu, sigma, lat: np.exp(-(lat - mu)**2/(2*sigma**2)) 
            lowerlevels = np.where(pressures/100 > 875)[0]   
            midlevels = np.where(np.logical_and(pressures/100 < 875, pressures/100 > 200))[0]   
            midupperlevels = np.where(np.logical_and(pressures/100 < 200, pressures/100 > 100))[0]   
            def generate_RH_dist(lat_center):
                """
                Make the RH_lat_profile a gaussian and shift its max to the EFE.
                """
                RH_dist = np.zeros((self.N_levels, self.N_pts, 1))

                # Set up lower levels as constant
                RH_dist[lowerlevels, :, 0] = 0.9 

                # Set up mid levels as four gaussians
                width_center = np.sin(np.deg2rad(30))
                width_left = 1 + np.sin(lat_center) - width_center/2
                width_right = 1 - np.sin(lat_center) - width_center/2
                
                left = np.where(self.sin_lats <= width_left - 1)[0]
                centerL = np.where(np.logical_and(self.sin_lats > np.sin(lat_center) - width_center/2, self.sin_lats <= np.sin(lat_center)))[0]
                centerR = np.where(np.logical_and(self.sin_lats > np.sin(lat_center), self.sin_lats < np.sin(lat_center) + width_center/2))[0]
                right = np.where(self.sin_lats >= 1 - width_right)[0]

                spread_left = 1/4*width_left
                spread_centerL = 1/8*width_center
                spread_centerR = 1/8*width_center
                spread_right = 1/4*width_right

                ## RH Feedback:
                RH_min_ctrl = 0.145
                if rh_feedback:
                    slope_L = -0.0203954292274862
                    slope_R = 0.013515143699796586
                else:
                    slope_L = 0
                    slope_R = 0

                RH_L_max = 0.9
                RH_R_max = 0.9
                RH_C_max = 0.8
                RH_L_min = RH_min_ctrl + slope_L*np.rad2deg(lat_center)
                RH_R_min = np.max([RH_min_ctrl + slope_R*np.rad2deg(lat_center), 0])

                # RH_dist[midlevels, :, 0] = RH
                RH_dist[midlevels, left[0]:left[-1]+1, 0] = np.repeat( 
                    RH_L_min + (RH_L_max - RH_L_min) * gaussian(-1, spread_left, self.sin_lats[left]), 
                    len(midlevels)).reshape( (len(left), len(midlevels)) ).T
                RH_dist[midlevels, centerL[0]:centerL[-1]+1, 0] = np.repeat( 
                    RH_L_min + (RH_C_max - RH_L_min) * gaussian(np.sin(lat_center), spread_centerL, self.sin_lats[centerL]), 
                    len(midlevels)).reshape( (len(centerL), len(midlevels)) ).T
                RH_dist[midlevels, centerR[0]:centerR[-1]+1, 0] = np.repeat( 
                    RH_R_min + (RH_C_max - RH_R_min) * gaussian(np.sin(lat_center), spread_centerR, self.sin_lats[centerR]), 
                    len(midlevels)).reshape( (len(centerR), len(midlevels)) ).T
                RH_dist[midlevels, right[0]:right[-1]+1, 0] = np.repeat( 
                    RH_R_min + (RH_R_max - RH_R_min) * gaussian(1, spread_right, self.sin_lats[right]), 
                    len(midlevels)).reshape( (len(right), len(midlevels)) ).T

                # Set up upper levels as one gaussian
                RH_dist[midupperlevels, :, 0] = np.repeat( 
                    0.6 * gaussian(np.sin(lat_center), np.sin(np.deg2rad(20)), self.sin_lats), 
                    len(midupperlevels)).reshape( (self.N_pts, len(midupperlevels)) ).T
                return RH_dist
            self.generate_RH_dist = generate_RH_dist 
            lat_efe = 0 
            self.RH_dist = self.generate_RH_dist(lat_efe)

            # # Save RH dists for plotting:
            # np.savez("RH_M0_mebm.npz", RH=self.RH_dist, lats=self.lats, pressures=pressures)
            # np.savez("RH_shifted_mebm.npz", RH=self.generate_RH_dist(np.deg2rad(-10.74)), lats=self.lats, pressures=pressures)
            # np.savez("RH_M18_mebm.npz", RH=self.generate_RH_dist(np.deg2rad(-15.15)), lats=self.lats, pressures=pressures)
            # os.sys.exit()

            # # Debug: Plot RH dist
            # f, ax = plt.subplots(1)
            # levels = np.arange(0, 1.05, 0.05)
            # cf = ax.contourf(self.sin_lats, pressures/100, self.RH_dist[:, :, 0], cmap="BrBG", levels=levels)
            # cb = plt.colorbar(cf, ax=ax, pad=0.1, fraction=0.2)
            # cb.set_ticks(np.arange(0, 1.05, 0.1))
            # ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            # ax.set_xticklabels(["90°S", "", "", "60°S", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "60°N", "", "", "90°N"])
            # ax.set_yticks(np.arange(0,1001,100))
            # plt.gca().invert_yaxis()
            # plt.grid(True)
            # plt.savefig("RH.png")
            # plt.close()

            # Create the 2d interpolation function: gives function T_moist(T_surf, p)
            moist_data = np.load(package_path + "/data/moist_adiabat_data.npz")    
            T_surf_sample = moist_data["T_surf_sample"]    
            T_data = moist_data["T_data"]    

            pressures_flipped = np.flip(pressures, axis=0) 
            T_data = np.flip(T_data, axis=1)
            self.interpolated_moist_adiabat = sp.interpolate.RectBivariateSpline(T_surf_sample, pressures_flipped, T_data)   

            if wv_feedback == False:
                # prescribe WV from control simulation
                T_control = self.ctrl_data["ctrl_state_temp"][0, :, 0]    
                Tgrid_control = np.repeat(T_control, self.N_levels).reshape( (self.N_pts, self.N_levels) )
                air_temp = self.interpolated_moist_adiabat.ev(Tgrid_control, pressures).T.reshape( (self.N_levels, self.N_pts, 1) )
                self.state["specific_humidity"].values[:] = self.RH_dist * self._humidsat(air_temp, self.state["air_pressure"].values[:] / 100)[1]    

            self.pressures = pressures
            self.pressures_flipped = pressures_flipped

            def L(T):
                """ 
                OLR function.
                Outputs OLR given T_surf.
                Assumes moist adiabat structure, uses full blown radiation code from CliMT.
                Sets temp profile with interpolation of moist adiabat calculations from MetPy.
                Sets specific hum profile by assuming constant RH and using _humidsat function from Boos
                """
                # Set surface state
                self.state["surface_temperature"].values[:] = T.reshape((self.N_pts, 1))
                if lr_feedback == False:
                    # Retain LR from control simulations by just shifting all levels by difference at surface
                    Tgrid_diff = np.repeat(T - self.ctrl_data["ctrl_state_temp"][0, :, 0], self.N_levels).reshape((self.N_pts, self.N_levels)).T.reshape((self.N_levels, self.N_pts, 1))
                    self.state["air_temperature"].values[:] = self.ctrl_data["ctrl_state_temp"][:] + Tgrid_diff
                else:
                    # Create a 2D array of the T vals and pass to self.interpolated_moist_adiabat
                    Tgrid = np.repeat(T, self.N_levels).reshape( (self.N_pts, self.N_levels) )
                    self.state["air_temperature"].values[:] = self.interpolated_moist_adiabat.ev(Tgrid, pressures).T.reshape( (self.N_levels, self.N_pts, 1) )
                if wv_feedback == True:
                    # Shift RH_dist based on ITCZ
                    self._calculate_efe()
                    self.RH_dist = self.generate_RH_dist(self.EFE)
                    # Recalculate q
                    self.state["specific_humidity"].values[:] = self.RH_dist * self._humidsat(self.state["air_temperature"].values[:], self.state["air_pressure"].values[:] / 100)[1]
                tendencies, diagnostics = self.longwave_radiation(self.state)
                return diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0]

            self.lr_feedback = lr_feedback
            self.wv_feedback = wv_feedback
            self.rh_feedback = rh_feedback
        elif olr_type == "no_feedback":
            # NO LW FEEDBACKS
            dS_bar = self._area_weighted_avg(self.dS*(1 - self.alb))
            self.L_ctrl = self.ctrl_data["L"]
            def L(T):
                return self.L_ctrl + dS_bar
        else:
            os.sys.exit("Unsupported OLR type.")

        # save to class
        self.L = L  


    def _restrict(self, array):
        array_2h = array[::2]
        return array_2h

    
    def _prolongate(self, array_2h):
        array = np.zeros(2*array_2h.shape[0] - 1)
        array[::2] = array_2h
        array[1::2] = ((array_2h + np.roll(array_2h, 1))/2)[1:]
        return array

        
    def _compute_mats(self, grid_num):
        if grid_num == 0:
            C_mids = -self.D_mids / self.dx**2 * (1 - self.sin_lats_mids**2)
        else:
            sin_lats_mids = self.sin_lats_mids[::2**grid_num]
            D_mids = self.D_mids[::2**grid_num]
            C_mids = -D_mids / self.dx**2 * (1 - sin_lats_mids**2)

        N = self.Ns[grid_num]

        diag = np.zeros(N)
        for i in range(1, N-1):
            diag[i] = -(C_mids[i] + C_mids[i-1])
        diag[0] = -C_mids[0]
        diag[N-1] = -C_mids[-1]
        upper_diag = C_mids
        lower_diag = C_mids
        D = sp.sparse.diags(diag, 0)
        L = sp.sparse.diags(upper_diag, 1)
        U = sp.sparse.diags(lower_diag, -1)

        # Gauss-Sidel:
        LHS = (D + L).tocsc()
        RHS = -U

        LHS_LU = sp.sparse.linalg.splu(LHS)
        return D, L, U, LHS_LU, RHS

    
    def _smoothing(self, its, u, f, LHS, RHS, tol=0):
        if tol > 0:
            error = tol
            it = 0
            while error < tol and it < its:
                u_new = LHS.solve(RHS.dot(u) + f)
                error = np.max(np.abs(u_new - u))
                u = u_new
                it += 1
        else:
            for i in range(its):
                u = LHS.solve(RHS.dot(u) + f)
        return u

        
    def V_cycle(self, u, f, grid_num):
        smoothing_its = 20
        u = self._smoothing(smoothing_its, u, f, self.LHSs[grid_num], self.RHSs[grid_num])
        
        r = f - self.As[grid_num].dot(u)
        r_2h = self._restrict(r)
    
        e_2h = np.zeros(self.Ns[grid_num+1])
    
        if grid_num+1 == self.grid_nums[-1]:
            e_2h = self._smoothing(10*smoothing_its, e_2h, r_2h, self.LHSs[grid_num+1], self.RHSs[grid_num+1], tol=1e-10)
        else:
            e_2h = self.V_cycle(e_2h, r_2h, grid_num+1)
    
        e = self._prolongate(e_2h)
        u += e
        
        u = self._smoothing(smoothing_its, u, f, self.LHSs[grid_num], self.RHSs[grid_num])
        return u


    def solve(self, numerical_method, frames):
        """
        Loop through integration time steps.

        INPUTS 
            numerical_method: "implicit" -> fully implicit method
            frames: int -> number of steps of data to save
        
        OUTPUTS
           Creates set of T, E, alb, and q arrays saved to the class. 
        """
        # begin by computing the take_step_matrix for particular scheme
        self.numerical_method = numerical_method
        
        if numerical_method == "multigrid":
            self.grid_nums = range(np.min([int(np.log2(self.N_pts)), 6]))
            self.Ns = [self.N_pts]
            for grid_num in range(1, len(self.grid_nums)):
                self.Ns.append(self.N_pts//(2**grid_num) + 1)

            self.LHSs = []
            self.RHSs = []
            self.As = []
            for grid_num in self.grid_nums:
                D, L, U, LHS, RHS = self._compute_mats(grid_num)
                A = D + L + U
                self.LHSs.append(LHS)
                self.RHSs.append(RHS)
                self.As.append(A)
        else:
            os.sys.exit("Invalid numerical method.")

        # Print some useful information
        print("\nModel Params:")
        print("grids:            {}".format(self.Ns))
        print("dx:               {:.5f}".format(self.dx))
        print("max dlat:         {:.5f}".format(np.rad2deg(np.max(np.abs( (np.roll(self.lats, -1) - self.lats)[:-1])))))
        print("min dlat:         {:.5f}".format(np.rad2deg(np.min(np.abs( (np.roll(self.lats, -1) - self.lats)[:-1])))))
        print("tolerance:        {:.2E}".format(self.tol))
        print("frames:           {}".format(frames))
        
        print("\nDiffusivity Type:  {}".format(self.diffusivity))
        print("Insolation Type:   {}".format(self.insol_type))
        if self.insol_type == "perturbation":
            print("\tlat0 = {:.0f}, M = {:.0f}, sigma = {:.2f}".format(
                self.perturb_center, self.perturb_intensity, self.perturb_spread))
        print("Initial Temp Dist: {}".format(self.init_temp_type))
        print("AL Feedback:       {}".format(self.al_feedback))
        print("OLR Scheme:        {}".format(self.olr_type))
        if self.olr_type == "linear":
            print("\tA = {:.2f}, B = {:.2f}".format(self.A, self.B))
        elif "full_radiation" in self.olr_type:
            print("\tLR Feedback:       {}".format(self.lr_feedback))
            print("\tWV Feedback:       {}".format(self.wv_feedback))
            print("\tRH Feedback:       {}".format(self.rh_feedback))
        print("control sim:       {}".format(self.control_file))
        print("Numerical Method:  {}\n".format(self.numerical_method))
        
        # Setup arrays to save data in
        T_array = np.zeros((frames, self.lats.shape[0]))
        alb_array = np.zeros((frames, self.lats.shape[0]))
        L_array = np.zeros((frames, self.lats.shape[0]))
        
        self.T = self.init_temp
        self.E = self._T_to_E(self.T)
        self.alb = self.init_alb

        # Loop through self.take_step() until converged
        t0 = clock()
        its_per_frame = int(self.max_iters / (frames - 1))
        error = self.tol + 1
        iteration = 0
        frame = 0
        while frame < frames:
            if iteration % its_per_frame == 0:
                T_array[frame, :] = self.T
                L_array[frame, :] = self.L(self.T)
                alb_array[frame, :] = self.alb

                # error = np.sqrt(self._area_weighted_avg((T_array[frame, :] - T_array[frame-1, :])**2))
                error = self._area_weighted_avg(np.abs(T_array[frame, :] - T_array[frame-1, :]))

                # Print progress 
                T_avg = self._area_weighted_avg(self.T)
                self._calculate_efe()
                energy_balance = np.abs(10**-15 * self._integrate_lat(self.S*(1-self.alb) - L_array[frame, :]))
                if frame == 0:
                    if self.EFE >= 0:
                        print("frame = {:5d}; EFE = {:2.3f}°N; T_avg = {:3.1f} K; NEI = {:.2e} PW".format(0, np.rad2deg(self.EFE), T_avg, energy_balance))
                    else:
                        print("frame = {:5d}; EFE = {:2.3f}°S; T_avg = {:3.1f} K; NEI = {:.2e} PW".format(0, -np.rad2deg(self.EFE), T_avg, energy_balance))
                else:
                    if self.EFE >= 0:
                        print("frame = {:5d}; EFE = {:2.3f}°N; T_avg = {:3.1f} K; NEI = {:.2e} PW; Change = {:.2e} K".format(frame, np.rad2deg(self.EFE), T_avg, energy_balance, error))
                    else:
                        print("frame = {:5d}; EFE = {:2.3f}°S; T_avg = {:3.1f} K; NEI = {:.2e} PW; Change = {:.2e} K".format(frame, -np.rad2deg(self.EFE), T_avg, energy_balance, error))
                if error < self.tol:
                    break
                frame += 1

            # do a cycle
            if self.N_pts > 512:
                self.E = self.V_cycle(self.E, self.S*(1 - self.alb) - L_array[frame-1, :], 0)
            else:
                self.E = self.V_cycle(self.E, self.S*(1 - self.alb) - self.L(self.T), 0)
            T_old = self.T
            self.T = self._E_to_T(self.E)
            if self.al_feedback:
                self.alb = self.reset_alb(self.T)
            iteration += 1


        tf = clock()
        sim_time = tf - t0

        self.T_f = self.T
        self.E_f = self.E
        self.L_f = self.L(self.T_f)
        self.alb_f = self.alb
        self.S_f = self.S 

        # Truncate arrays
        if iteration-1 % its_per_frame == 0:
            frame += 1
        T_array = T_array[:frame, :]
        alb_array = alb_array[:frame, :]
        L_array = L_array[:frame, :]

        # Print exit messages
        if frame == frames:
            print("Failed to reach equilibrium in {} iterations".format(iteration))
        else:
            print("Equilibrium reached in {} iterations.".format(iteration))

        
        print("\nEfficiency: \n{:10.10f} total seconds\n{:10.10f} seconds/iteration\n".format(sim_time, sim_time / iteration))

        # Save arrays to class
        self.T_array = T_array
        self.alb_array = alb_array
        self.L_array = L_array


    def save_data(self, control=False):
        """
        Save arrays of state variables.

        INPUTS

        OUTPUTS
        """
        trans_total = self._calculate_trans(self.S_f * (1 - self.alb_f) - self.L(self.T_f), force_zero=True)
        np.savez("simulation_data.npz", T=self.T_array, L=self.L_array, alb=self.alb_array, sin_lats=self.sin_lats, trans_total=trans_total)
        # Save control simulations
        if control:
            if "full_radiation" in self.olr_type:
                L_bar = self._area_weighted_avg(self.L(self.T_f))
                ctrl_state_temp = self.state["air_temperature"].values[:]
                fname = "ctrl.npz"
                np.savez(fname, S=self.S_f, L=self.L(self.T_f), L_bar=L_bar, trans_total=trans_total, ctrl_state_temp=ctrl_state_temp, alb=self.alb, RH_dist=self.RH_dist)
                print("{} created\n".format(fname))
            else:
                print("\nCannot save as control simulation without CliMT state.\n")


    def _calculate_efe(self):
        """
        EFE = latitude of max of E

        INPUTS

        OUTPUTS
            Creates float EFE saved to class.
        """
        # Interp and find roots
        spl = sp.interpolate.UnivariateSpline(self.lats, self.E, k=4, s=0)
        roots = spl.derivative().roots()
        
        # Find supposed root based on actual data
        max_index = np.argmax(self.E)
        efe_lat = self.lats[max_index]
        
        # Pick up closest calculated root to the supposed one
        min_error_index = np.argmin( np.abs(roots - efe_lat) )
        closest_root = roots[min_error_index]

        self.EFE = closest_root


    def log_efe(self, fname_efe):
        """
        Write EFE data to a file.

        INPUTS
            fname_efe: string -> name of file to save to

        OUTPUTS
        """
        print("Calculating EFE...")
        self._calculate_efe()

        with open(fname_efe, "a") as f:
            if self.insol_type == "perturbation":
                data = "{:2d}, {:2.2f}, {:2d}, {:2.16f}".format(self.perturb_center, self.perturb_spread, self.perturb_intensity, np.rad2deg(self.EFE))
            else:
                data = "{:2d}, {:2.2f}, {:2d}, {:2.16f}".format(0, 0, 0, np.rad2deg(self.EFE))
            f.write(data + "\n")
        print("Logged '{}' in {}".format(data, fname_efe))


    def _integrate_lat(self, f, i=-1):
        """
        Integrate some array f over x up to index i.

        INPUTS
            f: float, array -> some array or constant to integrate
            i: int -> index in array to integrate up to (default all).

        OUTPUTS
            Returns integral using trapezoidal method.
        """
        if isinstance(f, np.ndarray):
            if i == -1:
                return  2*np.pi*Re**2 * np.trapz(f, dx=self.dx) 
            else:
                return  2*np.pi*Re**2 * np.trapz(f[:i+1], dx=self.dx) 
        else:
            if i == -1:
                return  2*np.pi*Re**2 * np.trapz(f * np.ones(self.N_pts), dx=self.dx) 
            else:
                return  2*np.pi*Re**2 * np.trapz(f * np.ones(self.N_pts)[:i+1], dx=self.dx) 


    def _area_weighted_avg(self, f):
        """
        Return area weighted average of some array f.

        INPUTS
            f: float, array -> some array or constant to integrate

        OUTPUTS
            f_bar
        """
        return 1/self._integrate_lat(1) * self._integrate_lat(f)
    

    def _calculate_trans(self, f, force_zero=False):
        """
        Perform integral calculation to get energy transport.

        INPUTS
            f: array 
            force_zero: boolean

        OUTPUTS
            Returns transport array
        """
        if force_zero:
            f_bar = self._area_weighted_avg(f)
        if isinstance(f, np.ndarray):
            trans = np.zeros(f.shape)
        else:
            trans = np.zeros(self.N_pts)
        for i in range(self.N_pts):
            if force_zero:
                trans[i] = self._integrate_lat(f - f_bar, i)
            else:
                trans[i] = self._integrate_lat(f, i)
        return trans


    def log_feedbacks(self, fname_feedbacks):
        """
        Calculate each feedback transport and log data on the shifts.

        INPUTS
            fname_feedbacks: string -> file to write to.

        OUTPUTS
            Creates arrays for each feedback transport saved to class.
        """
        if "full_radiation" not in self.olr_type:
            print("\nCannot calculate feedbacks without CliMT state.\n")
        else:
            print("\nCalculating feedbacks...")

            self.plot_transports = True

            self.L_bar = self._area_weighted_avg(self.L_f)

            # Get ctrl data
            ctrl_state_temp = self.ctrl_data["ctrl_state_temp"]
            pert_state_temp = np.copy(self.state["air_temperature"].values[:])

            ctrl_state_RH_dist = self.ctrl_data["RH_dist"]
            ctrl_state_qstar = self._humidsat(ctrl_state_temp, self.state["air_pressure"].values[:] / 100)[1]
            ctrl_state_q = ctrl_state_RH_dist * ctrl_state_qstar
            pert_state_RH_dist = self.generate_RH_dist(self.EFE)
            pert_state_qstar = self._humidsat(pert_state_temp, self.state["air_pressure"].values[:] / 100)[1]
            pert_state_q = np.copy(self.state["specific_humidity"].values[:])

            self.S_ctrl = self.ctrl_data["S"]
            self.alb_ctrl = self.ctrl_data["alb"]
            self.L_bar_ctrl = self.ctrl_data["L_bar"]
            self.dalb = self.alb_f - self.alb_ctrl

            self.trans_total_ctrl = self.ctrl_data["trans_total"]

            self.T_f_ctrl = ctrl_state_temp[0, :, 0]
            self.ctrl_state_temp = ctrl_state_temp
            self.pert_state_temp = pert_state_temp
            self.ctrl_state_q = ctrl_state_q
            self.pert_state_q = pert_state_q

            self.L_ctrl = self.ctrl_data["L"]
            self.dL = self.L_f - self.L_ctrl

            ## dS
            self.dtrans_dS = self._calculate_trans(self.dS * (1 - self.alb_ctrl), force_zero=True)

            ## dalb
            self.dtrans_dalb = self._calculate_trans((self.S_ctrl + self.dS) * self.dalb, force_zero=True)

            ## Total
            self.trans_total = self._calculate_trans(self.S_f*(1 - self.alb_f) - self.L_f, force_zero=True)
            self.dtrans_total = self.trans_total - self.trans_total_ctrl

            ## Planck
            Tgrid_diff = np.repeat(pert_state_temp[0, :, 0] - ctrl_state_temp[0, :, 0], self.N_levels).reshape((self.N_pts, self.N_levels)).T.reshape((self.N_levels, self.N_pts, 1))
            self.state["air_temperature"].values[:] = pert_state_temp - Tgrid_diff
            self.state["surface_temperature"].values[:] = self.state["air_temperature"].values[0, :, :]
            self.state["specific_humidity"].values[:] = pert_state_q
            tendencies, diagnostics = self.longwave_radiation(self.state)
            self.dL_pl = (self.L_f - diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0])
            self.dtrans_pl = self._calculate_trans(self.dL_pl, force_zero=True)

            ## Water Vapor 
            self.state["air_temperature"].values[:] = pert_state_temp
            self.state["surface_temperature"].values[:] = pert_state_temp[0, :, :]
            self.state["specific_humidity"].values[:] = ctrl_state_q
            # self.state["specific_humidity"].values[:] = pert_state_RH_dist * ctrl_state_qstar
            tendencies, diagnostics = self.longwave_radiation(self.state)
            self.dL_wv =  (self.L_f - diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0])
            self.dtrans_wv = self._calculate_trans(self.dL_wv, force_zero=True)

            # ## Relative Humidity
            # self.state["air_temperature"].values[:] = pert_state_temp
            # self.state["surface_temperature"].values[:] = pert_state_temp[0, :, :]
            # self.state["specific_humidity"].values[:] = ctrl_state_RH_dist * pert_state_qstar
            # tendencies, diagnostics = self.longwave_radiation(self.state)
            # self.dL_rh =  (self.L_f - diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0])
            # self.dtrans_rh = self._calculate_trans(self.dL_rh, force_zero=True)
            
            ## Lapse Rate
            Tgrid_diff = np.repeat(pert_state_temp[0, :, 0] - ctrl_state_temp[0, :, 0], self.N_levels).reshape((self.N_pts, self.N_levels)).T.reshape((self.N_levels, self.N_pts, 1))
            self.state["air_temperature"].values[:] =  ctrl_state_temp + Tgrid_diff
            self.state["surface_temperature"].values[:] = self.state["air_temperature"].values[0, :, :]
            self.state["specific_humidity"].values[:] = pert_state_q
            tendencies, diagnostics = self.longwave_radiation(self.state)
            self.dL_lr = (self.L_f - diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0])
            self.dtrans_lr = self._calculate_trans(self.dL_lr, force_zero=True)


    def save_plots(self):
        """
        Plot various data from the simulation

        INPUTS

        OUTPUTS
        """
        ### TEMP 
        print("\nPlotting T")
        
        T_avg = self._area_weighted_avg(self.T_f)
        print("Mean T: {:.2f} K".format(T_avg))

        f, ax = plt.subplots(1)
        ax.plot(self.sin_lats, self.T_f, "k", label="Mean $T={:.2f}$ K".format(T_avg))
        ax.set_title("Final Temperature")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("$T$ (K)")
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
        ax.legend(loc="lower center")
        
        plt.tight_layout()
        
        fname = "temp.png"
        plt.savefig(fname)
        print("{} created.".format(fname))
        plt.close()
        
        ### dTEMP 
        print("\nPlotting dT")
        
        T_ctrl = self.ctrl_data["ctrl_state_temp"][0, :, 0]

        f, ax = plt.subplots(1)
        ax.plot(self.sin_lats, self.T_f - T_ctrl, "k")
        ax.set_title("Final Temperature Anomaly")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("$T$ (K)")
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
        
        plt.tight_layout()
        
        fname = "dtemp.png"
        plt.savefig(fname)
        print("{} created.".format(fname))
        plt.close()

        ### MSE
        print("\nPlotting MSE")
        
        self._calculate_efe()
        print("EFE = {:.5f}".format(np.rad2deg(self.EFE)))
        
        f, ax = plt.subplots(1)
        ax.plot(self.sin_lats, self.E_f / 1000, "c")
        ax.plot([np.sin(self.EFE), np.sin(self.EFE)], [0, np.max(self.E_f)/1000], "r", label="EFE $\\approx {:.2f}$°".format(np.rad2deg(self.EFE)))
        ax.set_title("Final Energy")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("MSE (kJ kg$^{-1}$)")
        ax.set_ylim([np.min(self.E_f)/1000 - 1, np.max(self.E_f)/1000 + 1])
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
        ax.legend(loc="upper right")
        
        plt.tight_layout()
        
        fname = "mse.png"
        plt.savefig(fname)
        print("{} created.".format(fname))
        plt.close()
        
        ### dMSE
        print("\nPlotting dMSE")
        
        E_ctrl = self._T_to_E(T_ctrl)
        dE = (self.E_f - E_ctrl) / 1000

        f, ax = plt.subplots(1)
        ax.plot(self.sin_lats, dE, "c")
        ax.plot([np.sin(self.EFE), np.sin(self.EFE)], [np.min(dE) - 100, np.max(dE) + 100], "r", label="EFE $\\approx {:.2f}$°".format(np.rad2deg(self.EFE)))
        ax.set_title("Final Energy Anomaly")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("MSE (kJ kg$^{-1}$)")
        ax.set_ylim([np.min(dE) - 1, np.max(dE) + 1])
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
        ax.legend(loc="upper right")
        
        plt.tight_layout()
        
        fname = "dmse.png"
        plt.savefig(fname)
        print("{} created.".format(fname))
        plt.close()

        ### RADIATION
        print("\nPlotting Radiation")

        S_i = self.S * (1 - self.init_alb)
        L_i = self.L(self.init_temp)
        print("Integral of (S - L): {:.5f} PW".format(10**-15 * self._integrate_lat(self.S_f*(1 - self.alb_f) - self.L_f)))
        
        f, ax = plt.subplots(1)
        l1, = ax.plot(self.sin_lats, self.S_f*(1 - self.alb_f), "r", label="Final $S(1-\\alpha)$")
        l2, = ax.plot(self.sin_lats, S_i, "r--", label="Initial $S(1-\\alpha)$")
        l3, = ax.plot(self.sin_lats, self.L_f, "b", label="Final OLR")
        l4, = ax.plot(self.sin_lats, L_i, "b--", label="Initial OLR")
        l5, = ax.plot(self.sin_lats, self.S_f*(1 - self.alb_f) - self.L_f, "g", label="Final Net")
        l6, = ax.plot(self.sin_lats, S_i - L_i, "g--", label="Initial Net")
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
        ax.set_ylim([-200, 400])
        ax.set_yticks(np.arange(-200, 401, 50))
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Energy Flux (W m$^{-2})$")
        
        handles = (l1, l2, l3, l4, l5, l6)
        labels = ("Final $S(1-\\alpha)$", "Initial $S(1-\\alpha)$", "Final $L$", "Initial $L$", "Final $NEI$", "Initial $NEI$")
        f.legend(handles, labels, loc="upper center", ncol=3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)
        
        fname = "radiation.png"
        plt.savefig(fname)
        print("{} created.".format(fname))
        plt.close()
        
        if self.plot_transports:
            ### Differences and Transports
            print("\nPlotting Differences and Transports")
            colors = ["c", "g", "r", "m", "b", "y", "k", "k"]
            linestyles = [(0, (10, 1)), (0, (1, 1)), (0, (5, 1, 1, 1)), (0, (5, 1, 1, 1, 1, 1)), (0, (5, 1, 1, 1, 1, 1, 1, 1)), (0, (2, 2)), "--", "-"]

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.057, 7.057/1.62/2))

            l0, = ax1.plot(self.sin_lats, self.dS*(1 - self.alb), color=colors[0], linestyle=linestyles[0])
            l1, = ax1.plot(self.sin_lats, -(self.S + self.dS)*self.dalb, color=colors[1], linestyle=linestyles[1])
            l2, = ax1.plot(self.sin_lats, -self.dL_pl, color=colors[2], linestyle=linestyles[2])
            l3, = ax1.plot(self.sin_lats, -self.dL_wv, color=colors[3], linestyle=linestyles[3])
            # l4, = ax1.plot(self.sin_lats, -self.dL_rh, color=colors[4], linestyle=linestyles[4])
            l5, = ax1.plot(self.sin_lats, -self.dL_lr, color=colors[5], linestyle=linestyles[5])
            # l6, = ax1.plot(self.sin_lats, self.dS*(1 - self.alb) - (self.S + self.dS)*self.dalb - (self.dL_pl + self.dL_wv + self.dL_rh + self.dL_lr), color=colors[6], linestyle=linestyles[6])
            l6, = ax1.plot(self.sin_lats, self.dS*(1 - self.alb) - (self.S + self.dS)*self.dalb - (self.dL_pl + self.dL_wv + self.dL_lr), color=colors[6], linestyle=linestyles[6])
            l7, = ax1.plot(self.sin_lats, self.dS*(1 - self.alb) - (self.S + self.dS)*self.dalb - self.dL, color=colors[7], linestyle=linestyles[7])
            l8, = ax1.plot(np.sin(self.EFE), 0,  "or")

            ax1.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            ax1.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
            ax1.set_xlabel("Latitude")
            ax1.set_ylabel("Energy Perturbation (W m$^{-2}$)")

            ax2.plot(self.sin_lats, 10**-15  * self.dtrans_dS, c=colors[0], ls=linestyles[0])
            ax2.plot(self.sin_lats, -10**-15 * self.dtrans_dalb, c=colors[1], ls=linestyles[1])
            ax2.plot(self.sin_lats, -10**-15 * self.dtrans_pl, c=colors[2], ls=linestyles[2])
            ax2.plot(self.sin_lats, -10**-15 * self.dtrans_wv, c=colors[3], ls=linestyles[3])
            # ax2.plot(self.sin_lats, -10**-15 * self.dtrans_rh, c=colors[4], ls=linestyles[4])
            ax2.plot(self.sin_lats, -10**-15 * self.dtrans_lr, c=colors[5], ls=linestyles[5])
            # ax2.plot(self.sin_lats, 10**-15  * (self.dtrans_dS - self.dtrans_dalb - (self.dtrans_pl + self.dtrans_wv + self.dtrans_rh + self.dtrans_lr)), c=colors[6], ls=linestyles[6])
            ax2.plot(self.sin_lats, 10**-15  * (self.dtrans_dS - self.dtrans_dalb - (self.dtrans_pl + self.dtrans_wv + self.dtrans_lr)), c=colors[6], ls=linestyles[6])
            ax2.plot(self.sin_lats, 10**-15  * self.dtrans_total, c=colors[7], ls=linestyles[7])
            ax2.plot(np.sin(self.EFE), 0,  "or")
            
            ax2.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            ax2.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
            ax2.set_xlabel("Latitude")
            ax2.set_ylabel("Energy Transport (PW)")

            ax1.annotate("(a)", (0.01, 0.92), xycoords="axes fraction")
            ax2.annotate("(b)", (0.01, 0.92), xycoords="axes fraction")

            # handles = (l0, l1, l2, l3, l4, l5, l6, l7, l8)
            handles = (l0, l1, l2, l3, l5, l6, l7, l8)
            # labels = ("$S'(1 - \\alpha)$", "$-(S + S')\\alpha'$", "$-L_{PL}'$", "$-L_{WV}'$", "$-L_{RH}'$", "$-L_{LR}'$", "Sum", "$NEI'$", "EFE")
            labels = ("$S'(1 - \\alpha)$", "$-(S + S')\\alpha'$", "$-L_{PL}'$", "$-L_{WV}'$", "$-L_{LR}'$", "Sum", "$NEI'$", "EFE")
            f.legend(handles, labels, loc="upper center", ncol=9)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.89)
            
            fname = "differences_transports.png"
            plt.savefig(fname)
            print("{} created.".format(fname))
            plt.close()

            # np.savez("differences_transports.npz", 
            #     EFE=self.EFE, 
            #     sin_lats=self.sin_lats, 
            #     dtrans_total=self.dtrans_total, 
            #     dtrans_pl=self.dtrans_pl, 
            #     dtrans_wv=self.dtrans_wv, 
            #     # dtrans_rh=self.dtrans_rh, 
            #     dtrans_lr=self.dtrans_lr, 
            #     dtrans_dS=self.dtrans_dS, 
            #     dtrans_dalb=self.dtrans_dalb, 
            #     dL_pl=self.dL_pl, 
            #     dL_wv=self.dL_wv, 
            #     # dL_rh=self.dL_rh, 
            #     dL_lr=self.dL_lr, 
            #     dL=self.dL,
            #     dS=self.dS,
            #     dalb=self.alb_f - self.alb_ctrl,
            #     alb=self.alb_ctrl,
            #     S=self.S)
