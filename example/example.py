import mebm 

# Instantiate `model` object
model = mebm.MoistEnergyBalanceModel(N_pts=2**9+1, max_iters=1e3, tol=1e-4, 
            diffusivity="constant", control_file="default")

# Set its initial temperature, insolation type, albedo, and outgoing longwave
model.set_init_temp(init_temp_type="legendre", low=250, high=300)
model.set_insol(insol_type="perturbation", perturb_center=15, 
            perturb_spread=4.94, perturb_intensity=10)
model.set_albedo(al_feedback=True, alb_ice=0.6, alb_water=0.2)
model.set_olr(olr_type="full_radiation", A=None, B=None, emissivity=None)

# Solve model equation
model.solve(numerical_method="multigrid", frames=100)

# Save model data to `simulation.npz`
model.save_data(control=False)

# Log energy flux equator information to `itcz.log`
model.log_efe(fname_efe="itcz.log")

# Calculate climate feedback transports 
model.calculate_feedbacks()

# Create some plots based on model solution
model.save_plots()
