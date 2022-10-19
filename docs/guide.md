# Moist Energy Balance Model (`mebm`) Guide

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Installation
First, get the repository on your computer with
```
git clone https://github.com/hgpeterson/mebm
```
and  
```
cd mebm
``` 
so that you're in the package.
Next, install the Python packages you'll need using 
```
pip install -r requirements.txt
```
which will install `matplotlib`, `numpy`, `scipy`, `climt`, and `sympl` if you don't already have them.
`climt` (and its dependency, `sympl`) are only used for the more complicated radiative transfer schemes.
Now, to install the `mebm` package, type
```
pip install -e .
```
where the `-e` flag allows you to edit the `mebm` source if you wish. 
Now you can use the `mebm` package in any Python script by putting 
```
...
import mebm
...
```
at the top.

## Quick Guide
For an informative example, see the file in this repository under `example/example.py`, the contents of which are 

```python 
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
```
Let's go through each line. First, we instantiate a `model` object with
```python
model = mebm.MoistEnergyBalanceModel(N_pts=2**9+1, max_iters=1e3, tol=1e-4, 
            diffusivity="constant", control_file="default")
```
This creates a model with 513 evenly spaced gridpoints (in x = sin(phi) coordinates).
The `solve` method will use at most 1000 multigrid V-cycle iterations or terminate when changes are less than 0.0001 K.
This particular model run will use a constant eddy diffusivity (2.6e-4 kg m2 s-1 from Hwange and Frierson 2010) and comparer the results to the default control file, though you can supply your own after using `control=True` in the method `save_data` which will output a `ctrl.npz` file.

TBC...



## References
Clark, S.K., Y. Ming, I.M. Held, P.J. Phillipps, 2018: The Role of the Water Vapor Feedback in the ITCZ Response to Hemispherically Asymmetric Forcings. *Journal of Climate*. **31**, 3659–3678.

Hwang, Y.T. and D.M.W. Frierson, 2010: Increasing Atmospheric Poleward Energy Transport with Global Warming. *Geo. Res. Lett.*. **37**, 1–5.

Peterson, H. and W. Boos, 2020: Feedbacks and Eddy Diffusivity in an Energy Balance Model of Tropical Rainfall Shifts. *npj Climate and Atmospheric Science*. **3**, 11.
