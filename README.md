# Moist Energy Balance Model (`mebm`)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## About
This repository contains a simple Python package for a moist energy balance model (`mebm`).
The model is used in Peterson and Boos (2020) to analyze feedbacks and eddy diffusivity in the context of tropical rainfall shifts.
Features include
- Easily customizable initial conditions, albedo, insolation, and outgoing longwave radiation schemes
- Efficient and scalable multigrid numerical scheme
- Ability to perturb insolation (as in Clark et al. 2018) and analyze energy flux equator (EFE) shifts.

## Getting Started
See the documentation [here](docs/guide.md) to get started. 

## Contact
For questions and suggestions, either submit an issue here or send an email to hgpeterson "at" caltech "dot" edu.

## References
Clark, S.K., Y. Ming, I.M. Held, P.J. Phillipps, 2018: The Role of the Water Vapor Feedback in the ITCZ Response to Hemispherically Asymmetric Forcings. *Journal of Climate*. **31**, 3659–3678.

Peterson, H. and W. Boos, 2020: Feedbacks and Eddy Diffusivity in an Energy Balance Model of Tropical Rainfall Shifts. *npj Climate and Atmospheric Science*. **3**, 11.
