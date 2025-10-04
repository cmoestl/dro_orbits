# dro_orbits
## Calculation of distant retrograde orbits for space weather forecast analyses


---


Dependencies:
- needs ffmpeg for making movies
- environment "dro" is defined in /env/env_dro.yml
- need to download de442.bsp file from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/  (114 MB) and place file in folder kernels/

---


### Scripts


dro.ipynb: generates numerical solutions for dro orbits and makes plots and animations


---


### Papers:

Frnka 2010: https://jan.ucc.nau.edu/~ns46/student/2010/Frnka_2010.pdf

Perozzi 2017: 

Cicalo 2025: 


---

### Installation:


Create a conda environment using the "envs/env_dro.yml", and activate the environment:

    conda env create -f env_dro.yml

    conda activate dro












