# dro_orbits

## Calculation of distant retrograde orbits for space weather forecast analyses


Author: C. Möstl, Austrian Space Weather Office, GeoSphere Austria

Last update: July 2026

Sample distant retrograde orbits in HEE are available in folder "orbit_files", this is work in progress.

If you want to use this for anything, please contact me.


---

### Scripts

**find_dro.ipynb**: finds optimized numerical solutions for dro orbits to provide initial conditions 

**dro.ipynb**: main notebook, generates numerical solutions for dro orbits with given initial conditions, makes plots and animations for DRO analyses

**example_for_usage.ipynb**: reads in an orbit file from folder orbit_files and plots a DRO in cartesian and polar coordinates

---


### Dependencies
- environment *dro* is defined in /env/env_dro.yml, includes only standard packages
- needs ffmpeg for making movies https://www.ffmpeg.org
- file de442.bsp (114 MB) is automatically downloaded when running dro.ipynb for the first time from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/ to folder *kernels/*
- spiceypy is used for generating the positions of the planets

---

### Papers

St. Cyr+ 2000: https://www.sciencedirect.com/science/article/abs/pii/S1364682600000699?via%3Dihub      
Frnka 2010: https://jan.ucc.nau.edu/~ns46/student/2010/Frnka_2010.pdf     
Perozzi+ 2017:  https://link.springer.com/article/10.1140/epjp/i2017-11644-0    
Lugaz+ 2024: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024SW004189    
Cicalo+ 2025:  https://arxiv.org/abs/2508.02138  
Prete+ 2026: https://www.swsc-journal.org/articles/swsc/full_html/2026/01/swsc250088/swsc250088.html


---

### Installation


Create a conda environment using the "envs/env_dro.yml", and activate the environment:

    conda env create -f env_dro.yml

    conda activate dro


---

### Demo plot

![DRO sample](results/dro_all_icme_polar_zoom.png)









