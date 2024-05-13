
MG_parallel.py contains main function for solving equations for metric functions in a case of R2 gravity.

The comments in this file allows to understand general scheme of solution and main problem. 
The algorithm uses fenics which should be installed in Python.
If DOCKER is used one need to make the following command in shell:

docker run -ti -v d:\:/home/fenics/shared quay.io/fenicsproject/stable

(for this example corresponding repository on disk D:)

GR_parallel.py contains the same function for solving equations for metric functions in a case general relativity. In this file there are comments on russian but the main procedure can 
be understanded from comments to MG_parallel.py

Launcher_0.py activates calculations on multiple cores. Comments in this file are important also.

Folder /data contains some equations of state (enthalpy-density-pressure-number density) in corresponding units (these table are obtained from tables in SI or SGS units).

Folder /init_files contains obtained solutions for various densities and frequencies. There are solutions for GR (folders corresponds to various EoS) and R2 gravity with alpha=0.25 
(in units of r_g^2 where r_g is gravitational radius of Sun) or alpha=1 (in units of r_g^2 where r_g is half of gravitational radius of Sun). There are solutions for alpha=2.5 and 1.25 
(5 and 10) but I didn't upload these files.  
These files can be used as initial solutions for some another frequencies, densities which are close to such for given file.





