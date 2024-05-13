from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import sympify
import math
from shutil import copyfile
import os
import json

import sys
sys.path.append('functions')
from NumSrcF import numer_src_function, numer_src_function0
from Graph_mod import graphic_module, graphic_in
from enthalpy import energy_enthalpy

from MG_parallel import SOLUTION_OF_TOV #for GR calculations one need to simply MG_parallel change on GR_parallel

#the launch of this file activates function of solution on multiple cores for various densities in the center. For this example - 5 cores.
EOS = 'EOS_GM1' #specify corresponds EoS and then to put in init_files corresponding files with densities, frequencies specified below. 

energy0 = 250
energy1 = 275
energy2 = 300
energy3 = 325
energy4 = 350

frequency0 = 200
frequency1 = 200
frequency2 = 200
frequency3 = 200
frequency4 = 200

#specify the shifts from initial solutions. Zeros for checking of solutions
delta_f0 = 0
delta_f1 = 0
delta_f2 = 0
delta_f3 = 0
delta_f4 = 0

delta_J20, delta_J0 = 0.0, 0.0
delta_J21, delta_J1 = 0.0, 0.0
delta_J22, delta_J2 = 0.0, 0.0
delta_J23, delta_J3 = 0.0, 0.0
delta_J24, delta_J4 = 0.0, 0.0

delta_m0 = -0.00
delta_m1 = -0.00
delta_m2 = -0.00
delta_m3 = -0.00
delta_m4 = -0.00

curv0 = 0.0000
curv1 = 0.0000
curv2 = 0.0000
curv3 = 0.0000
curv4 = 0.0000

Num = 7 #number of iterations

import time
import multiprocessing

import os
from multiprocessing import Process

descriptor = [[EOS, energy0, frequency0, delta_f0, delta_m0, delta_J0, delta_J20, curv0, Num, 0], 
              [EOS, energy1, frequency1, delta_f1, delta_m1, delta_J1, delta_J21, curv1, Num, 0], 
              [EOS, energy2, frequency2, delta_f2, delta_m2, delta_J2, delta_J22, curv2, Num, 0], 
              [EOS, energy3, frequency3, delta_f3, delta_m3, delta_J3, delta_J23, curv3, Num, 0],
              [EOS, energy4, frequency4, delta_f4, delta_m4, delta_J4, delta_J24, curv4, Num, 0]]

#SOLUTION_OF_TOV(descriptor[0])


def main_function(arg):
    SOLUTION_OF_TOV(arg)


print(round(time.perf_counter(),1))
#main_function(parser.EoS_descriptor[0])

if __name__ =='__main__':

    pool_obj = multiprocessing.Pool(processes=5)

    args = [([descriptor[i]]) for i in range(5)]

    pool_obj.starmap(main_function,args)

print(round(time.perf_counter(),1))