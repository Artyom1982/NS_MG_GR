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

#initial constants in CGS units
solar_mass=1.989E33
grav_constant=6.67408E-8
speed_of_light=299792458E2

#calculaton of natural units for coordinate, density and pressure
r_g=grav_constant*solar_mass/(speed_of_light**2)
rho_0=solar_mass/(r_g**3)
p_0=rho_0*speed_of_light**2
core_density = 4*10**(14)/rho_0

def SOLUTION_OF_TOV(args):
    # Начальные параметры для решения задачи
    R0=0.000 #coordinate of center of star

    R2=400 #coordinate of right boundary of outer region

    #tolerance
    tol=1E-14

    #energy density in center of star MeV/fm^3
    e1 = args[1]
    Omega0 = args[2] #frequency of rotation in Hz

    Name0='init_files/' + 'e=' + str(e1)+ ',' + 'omega=' + str(Omega0) + '_sum.txt' #init parameters from corresponding file
    
    "Program takes from this file the following quantities: 
    """""
    mass - expected value of ADM
    mass2 - value of gravitational mass on boundary of inner domain.
    m_r - expected value of moment of rotation
    m_r2 - value of rotation moment on bounday of inner domain
    omega - value of frequency (Hz)
    R1 - radial coordinate of inner domain (in units of r_g=GM/c**2=1.47 km, i.e. one half of the gravitational radius of the Sun). 
    Curv_R - value of curvature om boundary og inner domain (in units of 1/r_g**2)
    gamma - value of parameter gamma for f(R)=R+gamma R^2 gravity (in units of r_g**2). For many papers we used for r_g value 2GM/c**2 and therefore we need to multiply by 4 value 
    given in 2GM/c**2.
    resolution - parameter of spatial resolution in one dimension.
    Another quantities will be calculated and written.
    """""""""

    Name1='init_files/' + 'e=' + str(e1)+ ',' + 'omega=' + str(Omega0) + '.txt' #file with starting solution
    """""""""""""""
    This file has the following format:
    1st column: radial coordinate (from 0 to R1) 
    2st column: angular coordinate (from 0 to Pi) 
    3st column: enthalpy in corresponding point
    4st column: function A
    5st column: velocity 
    6st column: (B/N)**2
    """""

    Name_EOS = 'data/'+args[0]+'.txt' #table of EoS (in dimensionless units, format - log-enthalpy - energy density - pressure - concentration (1/fm^3))

    energy0 = e1*1.78E12/rho_0 #converting of energy density in dimensionless units
    h0 = energy_enthalpy(Name_EOS,energy0)[0] #enthalpy corresponding of center energy density
    h0_min = energy_enthalpy(Name_EOS,energy0)[1] #minimal value of enthalpy

    d_Omega = args[3] #increasing of Omega in comparison with value in init_file. If we want to recalculate for the same frequency one need to use d_Omega=0
    delta_m = args[4] #possible increasing of ADM
    delta_J = args[5] #possible increasing of angular momentum
    delta_J2 = args[6] #possible increasing of angular momentum at R1

    #getting of the data from _sum.txt file and adding of corresponding increasing
    with open(Name0) as f:
        data = json.load(f)
    mass=float(sympify(data["mass"]))
    mass=mass+delta_m
    mass2=float(sympify(data["mass2"]))
    moment_of_rotation=float(sympify(data["m_r"])) + delta_J
    moment_of_rotation2=float(sympify(data["m_r2"])) + delta_J2

    Omega0=int(sympify(data["omega"]))
    Omega0=Omega0+d_Omega
    R1=float(sympify(data["R1"]))
    Curv_R=float(sympify(data["curv"])) + args[7]
    gamma=4*float(sympify(data["gamma"]))
    resolution = int(sympify(data["resolution"]))

    #calculation of conformal coordinates of inner domain
    X1=1-R1/R1
    X2=1-R1/R2

    Omega1=Omega0

    #converting of frequency in dimensionless units
    Omega=2*np.pi*Omega0*r_g/speed_of_light

    #Estimated value of radius of star (for first iteration) in units of r_g=1.47km and quasi-isotropical coordinates. Physical radius of star is this quantity multiplied by 
    #factor A and r_g. 
    radius=9
    
    key=1 #if key=1 we use starting solution from init file. For key=0 we construct solution from beginning.
  
    Number_of_iterations=args[8] #number of iteration for converging solution. For low densities in center solution converges sufficienly fast. If we use for starting solution solution
    #with lower density and increasing of density is sufficiently small the 10-20 iterations are sufficiently for converging. 

    #имена файлов, в которые записывается решение (могу затем использоваться как
    #стартовые при переименовании в H.txt и Factors.txt соответственно)

    dir_name='results' + '/' + 'gamma='+str(gamma)+'/' #name of the files in which the obtained solution will be written. This solution can be used as starting for another central density
    #and so on

    if os.path.exists(dir_name):
        dir_name=dir_name
    else:
        os.mkdir(dir_name, mode=0o777, dir_fd=None)

    filename_solution='init_files/' + 'e=' + str(e1)+ ','\
    +'omega='+str(Omega0)+'.txt'

    domain=Rectangle(Point(R0,0), Point(R1, np.pi/2)) #inner domain (we use spherical coordinates, 1/2 of the sphere, for FENICS this is rectangle)
    
    mesh=generate_mesh(domain, resolution) #generation of mesh

    #boundary conditions on inner boundary (R1)
    A_R=(1+mass2/(2*R1))**2
    N_R=(1-mass/(2*R1))/(1+mass/(2*R1))

    #boundary conditions on outer boundary (R2)
    A_inf=(1+mass/(2*R2))**2
    N_inf=(1-mass/(2*R2))/(1+mass/(2*R2))
    Curv_inf=0.000

    #boundary conditions on inner boundary (R1) for nu, mu, omega
    nu_R=ln(N_R)
    mu_R=ln(A_R*N_R)
    omega_R=2*moment_of_rotation2/(R1**3)

    #boundary conditions on outer boundary (R2)
    omega_inf=2*moment_of_rotation/((R2)**3)
    nu_inf=ln(N_inf)
    mu_inf=ln(N_inf*A_inf)

    u_R=[Constant(nu_R), Constant(mu_R), Constant(omega_R), Constant(mu_R), Constant(Curv_R)] #boundary conditions on right boundary of inner domain
    u_B=[Constant(nu_R), Constant(mu_R), Constant(omega_R), Constant(mu_R), Constant(Curv_R)] #boundary conditions on left boundary of outer domain
    u_inf=[Constant(nu_inf), Constant(mu_inf), Constant(omega_inf), Constant(mu_inf), Constant(Curv_inf)] #boundary conditions on right boundary of outer domain

    SRC = FunctionSpace(mesh, 'CG', 1) # functional space for sources functions
    n = SRC.dim() #number of nodes
    d = mesh.geometry().dim() #number of space dimensions

    #array of nodes coords
    F_dof_coordinates = SRC.tabulate_dof_coordinates().reshape(n,d)
    F_dof_coordinates.resize((n, d))

    #array of x and y coords of nodes
    F_dof_x = F_dof_coordinates[:, 0]
    F_dof_y = F_dof_coordinates[:, 1]

    x=sym.Symbol('x')
    y=sym.Symbol('y')

    h=h0*(1-x*x/(radius*radius)) #probing profile for enthalty (if we found solution from beginnning without starting solution). In principle one can use monotonic function


    # This cycle creates file with init solution in which the coords of nodes, enthalpy, A (for simplicity 1.5 in all points), 
    # velocity (zero value for first iteration), (B/N)**2 (1) will be written. 
    # If key is not 0 this step will be passed.
    if key==0:
        G = open(Name1, 'w')
        for j in range(n):
            X=F_dof_x[j]
            Y=F_dof_y[j]
            enthalpy0=h.subs(x,X)
            a=[]
            a.append(X)
            a.append(Y)
            a.append(enthalpy0)
            G.write(str(a[0]))
            G.write(' ')
            G.write(str(a[1]))
            G.write(' ')
            G.write(str(a[2]))
            G.write(' ')
            G.write('1.5')
            G.write(' ')
            G.write('0')
            G.write(' ')
            G.write('1.0')
            G.write('\n')
        G.close()

    central_density=numer_src_function0(Name_EOS,h0)[0]

    #functional space for unknown functions (u) and probing functions (v)
    P1 = FiniteElement('CG', triangle, 3)
    element = MixedElement([P1, P1, P1, P1, P1]) #number of function
    V = FunctionSpace(mesh, element)

    #boundary of inner domain
    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], R1, tol)

    #Dirichlet conditions on right boundary
    bcs = DirichletBC(V, u_R, boundary_R)

    #Lame coefficients and Jacobi determinants for equations in inner domain
    H_theta=Expression('x[0]', degree=2)
    H1_theta=Expression('cos(x[1])', degree=2)
    J1=Expression('pow(x[0],2)*sin(x[1])',degree=2)
    J2=Expression('pow(x[0]*sin(x[1]),2)*x[0]',degree=2)
    J3=Expression('pow(x[0]*sin(x[1]),3)*x[0]',degree=2)
    J4=Expression('x[0]',degree=2)

    #The same for outer domain with acount of conformal transformation r=R1/(1-x) (x varis from 0 to value close to 1 for distant point R2)
    H_r_out=R1*Expression('pow(1/(1-x[0]),2)', degree=2)
    H_theta_out=R1*Expression('1/(1-x[0])',degree=2)
    J1_out=H_r_out*H_theta_out*H_theta_out
    J2_out=H_r_out*H_theta_out*H_theta_out*H_theta_out
    J3_out=H_r_out*H_theta_out*H_theta_out*H_theta_out*H_theta_out
    J4_out=H_r_out*H_theta_out

    """"
    The next step is iteration procedure for construction of converging solution. Using initial solution (file or robust solution)
    we recalculate enthalpy, velocity of rotation in nodes, A, B/N.
    We use conservation H(r)+nu(r)+ln(Gamma)=const. Gamma=(1-U**2)**(-1/2) is Lorentz factor
    Therefore enthalpy is H=h0+nu(0)-nu(r)-ln(Gamma).  
    We don't comment solution of the equations with use of FENICS and corresponding notation (see refs. for FENICS)

    """"

    Name1='init_files/' + 'e=' + str(e1)+ ',' + 'omega=' + str(Omega1) + '.txt'

    for k in range(Number_of_iterations):
        print()
        print('***********************************************************')
        print('ITERATION =',k+1)

        v_1, v_2, v_3, v_4, v_5 = TestFunctions(V)
        u = Function(V)
        u_1, u_2, u_3, u_4, u_5 = split(u)

        Z1=1+2*gamma*u_5 #derivative of df(R)/dR
        Z2=2*gamma #second derivative of f(R) on R
        Z0=gamma*u_5*u_5 #f'R-f

        result = numer_src_function(Name_EOS, Name1, n)

        f1 = Function(SRC)
        f2 = Function(SRC)
        f3 = Function(SRC)
        f4 = Function(SRC)
        f5 = Function(SRC)
        f0 = Function(SRC)
        f6 = Function(SRC)

        f1.vector()[:] = result[0]
        f2.vector()[:] = result[1]
        f3.vector()[:] = result[2]
        f4.vector()[:] = result[3]
        f5.vector()[:] = result[4]
        f0.vector()[:] = result[5]
        f6.vector()[:] = result[6]

        F0=-J1*Z1*u_1.dx(0)*v_1.dx(0)*dx\
        -J1*Z1*((1/H_theta)**2)*u_1.dx(1)*v_1.dx(1)*dx \
        -J2*Z1*u_2.dx(0)*v_2.dx(0)*dx\
        -J2*Z1*((1/H_theta)**2)*u_2.dx(1)*v_2.dx(1)*dx \
        -J3*Z1*u_3.dx(0)*v_3.dx(0)*dx\
        -J3*Z1*((1/H_theta)**2)*u_3.dx(1)*v_3.dx(1)*dx \
        -J4*Z1*u_4.dx(0)*v_4.dx(0)*dx\
        -J4*Z1*u_4.dx(1)*((1/H_theta)**2)*v_4.dx(1)*dx

        F0_MG=-0.5*Z2*J1*u_5.dx(0)*v_1.dx(0)*dx\
        -0.5*Z2*J1*((1/H_theta)**2)*u_5.dx(1)*v_1.dx(1)*dx\
        -Z2*J2*u_5.dx(0)*v_2.dx(0)*dx\
        -Z2*J2*((1/H_theta)**2)*u_5.dx(1)*v_2.dx(1)*dx\
        -Z2*J3*u_5.dx(0)*u_3.dx(0)*v_3*dx\
        -Z2*J3*((1/H_theta)**2)*u_5.dx(1)*u_3.dx(1)*v_3*dx\
        -4*Z2*J3*u_5.dx(0)*(u_2.dx(0)-u_1.dx(0))*u_3*v_3*dx\
        -4*Z2*J3*((1/H_theta)**2)*u_5.dx(1)*(u_2.dx(1)-u_1.dx(1))*u_3*v_3*dx\
        -4*Z2*J3*(1/H_theta)*u_5.dx(0)*u_3*v_3*dx\
        -4*Z2*J2*(1/H_theta)*H1_theta*u_5.dx(1)*u_3*v_3*dx\
        -Z2*J4*u_5.dx(0)*v_4.dx(0)*dx\
        -Z2*J4*((1/H_theta)**2)*u_5.dx(1)*v_4.dx(1)*dx\

        F0_curv=-J1*Z2*u_5.dx(0)*v_5.dx(0)*dx\
        -J1*Z2*((1/H_theta)**2)*u_5.dx(1)*v_5.dx(1)*dx\
        #sources function, i.e. terms with rho and pressure
        F_src=-4*J1*np.pi*f1*v_1*dx\
        -16*J2*np.pi*f2*v_2*dx\
        +16*J3*np.pi*f3*(Omega-u_3)*v_3*dx\
        -8*J4*np.pi*f4*v_4*dx\
        +8*np.pi*(1/3)*f5*J1*v_5*dx

        F_src_MG=0.5*exp(2*u_4-2*u_1)*(Z1*u_5-u_5-Z0)*v_1*J1*dx\
        +exp(2*u_4-2*u_1)*(Z1*u_5-u_5-Z0)*v_2*J2*dx\
        +0.5*exp(2*u_4-2*u_1)*(Z1*u_5-u_5-Z0)*v_4*J4*dx\
        -(1/3)*exp(2*u_4-2*u_1)*J1*(2*Z0+2*u_5-Z1*u_5)*v_5*dx
        #nonlinear terms without sotation
        F_nl=J1*Z1*v_1*u_1.dx(0)*u_2.dx(0)*dx\
        +J1*Z1*((1/H_theta)**2)*v_1*u_1.dx(1)*u_2.dx(1)*dx\
        +J2*Z1*v_2*u_2.dx(0)*u_2.dx(0)*dx\
        +J2*Z1*((1/H_theta)**2)*v_2*u_2.dx(1)*u_2.dx(1)*dx\
        -4*J3*Z1*u_3.dx(0)*u_1.dx(0)*v_3*dx\
        -4*J3*Z1*((1/H_theta)**2)*u_3.dx(1)*u_1.dx(1)*v_3*dx\
        +3*J3*Z1*u_3.dx(0)*u_2.dx(0)*v_3*dx\
        +3*J3*Z1*((1/H_theta)**2)*u_3.dx(1)*u_2.dx(1)*v_3*dx\
        +J4*Z1*u_1.dx(0)*u_1.dx(0)*v_4*dx\
        +J4*Z1*((1/H_theta)**2)*u_1.dx(1)*u_1.dx(1)*v_4*dx\

        F_nl_MG=0.5*Z2*J1*u_2.dx(0)*u_5.dx(0)*v_1*dx\
        +0.5*Z2*J1*((1/H_theta)**2)*v_1*u_2.dx(1)*u_5.dx(1)*dx\
        +Z2*J2*u_2.dx(0)*u_5.dx(0)*v_2*dx\
        +Z2*J2*((1/H_theta)**2)*u_2.dx(1)*u_5.dx(1)*v_2*dx\
        -Z2*J3*u_3.dx(0)*u_5.dx(0)*v_3*dx\
        -Z2*J3*((1/H_theta)**2)*u_3.dx(1)*u_5.dx(1)*v_3*dx\
        -Z2*J4*u_5.dx(0)*(u_4.dx(0)-u_1.dx(0))*v_4*dx\
        -Z2*J4*((1/H_theta)**2)*u_5.dx(1)*(u_4.dx(1)-u_1.dx(1))*v_4*dx\
        +Z2*J1*u_2.dx(0)*u_5.dx(0)*v_5*dx\
        +Z2*J1*((1/H_theta)**2)*u_5.dx(1)*u_2.dx(1)*v_5*dx\

        #nonlinear terms due to the rotation
        F_rot=-0.5*J3*Z1*f0*u_3.dx(0)*u_3.dx(0)*v_1*dx\
        -0.5*J3*Z1*f0*((1/H_theta)**2)*u_3.dx(1)*u_3.dx(1)*v_1*dx\
        -0.75*J2*Z1*f0*u_3.dx(0)*u_3.dx(0)*v_4*dx\
        -0.75*J2*Z1*f0*u_3.dx(1)*((1/H_theta)**2)*u_3.dx(1)*v_4*dx\

        F=F0+F0_curv+F0_MG+F_src+F_src_MG+F_nl+F_nl_MG+F_rot

        solve(F == 0, u, bcs) #solution

        p0=Point(R0,0) #center of the star
        nu0=u(p0.x(), p0.y())[0] #function nu in the center

        print(nu0) #Print the value of nu in the center of star. Only for control. If solution converges these values become very close from one iteration to another 

        Name1='init_files/' + 'e=' + str(e1)+ ',' + 'omega=' + str(Omega0) + '.txt'
        #write of recalculated quantitites in file of solution
        G = open(Name1, 'w')
        for i in range(n):
            a=[]
            p=Point(F_dof_x[i], F_dof_y[i])
            nu=u(p.x(), p.y())[0] #nu
            scale=exp(u(p.x(), p.y())[3]-u(p.x(), p.y())[0]) #A
            scale2=exp(u(p.x(), p.y())[1]-2*u(p.x(), p.y())[0]) #B/N
            velocity=scale2*F_dof_x[i]*sin(F_dof_y[i])*(Omega-u(p.x(), p.y())[2]) #velocity
            H=h0+nu0-nu-0.5*ln(1-velocity**2) #enthalpy
            X=F_dof_x[i]
            Y=F_dof_y[i]
            a.append(H)
            a.append(scale)
            a.append(velocity)
            a.append(scale2*scale2)
            G.write(str(X))
            G.write(' ')
            G.write(str(Y))
            G.write(' ')
            G.write(str(a[0]))
            G.write(' ')
            G.write(str(a[1]))
            G.write(' ')
            G.write(str(a[2]))
            G.write(' ')
            G.write(str(a[3]))
            G.write('\n')
        G.close()

    #Calculation of radial dependence if functions for equator (theta=Pi/2) and pole (theta=0)
    theta1 = 0
    theta2 = np.pi/2
    key1 = 1

    if Omega==0:
        Omega = 1
        key1 = 0

    data_x = []
    data_N = []
    data_B = []
    data_Om = []
    data_A = []
    data_R = []
    data_h = []
    density = []
    pressure = []

    data_x_2 = []
    data_N_2 = []
    data_B_2 = []
    data_Om_2 = []
    data_A_2 = []
    data_R_2 = []
    data_h_2 = []
    density_2 = []
    pressure_2 = []

    data_vel = []
    data_J = []
    data_m = []

    result_solution_1=dir_name + 'e=' + str(e1)+ ','\ 
    + 'omega=' + str(Omega0) + '_res' + '.txt' #file containing the profiles of function for theta=Pi/2

    result_solution_2=dir_name + 'e=' + str(e1)+ ','\
    + 'omega=' + str(Omega0) + '_res-eq' + '.txt' #file containig the profiles of function for theta=0

    G = open(result_solution_1, 'w')
    G2 = open(result_solution_2, 'w')

    for i in range (2000):
        x1 = R0 + 0.0005 * i * (R1 - R0)
        p1 = Point(x1,theta1)
        p2 = Point(x1,theta2)
        F0 = exp(u(p1.x(), p1.y())[0]) #N=exp(nu)
        F1 = exp(u(p1.x(), p1.y())[1])/F0 #B
        F2 = u(p1.x(),p1.y())[2]/Omega #omega/Omega
        F3 = exp(u(p1.x(), p1.y())[3])/F0 #A
        F4 = nu0 + h0 - ln(F0) #enthalpy on polar axis
        F5 = u(p1.x(), p1.y())[4] #curvature

        density.append(numer_src_function0(Name_EOS,F4)[0])
        if density[i]>=core_density:
            x_core = x1
            R_core = F3*x1*r_g/(10**5)
        pressure.append(numer_src_function0(Name_EOS,F4)[1])

        m = 2*(pow(F1,0.5)-1)*x1
        data_m.append(m) #ADM mass
        data_x.append(x1*1*r_g/(10**5)) #coordinate radius in km
        data_N.append(F0) #N
        data_B.append(F1) #B
        data_Om.append(F2)
        data_A.append(F3)
        data_R.append(F5)
        data_h.append(F4)
        G.write(str(F3*x1*r_g/(10**5))) #physical radius
        G.write(' ')
        G.write(str(F0)) #N
        G.write(' ')
        G.write(str(F1)) #B
        G.write(' ')
        G.write(str(F2)) #omega/Omega
        G.write(' ')
        G.write(str(F3)) #A
        G.write(' ')
        G.write(str(F4)) #enthalpy
        G.write(' ')
        G.write(str(4*F5)) #curvature. Factor 4 convert curvature in units of 1/r_g**2 (r_g=1.47 km) to units in which r_g=2.95 km.
        G.write('\n')
        #the same for equatirial plane
        F0 = exp(u(p2.x(), p2.y())[0])
        F1 = exp(u(p2.x(), p2.y())[1])/F0
        F2 = u(p2.x(),p2.y())[2]/Omega
        F3 = exp(u(p2.x(), p2.y())[3])/F0
        vel = (F1/F0) * x1 * Omega * (1 - F2) * key1
        F4 = nu0 + h0 - ln(F0) - 0.5 * ln(1 - vel * vel)
        F5 = u(p2.x(),p2.y())[4]
        J_rot = (x1**3) * Omega * F2/2
        data_J.append(J_rot)
        data_x_2.append(x1*1*r_g/(10**5))
        data_N_2.append(F0)
        data_B_2.append(F1)
        data_Om_2.append(F2)
        data_A_2.append(F3)
        data_R_2.append(F5)
        data_h_2.append(F4)
        data_vel.append(vel)

        density_2.append(numer_src_function0(Name_EOS,F4)[0])
        pressure_2.append(numer_src_function0(Name_EOS,F4)[1])
        G2.write(str(F3*x1*r_g/(10**5)))
        G2.write(' ')
        G2.write(str(F0))
        G2.write(' ')
        G2.write(str(F1))
        G2.write(' ')
        G2.write(str(F2))
        G2.write(' ')
        G2.write(str(F3))
        G2.write(' ')
        G2.write(str(F4))
        G2.write(' ')
        G2.write(str(4*F5))
        G2.write('\n')

    G.close()
    G2.close()


    #figures for illustration (to appear in folder /graphics)
    graphic_in(data_x,data_x_2,data_N,data_N_2,data_B,data_B_2,
                        data_Om,data_Om_2,data_A,data_A_2,
                        density, density_2, pressure, pressure_2,data_R,data_R_2,e1,Omega0)

    #We found the surface of star in which enthalpy is close to minimal value in table of EoS
    for i in range(2000):
        if data_h_2[i]<-0.0088:
            eq_radius=data_x_2[i]
            circ_radius=eq_radius*data_A_2[i] #calculation of equatorial radius (physical)
            eq_velocity=data_vel[i] #calculation of rotation velocity on equator
            break

    for i in range(2000):
        if data_h[i]<-0.0088:
            polar_radius=data_x[i] #calculation of polar radius (physical)
            break

    #Solution for outer domain. All designations are the same as for inner domain. Postfix out is used only for unknown and probing functions
    domain_out = Rectangle(Point(X1,0), Point(X2, np.pi/2))
    mesh_out = generate_mesh(domain_out, 64)
    #
    P2 = FiniteElement('CG', triangle, 2)
    element = MixedElement([P2, P2, P2, P2, P2])
    #
    V_out = FunctionSpace(mesh_out,element)
    #
    u_out = Function(V_out)
    u_out_1, u_out_2, u_out_3, u_out_4, u_out_5 = split(u_out)
    v_out_1, v_out_2, v_out_3, v_out_4, v_out_5 = TestFunctions(V_out)

    Z1=1+2*gamma*u_out_5
    Z2=2*gamma
    Z0=gamma*u_out_5*u_out_5

    def boundary_left(x, on_boundary):
        return on_boundary and near(x[0], X1, tol)
    def boundary_inf(x, on_boundary):
        return on_boundary and near(x[0], X2, tol)

    bcs_out=[]
    bcs_left=DirichletBC(V_out, u_B, boundary_left)
    bcs_out.append(bcs_left)
    bcs_inf=DirichletBC(V_out, u_inf, boundary_inf)
    bcs_out.append(bcs_inf)

    F0_out = -J1_out*Z1*(1/H_r_out)**2*u_out_1.dx(0)*v_out_1.dx(0)*dx\
    -J1_out*Z1*(1/H_theta_out)**2*u_out_1.dx(1)*v_out_1.dx(1)*dx\
    -J2_out*Z1*(1/H_r_out)**2*u_out_2.dx(0)*v_out_2.dx(0)*dx\
    -J2_out*Z1*(1/H_theta_out)**2*u_out_2.dx(1)*v_out_2.dx(1)*dx\
    -J3_out*Z1*(1/H_r_out)**2*u_out_3.dx(0)*v_out_3.dx(0)*dx\
    -J3_out*Z1*(1/H_theta_out)**2*u_out_3.dx(1)*v_out_3.dx(1)*dx\
    -J4_out*Z1*(1/H_r_out)**2*u_out_4.dx(0)*v_out_4.dx(0)*dx\
    -J4_out*Z1*(1/H_theta_out)**2*u_out_4.dx(1)*v_out_4.dx(1)*dx\

    F0_MG_out=-0.5*Z2*J1_out*(1/H_r_out)**2*u_out_5.dx(0)*v_out_1.dx(0)*dx\
    -0.5*Z2*J1_out*((1/H_theta_out)**2)*u_out_5.dx(1)*v_out_1.dx(1)*dx\
    -Z2*J2_out*(1/H_r_out)**2*u_out_5.dx(0)*v_out_2.dx(0)*dx\
    -Z2*J2_out*((1/H_theta_out)**2)*u_out_5.dx(1)*v_out_2.dx(1)*dx\
    -Z2*J3_out*(1/H_r_out)**2*u_out_5.dx(0)*u_out_3.dx(0)*v_out_3*dx\
    -Z2*J3_out*((1/H_theta_out)**2)*u_out_5.dx(1)*u_out_3.dx(1)*v_out_3*dx\
    -4*Z2*J3_out*(1/H_r_out)**2*u_out_5.dx(0)*(u_out_2.dx(0)-u_out_1.dx(0))*u_out_3*v_out_3*dx\
    -4*Z2*J3_out*((1/H_theta_out)**2)*u_out_5.dx(1)*(u_out_2.dx(1)-u_out_1.dx(1))*u_out_3*v_out_3*dx\
    -Z2*J4_out*(1/H_r_out)**2*u_out_5.dx(0)*v_out_4.dx(0)*dx\
    -Z2*J4_out*((1/H_theta_out)**2)*u_out_5.dx(1)*v_out_4.dx(1)*dx\

    F0_curv_out=-J1_out*Z2*(1/H_r_out)**2*u_out_5.dx(0)*v_out_5.dx(0)*dx\
    -J1_out*Z2*((1/H_theta_out)**2)*u_out_5.dx(1)*v_out_5.dx(1)*dx\

    F_src_MG_out=0.5*exp(2*u_out_4-2*u_out_1)*(Z1*u_out_5-u_out_5-Z0)*v_out_1*J1_out*dx\
    +exp(2*u_out_4-2*u_out_1)*(Z1*u_out_5-u_out_5-Z0)*v_out_2*J2_out*dx\
    +0.5*exp(2*u_out_4-2*u_out_1)*(Z1*u_out_5-u_out_5-Z0)*v_out_4*J4_out*dx\
    -(1/3)*exp(2*u_out_4-2*u_out_1)*J1_out*(2*Z0+2*u_out_5-Z1*u_out_5)*v_out_5*dx

    F_nl_out = J1_out*Z1*(1/H_r_out)**2*u_out_1.dx(0)*u_out_2.dx(0)*v_out_1*dx\
    +J1_out*Z1*(1/H_theta_out)**2*u_out_1.dx(1)*u_out_2.dx(1)*v_out_1*dx\
    +J2_out*Z1*(1/H_r_out)**2*u_out_2.dx(0)*u_out_2.dx(0)*v_out_2*dx\
    +J2_out*Z1*(1/H_theta_out)*u_out_2.dx(1)*u_out_2.dx(1)*v_out_2*dx\
    -4*J3_out*Z1*(1/H_r_out)**2*u_out_3.dx(0)*u_out_1.dx(0)*v_out_3*dx\
    -4*J3_out*Z1*(1/H_theta_out)**2*u_out_3.dx(1)*u_out_1.dx(1)*v_out_3*dx\
    +3*J3_out*Z1*(1/H_r_out)**2*u_out_3.dx(0)*u_out_2.dx(0)*v_out_3*dx\
    +3*J3_out*Z1*(1/H_theta_out)**2*u_out_3.dx(1)*u_out_2.dx(1)*v_out_3*dx\
    +J4_out*Z1*(1/H_r_out)**2*u_out_1.dx(0)*u_out_1.dx(0)*v_out_4*dx\
    +J4_out*Z1*(1/H_theta_out)**2*u_out_1.dx(1)*u_out_1.dx(1)*v_out_4*dx\

    F_nl_MG_out=0.5*Z2*J1_out*(1/H_r_out)**2*u_out_2.dx(0)*u_out_5.dx(0)*v_out_1*dx\
    +0.5*Z2*J1_out*((1/H_theta_out)**2)*v_out_1*u_out_2.dx(1)*u_out_5.dx(1)*dx\
    +Z2*J2_out*(1/H_r_out)**2*u_out_2.dx(0)*u_out_5.dx(0)*v_out_2*dx\
    +Z2*J2_out*((1/H_theta_out)**2)*u_out_2.dx(1)*u_out_5.dx(1)*v_out_2*dx\
    -Z2*J3_out*(1/H_r_out)**2*u_out_3.dx(0)*u_out_5.dx(0)*v_out_3*dx\
    -Z2*J3_out*((1/H_theta_out)**2)*u_out_3.dx(1)*u_out_5.dx(1)*v_out_3*dx\
    -Z2*J4_out*(1/H_r_out)**2*u_out_5.dx(0)*(u_out_4.dx(0)-u_out_1.dx(0))*v_out_4*dx\
    -Z2*J4_out*((1/H_theta_out)**2)*u_out_5.dx(1)*(u_out_4.dx(1)-u_out_1.dx(1))*v_out_4*dx\
    +Z2*J1_out*(1/H_r_out)**2*u_out_2.dx(0)*u_out_5.dx(0)*v_out_5*dx\
    +Z2*J1_out*((1/H_theta_out)**2)*u_out_5.dx(1)*u_out_2.dx(1)*v_out_5*dx\

    #factor N/B
    FF = exp(u_out_2-2*u_out_1)
    F_rot_out = -0.5*J3_out*Z1*(1/H_r_out)**2*FF*FF*u_out_3.dx(0)*u_out_3.dx(0)*v_out_1*dx\
    -0.5*J3_out*Z1*FF*FF*((1/H_theta_out)**2)*u_out_3.dx(1)*u_out_3.dx(1)*v_out_1*dx\
    -0.75*J2_out*Z1*FF*FF*(1/H_r_out)**2*u_out_3.dx(0)*u_out_3.dx(0)*v_out_4*dx\
    -0.75*J2_out*Z1*FF*FF*u_out_3.dx(1)*((1/H_theta_out)**2)*u_out_3.dx(1)*v_out_4*dx\

    F_out = F0_out+F0_curv_out+F0_MG_out+F_src_MG_out\
    +F_nl_out+F_nl_MG_out+F_rot_out

    print('**********************************************')
    print('SOLVING IN OUTER DOMAIN. PLEASE WAIT')
    #solution of variational problem
    solve(F_out == 0, u_out, bcs_out)

    #Solution in outer domain are assumed spherically symmetric, therefore angular coordinate can be not only Pi/2
    data_x_out = []
    theta0 = np.pi/2

    result_solution_out=dir_name + 'e=' + str(e1)+ ','\
    + 'omega=' + str(Omega0) + '_res-out' + '.txt'

    G = open(result_solution_out, 'w')

    for i in range (1000):
        x1 = X1 + 0.001 * i * (X2 - X1)
        p = Point(x1,theta0)
        F0 = math.pow(2.71828,u_out(p.x(), p.y())[0])
        F1 = exp(u_out(p.x(),p.y())[1])/F0
        F2 = (u_out(p.x(), p.y())[2])/Omega
        F3 = exp(u_out(p.x(),p.y())[3])/F0
        F5 = u_out(p.x(),p.y())[4]
        m=2*(pow(F1,0.5)-1)*R1/(1-x1)
        if R1*r_g*10**(-5)/(1-x1)<100:
            mass3=m #we calculate for control ADM mass on r=100 km
        if R1*r_g*10**(-5)/(1-x1)<150:
            mass4=m #we calculate for control ADM mass on r=150 km
        if R1*r_g*10**(-5)/(1-x1)<200:
            mass5=m #we calculate for control ADM mass on r=200 km
        if R1*r_g*10**(-5)/(1-x1)<300:
            mass6=m #we calculate for control ADM mass on r=300 km. Values mass3,4,5,6 are vary but for good solution are close to each other and to gravitational mass 
        J_rot=((R1/(1-x1))**3)*F2*Omega/2
        data_x_out.append(R1*r_g*10**(-5)/(1-x1))
        data_x.append(R1*r_g*10**(-5)/(1-x1))
        data_m.append(m)
        data_J.append(J_rot)
        data_N.append(F0)
        data_B.append(F1)
        data_Om.append(F2)
        data_A.append(F3)
        data_R.append(F5)
        G.write(str(F3*R1*r_g*10**(-5)/(1-x1)))
        G.write(' ')
        G.write(str(F0))
        G.write(' ')
        G.write(str(F1))
        G.write(' ')
        G.write(str(F2))
        G.write(' ')
        G.write(str(F3))
        G.write(' ')
        G.write(str(F4))
        G.write(' ')
        G.write(str(4*F5))
        G.write('\n')
    G.close()

    #figures for illustration
    graphic_module(data_x,data_x_2,data_N,data_N_2,data_B,data_B_2,
                        data_Om,data_Om_2,data_A,data_A_2,
                        data_h,data_h_2,data_m,data_J,data_vel,data_R, data_R_2, e1,Omega0)
    
    #functional spaces for inner and outer domain
    V1 = FunctionSpace(mesh, P1)
    V2 = FunctionSpace(mesh_out, P2)

    Q1 = FunctionSpace(mesh, 'DG', 1)
    Q2 = FunctionSpace(mesh_out, 'DG', 1)

    p=Point(R1,np.pi/2)
    p_out=Point(X1,np.pi/2)


    #calculation of derivatives of unknown functions on boundary of inner and outer domain (R1, equatorial point). This calculation is needed for control.
    derivative=[]
    derivative_out=[]

    for i in range(5):
        u1 = project(u[i],V1)
        u1_out = project(u_out[i],V2)
        u1_dx = project(u1.dx(0), Q1) # u.dx(1) for y-derivative
        u1_out_dx = project(u1_out.dx(0), Q2) # u.dx(1) for y-derivative
        derivative.append(u1_dx(p.x(),p.y()))
        derivative_out.append(((1-X1)**2/R1)*u1_out_dx(p_out.x(),p_out.y()))

    delta=0
    delta_rel=0

    for i in range(2):
        delta=(derivative[i]-derivative_out[i])**2+delta
        delta_rel=(1-derivative_out[i]/derivative[i])**2+delta_rel

    class My_domain(SubDomain):
        def inside(self,x,on_boundary):
            return x[0]<=x_core and x[1]>=0

    class My_domain_1(SubDomain):
        def inside(self,x,on_boundary):
            return x[1]>=0

    subdomain = MeshFunction('size_t',mesh,mesh.topology().dim(), 0)
    subdomain.set_all(0)
    my_domain = My_domain()
    my_domain.mark(subdomain,1 )

    subdomain_1 = MeshFunction('size_t',mesh,mesh.topology().dim(), 0)
    subdomain_1.set_all(0)
    my_domain_1 = My_domain_1()
    my_domain_1.mark(subdomain_1,1 )

    dx_sub = Measure('dx', domain=mesh, subdomain_id=1, subdomain_data=subdomain)
    dx1_sub = Measure('dx', domain=mesh, subdomain_id=1, subdomain_data=subdomain_1)



    J_ang=4*np.pi*assemble(f3/((1+2*gamma*u_5))*exp(3*u_2-4*u_1)*(Omega*key1-u_3)*J3*dx_sub) #angular momentum
    J_ang0=4*np.pi*assemble(f3/((1+2*gamma*u_5))*exp(3*u_2-4*u_1)*(Omega*key1-u_3)*J3*dx1_sub) #abgular momentum for core (if we use limit on density)

    J_1_ang0=4*np.pi*assemble(f3/((1+2*gamma*u_5))*(Omega*key1-u_3)*J3*dx1_sub)\
    +(1/4)*assemble((3*u_2.dx(0)*u_3.dx(0)+3*u_2.dx(1)*u_3.dx(1)/H_theta**2)*J3*dx1_sub)\
    -assemble((u_1.dx(0)*u_3.dx(0)+u_1.dx(1)*u_3.dx(1)/H_theta**2)*J3*dx1_sub)\
    -(1/2)*assemble(gamma/((1+2*gamma*u_5))*(u_5.dx(0)*u_3.dx(0)+u_5.dx(1)*u_3.dx(1)/H_theta**2)*J3*dx1_sub)

    J_1_ang0=J_1_ang0 + (1/8)*assemble((3*u_out_2.dx(0)*u_out_3.dx(0)/H_r_out**2)*J3_out*dx)\
    -(1/2)*assemble((u_out_1.dx(0)*u_out_3.dx(0)/H_r_out**2)*J3_out*dx)\
    -(1/4)*assemble(gamma/((1+2*gamma*u_out_5))*(u_out_5.dx(0)*u_out_3.dx(0)/H_r_out**2)*J3_out*dx)

    Mass_ADM=4*np.pi*assemble(f1/((1+2*gamma*u_5))*exp(u_2)*J1*dx) #ADM mass as integral
    DM=4*np.pi*assemble(2*u_3*f3*exp(3*u_2-4*u_1)*(Omega*key1-u_3)*J3*dx) #addition term due to the rotation

    Mass_b=1.673*10**(-24)*4*np.pi*assemble(f6*exp(2*u_2-3*u_1+u_4)*J1*dx)*r_g**3*10**39/solar_mass #baryon mass 

    chi_0 = J_ang0/((mass5)**2) #parameter chi
    I_0 = (J_ang0/Omega)/(mass5)**3 #moment of inertia

    # GRV2_1=8*np.pi*assemble(f4*J4*dx)
    # GRV2_2=-assemble((u_1.dx(0)**2+(u_1.dx(1)/H_theta)**2)*J4*dx)
    # GRV2_2_out=-assemble(((u_out_1.dx(0)/H_r_out)**2+(u_out_1.dx(1)/H_theta_out)**2)*J4_out*dx)
    # GRV2_3=0.75*assemble((u_3.dx(0)**2+(u_3.dx(1)/H_theta)**2)*J3*exp(2*u_2-4*u_1)*dx)
    # GRV2_3_out=0.75*assemble(((u_out_3.dx(0)/H_r_out)**2+(u_out_3.dx(1)/H_theta_out)**2)*J3_out*FF*FF*dx)

    # print("GRV2=",(GRV2_1+GRV2_2+GRV2_2_out+GRV2_3)*2)

    #calculation of moments
    P_2 = Expression('1.5*pow(cos(x[1]),2)-0.5', degree=4) #Legendre polynom of 2 order

    DP_3 = Expression('7.5*pow(cos(x[1]),2)-1.5', degree=3) #derivative of Legendre polynom of 3 order

    factor_1=Expression('(x[0])*sin(x[1])',degree=3) #some factors
    factor_2 = Expression('x[0]*x[0]',degree=2)
    factor_4 = Expression('cos(x[1])/x[0]',degree=3)

    #sources functions in r.h.s. of equations 
    sigma_nu = 4*(np.pi)*f1/(1+2*gamma*u_5)\
    +0.5*factor_1**2*f0*u_3.dx(0)*u_3.dx(0)+0.5*factor_1**2*f0*((1/H_theta)**2)*u_3.dx(1)*u_3.dx(1)\
    -u_1.dx(0)*u_2.dx(0)-((1/H_theta)**2)*u_1.dx(1)*u_2.dx(1)

    sigma_b = 16*(np.pi)*f2*factor_1*exp(u_2)/(1+2*gamma*u_5)

    sigma_omega = -16*(np.pi)*f3*(Omega-u_3)*factor_1/(1+2*gamma*u_5)\
    +(4*u_3.dx(0)*u_1.dx(0)+4*((1/H_theta)**2)*u_3.dx(1)*u_1.dx(1))*factor_1\
    -(3*u_3.dx(0)*u_2.dx(0)+3*(1/H_theta)**2*u_3.dx(1)*u_2.dx(1))*factor_1

    #additions due to the modified gravity
    sigma_nu_f = -u_1.dx(0)*gamma*u_5.dx(0)/(1+2*gamma*u_5)-((1/H_theta)**2)*u_1.dx(1)*gamma*u_5.dx(1)/(1+2*gamma*u_5)\
    +4/((1+2*gamma*u_5))*np.pi*f5/3-exp(2*u_4-2*u_1)*(u_5)/(1+2*gamma*u_5)/6 + gamma*exp(2*u_4-2*u_1)*(u_5*u_5)/(1+2*gamma*u_5)/2

    sigma_omega_f = 2*gamma*factor_1/(1+2*gamma*u_5)*(u_5.dx(0)*u_3.dx(0)+((1/H_theta)**2)*u_5.dx(1)*u_3.dx(1))\
    +2*gamma*factor_1/(1+2*gamma*u_5)*(4*u_3)*(u_5.dx(0)*u_2.dx(0)+((1/H_theta)**2)*u_5.dx(1)*u_2.dx(1))\
    -2*gamma*factor_1/(1+2*gamma*u_5)*(4*u_3)*(u_5.dx(0)*u_1.dx(0)+((1/H_theta)**2)*u_5.dx(1)*u_1.dx(1))\
    #+2*gamma/(1+2*gamma*u_5)*(factor_3_out*u_5.dx(0)+u_5.dx(1)*factor_4)

    #moments. With postfix _f - addition due to modified gravity terms in sources. 
    #The main problem is that solution is not sufficiently accurate for calculation of such quantities. We have good results for mass, moment of inertia, but for 
    #calculation of moments one need to more big inner domain. R1 should be increased. We take robust assumption that for r>R1 solution posseses spherical symmetry. 
    # This assumption donesn't affect significantly on mass and moment inertia but values of M2 so far from reality. For S3 situation is even worse.
    # R1 is very difficult to increase because one need to construct initial solution in (0,R1) domain. But from some value of R1 procedure for solution doen't converge. As density 
    # increases R1 decreases. 

    if Omega0==0:
        q_2 = 0
        b_0 = -0.25
        q_2_f = 0

        omega_2 = 0
        omega_2_f = 0

    if Omega0!=0:
        q_2 = (mass5)**(-3)*assemble(sigma_nu*P_2*factor_2*J1*dx)
        q_2_f = (mass5)**(-3)*assemble(sigma_nu_f*P_2*factor_2*J1*dx)
        b_0 = -0.25
        omega_2 = (1/12)*(mass5)**(-4)*assemble(sigma_omega*DP_3*factor_2*factor_1*J1*dx)
        omega_2_f = (1/12)*(mass5)**(-4)*assemble(sigma_omega_f*DP_3*factor_2*factor_1*J1*dx)

    M_2 = -(1/3)*(1+4*b_0+3*(q_2))*(mass5)**(3)

    M_2f = -(1/3)*(1+4*b_0+3*(q_2_f))*(mass5)**(3)

    S_3 = -(3/10)*(2*chi_0+8*chi_0*b_0-5*omega_2-0*omega_2_f)*(mass5)**(4)

    S_3f = -(3/10)*(2*chi_0+8*chi_0*b_0-0*omega_2-5*omega_2_f)*(mass5)**(4)

    if Omega0==0:
        m_2 = 0
        s_3 = 0
    if Omega0!=0:
        m_2 = - M_2 / ((mass5)**(3)*chi_0**2)
        m_2f = - M_2f / ((mass5)**(3)*chi_0**2)
        s_3 = - S_3 / ((mass5)**(4)*chi_0**3)
        s_3f = - S_3f / ((mass5)**(4)*chi_0**3)

    print("************************************************************") #print value of derivatives for control. If derivatives are differ significantly one need to
    #derive solution again with another mass, mass2, curv. Correct solution should be continuous. Unfortunately this process I could do only in 'manual mode'. Therefore
    # fine-tuning take large time.

    print("Derivatives on boundary of inner domain =", derivative)
    print("Derivatives on boundary of outer domain =", derivative_out)
    print("sum of differences in square =", delta) 

    print("************************************************************")
    print("STAR PARAMETERS")
    print("gravitational mass =", mass5)
    print("baryon mass =", Mass_b)
    print("frequency of rotation =", Omega0)
    print("central energy density in MeV/fm^3 =", central_density*rho_0/(1.78E12))
    print("equatorial radius =", eq_radius)
    print("polar radius =",polar_radius)
    print("axis ratio =", polar_radius/eq_radius)
    print("circumferential equatorial radius =", circ_radius)
    print("compactness =", mass5*r_g/((10**5)*circ_radius))
    print("velocity at the equator =", eq_velocity)
    print("************************************************************")
    print("MULTIPOLE COEFFICIENTS")
    print("J =", J_ang0)
    print("J1 =", J_1_ang0)
    print("chi =", chi_0)
    #multipole moments
    print("I =", I_0)
    print("m2 = ", m_2)
    print("m2_f = ", m_2f)
    print("s3 =", s_3)
    print("s3_f =", s_3f)
    print("M2 =", M_2)
    print("M2_f", M_2f)
    print("S3 =", S_3)
    print("S3_f =", S_3f)
    print("b0 =", b_0)
    print("q2 =", q_2)
    print("q2_f =", q_2_f)
    print("omega2 =", omega_2)
    print("omega2_f =", omega_2_f)

    output = {}

    result_solution_3='init_files/' + 'e=' + str(e1)+ ','\
    + 'omega=' + str(Omega0) + '_sum' + '.txt'

    with open(result_solution_3, "w") as init_file:
        output["gravitational mass"] = str(mass5)
        output["baryon mass"] = str(Mass_b)
        output["mass"] = str(mass)
        output["mass2"] = str(mass2)
        output["mass on 100km"] = str(mass3)
        output["mass on 150km"] = str(mass4)
        output["mass on 200km"] = str(mass5)
        output["mass on 300km"] = str(mass6)
        output["m_r"] = str(moment_of_rotation)
        output["m_r2"]=str(moment_of_rotation2)
        output["curv"] = str(Curv_R)
        output["omega"] = str(Omega0)
        output["enthalpy"] = str(h0)
        output["gamma"] = str(gamma/4)
        output["density"] = str(central_density*rho_0/(1.78E12))
        output["eq_rad"] = str(eq_radius)
        output["polar_rad"] = str(polar_radius)
        output["r/a"] = str(polar_radius/eq_radius)
        output["R_eq"] = str(circ_radius)
        output["comp"] = str(mass5*r_g/((10**5)*circ_radius))
        output["J_ang"] = str(J_ang0)
        output["Kerr"] = str(J_ang0/(mass5**2))
        #output["moment of inertia (core)"] = str(J_ang*1475**2*solar_mass*0.001/Omega)
        output["moment of inertia"] = str(J_ang0*1475**2*solar_mass*0.001/Omega)
        output["R1"] = str(R1)
        output["resolution"] = str(resolution)
        output["I"] = str(I_0)
        output["M2"] = str(m_2)
        output["M2f"] = str(m_2f)
        output["S3"] = str(s_3)
        output["S3f"] = str(s_3f)
        output["b0"] = str(b_0)
        output["q2"] = str(q_2 + q_2_f)
        output["omega_2"] = str(omega_2 + omega_2_f)

        json.dump(output, init_file, indent=4, separators=(',', ': '))
