import matplotlib.pyplot as plt
import os

dir_name='graphics' + '/'

if os.path.exists(dir_name):
    dir_name=dir_name
else:
    os.mkdir(dir_name, mode=0o777, dir_fd=None)

def graphic_module(data_x,data_x_2,data_N,data_N_2,data_B,data_B_2,
                    data_omega,data_omega_2,data_A,data_A_2,
                    data_h,data_h_2,data_m,data_J,data_vel,curv, curv_2, e1,Omega0):
    import numpy as np

    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x_2,data_N_2)
    plt.plot(data_x,data_N)
    plt.xlim([0, 100])
    plt.minorticks_on()
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'N.png')
    #график для функции B для двух углов
    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x_2,data_B_2)
    plt.plot(data_x,data_B)
    plt.xlim([0, 100])
    plt.minorticks_on()
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'B.png')
    #график для функции omega/Omega
    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x_2,data_omega_2)
    plt.plot(data_x,data_omega)
    plt.xlim([0, 100])
    plt.minorticks_on()
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'Omega.png')
    #график для функции A для двух углов
    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x_2,data_A_2)
    plt.plot(data_x,data_A)
    plt.plot(data_x_2,data_B_2)
    plt.plot(data_x,data_B)
    plt.xlim([0, 100])
    plt.minorticks_on()
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'A.png')
    #график для  энтальпии
    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x_2,data_h)
    plt.plot(data_x_2,data_h_2)
    plt.minorticks_on()
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'Enthalpy.png')
    #график для функции профиля массы на полярной очи
    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x,data_m)
    plt.minorticks_on()
    plt.xlim([0, 100])
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'M(r).png')

    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x,data_J)
    plt.minorticks_on()
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'J(r).png')

    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x_2,data_vel)
    plt.minorticks_on()
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'Velocity.png')

    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x,curv)
    plt.xlim([0, 100])
    plt.minorticks_on()
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'Curv.png')

def graphic_in(data_x,data_x_2,data_N,data_N_2,data_B,data_B_2,
                    data_omega,data_omega_2,data_A,data_A_2,
                    density,density_2,pressure,pressure_2,curv,curv_2,e1,Omega0):
    import numpy as np

    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x_2,data_N_2)
    plt.plot(data_x,data_N)
    plt.minorticks_on()
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'N-0.png')
    #график для функции B для двух углов
    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x_2,data_B_2)
    plt.plot(data_x,data_B)
    plt.minorticks_on()
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'B-0.png')
    #график для функции omega/Omega
    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x_2,data_omega_2)
    plt.plot(data_x,data_omega)
    plt.minorticks_on()
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'Omega-0.png')
    #график для функции A для двух углов
    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x_2,data_A_2)
    plt.plot(data_x,data_A)
    plt.minorticks_on()
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'A-0.png')
    #
    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x_2,density_2)
    plt.plot(data_x,density)
    plt.minorticks_on()
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'Density.png')
    #
    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x_2,pressure_2)
    plt.plot(data_x,pressure)
    plt.minorticks_on()
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'Pressure.png')


    fig = plt.figure(figsize=(8,8), facecolor='pink', frameon=True)
    plt.plot(data_x_2,curv_2)
    plt.plot(data_x,curv)
    plt.minorticks_on()
    plt.grid(which='major',linewidth = 2)
    plt.grid(which='minor')
    plt.savefig(dir_name+'e='+str(e1)+'_omega='+str(Omega0)+'_'+'Curv-0.png')
