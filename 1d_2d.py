import numpy as np
import sympy as smp


def initial_from_1d(Name1, Name0, r_max, res_theta):

    G = open(Name1, 'r')
    
    lines = G.read().split()
    
    print(len(lines))
    
    k0 = int(res_theta)
    
    G.close()
    
    G = open(Name0, 'w')
    
    #считывание х- и y-координат узлов и значения энтальпии и скорости
    for i in range(0, len(lines)-5, 6):
        for k in range(0, k0):
            BN = float(lines[i+5])
            U = float(lines[i+4])
            A = float(lines[i+3])
            H = float(lines[i+2])
            r = float(lines[i])
            theta = float(k*np.pi/(2*res_theta-2))
    
            if r<=r_max and (i/2 - int(i/2))==0:
    
                G.write(str(r))
                G.write(' ')
                G.write(str(theta))
                G.write(' ')
                G.write(str(H))
                G.write(' ')
                G.write(str(A))
                G.write(' ')
                G.write(str(U))
                G.write(' ')
                G.write(str(BN))
                G.write('\n')
    
    G.close()

def expansion_from_2d(Name1, Mame0, mass, r_1, r_0, res_r, res_theta):
    
    G = open(Name1, 'r') 
    lines = G.read().split()
    
    print(len(lines))
    k0 = int(res_theta)
    
    G.close()
    
    G = open(Name0, 'w')
    
    H_min = 0
    
    #считывание х- и y-координат узлов и значения энтальпии и скорости
    for i in range(0, len(lines)-5, 6):

        BN = float(lines[i+5])
        U = float(lines[i+4])
        A = float(lines[i+3])
        H = float(lines[i+2])
        if H<H_min:
            H_min = H
            
        r = float(lines[i])
        theta = float(lines[i+1])

        G.write(str(r))
        G.write(' ')
        G.write(str(theta))
        G.write(' ')
        G.write(str(H))
        G.write(' ')
        G.write(str(A))
        G.write(' ')
        G.write(str(U))
        G.write(' ')
        G.write(str(BN))
        G.write('\n')
        
    for k in range(0, int(res_r)+1):
        for j in range(res_theta):
            
            r = r_1 + (r_0-r_1)*k/res_r
            theta = float(j*np.pi/(2*res_theta-2))
            
            U = 0
            H = H_min
            A = (1+mass/(2*r))**2
            N = (1-mass/(2*r))/(1+mass/(2*r))
            BN = (A/N)**2
            
            G.write(str(r))
            G.write(' ')
            G.write(str(theta))
            G.write(' ')
            G.write(str(H))
            G.write(' ')
            G.write(str(A))
            G.write(' ')
            G.write(str(U))
            G.write(' ')
            G.write(str(BN))
            G.write('\n')
            
    G.close()
                            
    

def reduce_2d(Name1, Name0, r_max):
    
    G = open(Name1, 'r')
    
    lines = G.read().split()
    
    print(len(lines))
    
    G.close()
    
    G = open(Name0, 'w')
    
    s=0
    
    #считывание х- и y-координат узлов и значения энтальпии и скорости
    for i in range(0, len(lines)-5, 6):
        
        s=s+1

        BN = float(lines[i+5])
        U = float(lines[i+4])
        A = float(lines[i+3])
        H = float(lines[i+2])
        r = float(lines[i])
        theta = float(lines[i+1])
    
        if r<=r_max:
    
            G.write(str(r))
            G.write(' ')
            G.write(str(theta))
            G.write(' ')
            G.write(str(H))
            G.write(' ')
            G.write(str(A))
            G.write(' ')
            G.write(str(U))
            G.write(' ')
            G.write(str(BN))
            G.write('\n')
    
    G.close()
    
    

Name1 = 'init_files/e=500,omega=200.txt'

Name0 = 'init_files/e=500,omega=200.txt'

initial_from_1d(Name1, Name0, 10, 5)

expansion_from_2d(Name1, Name0, 2.02, 10, 12, 100, 5)