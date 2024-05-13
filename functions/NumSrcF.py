from fenics import *
import numpy as np
from mshr import *
import sympy as smp
from enthalpy import generate_src_function
import matplotlib.pyplot as plt

def numer_src_function(Name1, Name2, n):
    """
    Функция для расчета плотности и давления в узлах координатной сетки.

    Входные параметры:

    Name1 - имя файла, содержащего информацию об уравнении состояния.
    Формат: энтальпия - плотность - давление. Значения должны быть в порядке возрастания
    в столбцах

    Name2 - имя файла, содержащего значения энтальпии и скорости вращения в узлах сетки.
    Формат: x-координата узла - у-координата узла - энтальпия - A - U

    n - количество узлов сетки, т.е. фактически количество строк в файле Name1.
    Можно переписать код так, что этот аргумент не будет требоааться.

    Выходные данные: два списка -  плотность и давление в узлах сетки

    """
    # source term calculation

    #объявление списков
    f0_values = np.zeros(n)
    f1_values = np.zeros(n)
    f2_values = np.zeros(n)
    f3_values = np.zeros(n)
    f4_values = np.zeros(n)
    f5_values = np.zeros(n)
    f6_values = np.zeros(n)
    B_N_values = np.zeros(n)

    #открытие файла со значеииями энтальпии в узлах сетки
    G = open(Name1, 'r')

    lines = G.read().split()

    G.close()

    #открытие файла с рравнением состояния и считывание
    F1=open(Name2,'r')
    lines1 = F1.read().split()

    F1.close()

    x = []
    y= []
    H = []
    A= []
    U= []
    BN=[]

    #считывание х- и y-координат узлов и значения энтальпии и скорости
    for i in range(0, len(lines1)-5, 6):
        BN.append(float(lines1[i+5]))
        U.append(float(lines1[i+4]))
        A.append(float(lines1[i+3]))
        H.append(float(lines1[i+2]))
        x.append(float(lines1[i]))
        y.append(float(lines1[i+1]))

    #вызов функции, которая по энтальпии дает значение плотности и давления
    for i in range(n):
        enthalpy0=H[i]
        result=generate_src_function(lines,enthalpy0)
        f0_values[i]=A[i]*A[i]*(result[0]+result[1])*pow(1-U[i]*U[i],-1/2)\
        -A[i]*A[i]*result[1]
        f1_values[i]=3*A[i]*A[i]*result[1]+f0_values[i]*U[i]*U[i]\
        +A[i]*A[i]*result[1]*U[i]*U[i]+f0_values[i]
        f2_values[i]=A[i]*A[i]*result[1]
        f3_values[i]=f0_values[i]+f2_values[i]
        f4_values[i]=f2_values[i]+f3_values[i]*U[i]*U[i]
        f5_values[i]=-3*A[i]*A[i]*result[1]-f0_values[i]*U[i]*U[i]\
        -A[i]*A[i]*result[1]*U[i]*U[i]+f0_values[i]
        B_N_values[i]=BN[i]
        f6_values[i]=result[2]
        if x[i]>15:
            f1_values[i]=0
            f2_values[i]=0
            f3_values[i]=0
            f4_values[i]=0
            f5_values[i]=0

    return f1_values, f2_values, f3_values, f4_values, f5_values, B_N_values, f6_values

def numer_src_function0(Name1, H0):
    """


    """

    #открытие файла со значеииями энтальпии в узлах сетки
    G = open(Name1, 'r')

    lines = G.read().split()

    G.close()

    result=generate_src_function(lines,H0)

    return result[0], result[1]
