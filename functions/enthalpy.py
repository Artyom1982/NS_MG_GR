import numpy as np
import sympy as smp
import math

def generate_src_function(lines, enthalpy0):
    """
    Функция для расчета плотности и давления по энтальпии.

    Входные аргументы:
    lines - список, считанный из уравнения состояния в формате
    энтальпия - плотность энергии - давление - концентрация
    (значения должны идти в столбцах в порядке возрастания)

    enthalpy0 - значение энтальпии, для которого надо рассчитать плотность и давление.

    Алгоритм вычисления состоит в следующем. Функция находит наименьшее значение энтальпии
    в списке, которое больше заданного. Предыдущее значение в списке тогда будет меньше
    заданного. Тогда значение плотности (давления) для заданной энтальпии заключено
    между плотностями (давлениями), соответствующими табличным значениям энтальпии.
    Поскольку логарифм плотности (давления) с хорошей точностью линейно зависит энтальпии,
    то путем простой аппроксимации можно найти плотоость и давление для заданной энтальпии.

    Если энтальпия меньше нуля, то это означает, что плотность и давление равны нулю.
    """
    # Создание массивов
    density = []
    pressure = []
    enthalpy = []
    n = []

    # Заполнение массовов из считанных данных
    for i in range(0, len(lines)-3, 4):
        enthalpy.append(float(lines[i]))
        density.append(float(lines[i+1]))
        pressure.append(float(lines[i+2]))
        n.append(float(lines[i+3]))

    for i in range(len(pressure)):
        if enthalpy0<-0.002:
            result1=0
            result2=0
            result3=0
            break
        else:
            for j in range(len(pressure)):
                if enthalpy0<enthalpy[j]:
                    K1=(pressure[j]-pressure[j-1])/(math.exp(enthalpy[j])-math.exp(enthalpy[j-1]))
                    K2=(density[j]-density[j-1])/(math.exp(enthalpy[j])-math.exp(enthalpy[j-1]))
                    result2=pressure[j-1]+(math.exp(enthalpy0)-math.exp(enthalpy[j-1]))*K1
                    result1=density[j-1]+(math.exp(enthalpy0)-math.exp(enthalpy[j-1]))*K2
                    K3=(n[j]-n[j-1])/(math.exp(enthalpy[j])-math.exp(enthalpy[j-1]))
                    result3=n[j-1]+(math.exp(enthalpy0)-math.exp(enthalpy[j-1]))*K3
                    break

    return result1, result2, result3

def energy_enthalpy(Name, energy0):
    # Создание массивов
    density = []
    enthalpy = []

    F1=open(Name,'r')
    lines = F1.read().split()

    # Заполнение массовов из считанных данных
    for i in range(0, len(lines)-3, 4):
        enthalpy.append(float(lines[i]))
        density.append(float(lines[i+1]))

    enthalpy_min = min(enthalpy)

    for j in range(len(density)):
        if energy0<density[j]:
            K2=(math.exp(enthalpy[j])-math.exp(enthalpy[j-1]))/(density[j]-density[j-1])
            result1=math.exp(enthalpy[j-1])+(energy0-density[j-1])*K2
            result1=math.log(result1)
            break

    return result1, enthalpy_min
