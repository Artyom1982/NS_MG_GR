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
core_density = 1*10**(14)/rho_0

def SOLUTION_OF_TOV(args):

    # Начальные параметры для решения задачи
    R0=0.00 #координата центра звезды, не изменяется

    R2=400 #конечная радиальная координата внешней области

    #относительная погрешность, с которой фиксируется приближение к границе
    tol=1E-14

    #плотность энергии в центре (МэВ/фм^3)
    e1 = args[1]
    Omega0 = args[2]
    Name0='init_files/' + 'e=' + str(e1)+ ',' + 'omega=' + str(Omega0) + '_sum.txt'
    Name1='init_files/' + 'e=' + str(e1)+ ',' + 'omega=' + str(Omega0) + '.txt'
    Name_EOS = 'data/'+args[0]+'.txt'

    energy0 = e1*1.78E12/rho_0
    h0 = energy_enthalpy(Name_EOS,energy0)[0]
    h0_min = energy_enthalpy(Name_EOS,energy0)[1]

    d_Omega = args[3]
    delta_m = args[4]
    delta_J = args[5]
    delta_J2 = args[6]

    with open(Name0) as f:
        data = json.load(f)
    mass=float(sympify(data["mass"]))
    mass=mass+delta_m
    moment_of_rotation=0*float(sympify(data["m_r"])) + delta_J

    moment_of_rotation2=0*float(sympify(data["m_r2"])) + delta_J2

    Omega0=int(sympify(data["omega"]))
    Omega0=Omega0+d_Omega
    R1=float(sympify(data["R1"]))
    R1 = R1 + args[7]

    R_int = 1.25*R1

    #вычисление конформных координат границ внешней области
    X1=1-R1/R1
    X2=1-R_int/R2

    #относительная погрешность, с которой фиксируется приближение к границе
    tol=1E-14

    Omega1=Omega0

    #перевод частоты вращения в безразмерные единицы
    Omega=2*np.pi*Omega0*r_g/speed_of_light
    #приблизительный радиус звезды (для первой итерации). Следует учесть, что
    #величина задается не в километрах. Физический радиус звезды равен этому числу
    #умноженному на 1.475 и на значение фактора A на границе звезды.
    radius=10
    #ключ (значение 0 указывает на необходимость начинать решение задачи с нулевого
    #приближения, любое другое - если в качестве начального используется решение,
    #полученное ранее)
    key=1
    #число итераций для нахождения решения задачи во внутренней области
    #(обычно 50-100 для поиска решения с ""нуля"". Если требуется пересчитать решение
    #во внешней области, то можно поставить 1 для сокращения времени)
    Number_of_iterations=args[8]
    #имена файлов, в которые записывается решение (могу затем использоваться как
    #стартовые при переименовании в H.txt и Factors.txt соответственно)
    dir_name='results' + '/'

    if os.path.exists(dir_name):
        dir_name=dir_name
    else:
        os.mkdir(dir_name, mode=0o777, dir_fd=None)

    #внутренняя область, в которой ищется решение
    domain=Rectangle(Point(R0,0), Point(R1, np.pi/2))
    #генерация сетки
    mesh=generate_mesh(domain, args[9])

    #расчет граничных условий
    A_R=(1+mass/(2*R1))**2
    N_R=(1-mass/(2*R1))/(1+mass/(2*R1))
    A_inf=(1+mass/(2*R2))**2
    N_inf=(1-mass/(2*R2))/(1+mass/(2*R2))

    A_R1=(1+mass/(2*R1))**2
    N_R1=(1-mass/(2*R1))/(1+mass/(2*R1))

    A_int = (1+mass/(2*R_int))**2
    N_int = (1-mass/(2*R_int))/(1+mass/(2*R_int))

    #граничные условия на правой границе внутренней области
    nu_R=ln(N_R)
    mu_R=ln(A_R*N_R)
    omega_R=2*moment_of_rotation2/(R1**3)
    #граничные условия на правой границе внешней области
    omega_inf=2*moment_of_rotation/((R2)**3)
    nu_inf=ln(N_inf)
    mu_inf=ln(N_inf*A_inf)
    #граничные условия на правой границе промежуточной области
    omega_int=2*moment_of_rotation/((R_int)**3)
    nu_int=ln(N_int)
    mu_int=ln(N_int*A_int)

    #задание граничных условий для функций на правой границе внутренней области (u_R),
    #левой границе внешней области (u_B) и правой границе внешней области (u_inf), правой границе промежуточной области (u_int)
    u_R=[Constant(nu_R), Constant(mu_R), Constant(omega_R), Constant(mu_R)]
    u_B=[Constant(nu_R), Constant(mu_R), Constant(omega_R), Constant(mu_R)]
    u_inf=[Constant(nu_inf), Constant(mu_inf), Constant(omega_inf), Constant(mu_inf)]
    u_int_0 = [Constant(nu_int), Constant(mu_int), Constant(omega_int), Constant(mu_int)]

    # задание функционального пространства для функции источников
    SRC = FunctionSpace(mesh, 'CG', 1)

    #количество элементов пространства (оличество узлов, в которых определяется Функция
    #источника), размерность пространства
    n = SRC.dim()
    d = mesh.geometry().dim()

    #массив координат узлов
    F_dof_coordinates = SRC.tabulate_dof_coordinates().reshape(n,d)
    F_dof_coordinates.resize((n, d))

    #массив x-координат узлов и y-ооординат узлов
    F_dof_x = F_dof_coordinates[:, 0]
    F_dof_y = F_dof_coordinates[:, 1]

    x=sym.Symbol('x')
    y=sym.Symbol('y')

    # функция, определяющая приблизительный профиль энтальпии (для первого шага)
    h=h0*(1-x*x/(radius*radius))

    # создание файлов H.txt, в которой записываются построчно x и y-координаты узлов;
    # значение энтальпии, скорости вращения в этих узлах (H); фактора (B/N)**2 в уравнениях.
    # Координаты узлов берутся из массивов F_dof_x, F_dof_y.
    # Этот шаг пропускается, если ключ key не равен нулю
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

    # вызов функций, которая дает правые части уравнений, связанные с плотностью и
    #давлением (result, result2, result3) и множитель (B/N)**2
    result=numer_src_function(Name_EOS, Name1, n)
    #численно заданные функции источников в уравнениях (f1, f2, f3, f4) и фактор
    #(B/N)**2 (f0)
    f1 = Function(SRC)
    f2 = Function(SRC)
    f3 = Function(SRC)
    f4 = Function(SRC)
    f0 = Function(SRC)
    f6 = Function(SRC)
    f1.vector()[:] = result[0]
    f2.vector()[:] = result[1]
    f3.vector()[:] = result[2]
    f4.vector()[:] = result[3]
    f0.vector()[:] = result[5]
    f6.vector()[:] = result[6]

    #Функциональное пространство неизвестных и пробных функций
    P1 = FiniteElement('CG', triangle, 3)
    element = MixedElement([P1, P1, P1, P1]) #по числу функций
    V = FunctionSpace(mesh, element)

    #определение правой внутренней области.
    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], R1, tol)

    #задание граничных условий Дирихле на правой граиице
    bcs = DirichletBC(V, u_R, boundary_R)

    #Коэффициенты Ламе и якобианы переходов во внутренней области.
    H_theta=Expression('x[0]', degree=2)
    J1=Expression('pow(x[0],2)*sin(x[1])',degree=2)
    J2=Expression('pow(x[0]*sin(x[1]),2)*x[0]',degree=2)
    J3=Expression('pow(x[0]*sin(x[1]),3)*x[0]',degree=2)
    J4=Expression('x[0]',degree=2)

    #То же для внешней области. С учетом конформного преобразования r=R1/(1-x)
    H_r_out=R1*Expression('pow(1/(1-x[0]),2)', degree=2)
    H_theta_out=R1*Expression('1/(1-x[0])',degree=2)
    J1_out=H_r_out*H_theta_out*H_theta_out
    J2_out=H_r_out*H_theta_out*H_theta_out*H_theta_out
    J3_out=H_r_out*H_theta_out*H_theta_out*H_theta_out*H_theta_out
    J4_out=H_r_out*H_theta_out

    # Запуск итераций решения задачи. Используя решение на 1-м шаге, необходимо
    # пересчитать энтальпию и скорость вращения в узлах. Используется закон сохранения
    # H(r)+nu(r)+lnГ=const. Значение константы равно сумме h0 (изначально заданная энтальпия
    # в центре) и значению nu в центре. Таким образом, H=h0+nu(0)-nu(r)-lnГ.
    #Множитель Г=(1-U**2)**(-1/2). В файл Factors.txt записывается значение множителей
    #B/N в узлах сетки

    for k in range(Number_of_iterations):
        print()
        print('***********************************************************')
        print('ITERATION =',k+1)

        # повторение части кода, связанного с вариационной задачей.
        v_1, v_2, v_3, v_4 = TestFunctions(V)
        u = Function(V)
        u_1, u_2, u_3, u_4 = split(u)

        F0 = -J1*u_1.dx(0)*v_1.dx(0)*dx-J1*((1/H_theta)**2)*u_1.dx(1)*v_1.dx(1)*dx \
        -J2*u_2.dx(0)*v_2.dx(0)*dx-J2*((1/H_theta)**2)*u_2.dx(1)*v_2.dx(1)*dx \
        -J3*u_3.dx(0)*v_3.dx(0)*dx-J3*((1/H_theta)**2)*u_3.dx(1)*v_3.dx(1)*dx \
        -J4*u_4.dx(0)*v_4.dx(0)*dx-J4*((1/H_theta)**2)*u_4.dx(1)*v_4.dx(1)*dx

        F_nl = J1*v_1*u_1.dx(0)*u_2.dx(0)*dx+J1*((1/H_theta)**2)*v_1*u_1.dx(1)*u_2.dx(1)*dx \
        +J2*v_2*u_2.dx(0)*u_2.dx(0)*dx+J2*((1/H_theta)**2)*v_2*u_2.dx(1)*u_2.dx(1)*dx \
        -4*J3*u_3.dx(0)*u_1.dx(0)*v_3*dx-4*J3*((1/H_theta)**2)*u_3.dx(1)*u_1.dx(1)*v_3*dx \
        +3*J3*u_3.dx(0)*u_2.dx(0)*v_3*dx+3*J3*(1/H_theta)**2*u_3.dx(1)*u_2.dx(1)*v_3*dx \
        +J4*u_1.dx(0)*u_1.dx(0)*v_4*dx+J4*((1/H_theta)**2)*u_1.dx(1)*u_1.dx(1)*v_4*dx \

        F_src = -J1*4*(np.pi)*f1*v_1*dx \
        -J2*16*(np.pi)*f2*v_2*dx \
        +J3*16*(np.pi)*f3*(Omega-u_3)*v_3*dx \
        -8*J4*np.pi*f4*v_4*dx

        F_rot = -0.5*J3*f0*u_3.dx(0)*u_3.dx(0)*v_1*dx\
        -0.5*J3*f0*((1/H_theta)**2)*u_3.dx(1)*u_3.dx(1)*v_1*dx\
        -0.75*J2*f0*u_3.dx(0)*u_3.dx(0)*v_4*dx\
        -0.75*J2*f0*((1/H_theta)**2)*u_3.dx(1)*u_3.dx(1)*v_4*dx\

        F = F0+F_nl+F_src+F_rot

        solve(F == 0, u, bcs)

        #вывод на экран необязателен, просто для контроля
            #вычисление nu0 - значения nu=u_1 в центре
        p0=Point(R0,0)
        nu0=u(p0.x(), p0.y())[0]
        print(nu0)

        Name1='init_files/' + 'e=' + str(e1)+ ',' + 'omega=' + str(Omega0) + '.txt'

        #запись пересчитанной энтальпии и фактора (B/N)**2 в узлы сетки в файлы
        G = open(Name1, 'w')
        for i in range(n):
            a=[]
            p=Point(F_dof_x[i], F_dof_y[i])
            nu=u(p.x(), p.y())[0] #значение nu в узле
            scale=exp(u(p.x(), p.y())[3]-u(p.x(), p.y())[0]) #функция A
            scale2=exp(u(p.x(), p.y())[1]-2*u(p.x(), p.y())[0]) #функция B/N
            #пересчет скорости и энтальпии в узле
            velocity=scale2*F_dof_x[i]*sin(F_dof_y[i])*(Omega-u(p.x(), p.y())[2])
            H=h0+nu0-nu-0.5*ln(1-velocity**2)
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
        result = numer_src_function(Name_EOS, Name1,n)

        f1 = Function(SRC)
        f2 = Function(SRC)
        f3 = Function(SRC)
        f4 = Function(SRC)
        f0 = Function(SRC)
        f6 = Function(SRC)

        f1.vector()[:] = result[0]
        f2.vector()[:] = result[1]
        f3.vector()[:] = result[2]
        f4.vector()[:] = result[3]
        f0.vector()[:] = result[5]
        f6.vector()[:] = result[6]

    #вычисление неизвестных функций. Для примера создается массив - зависимость
    #функций от расстояния (r) на экваторе (theta=Pi/2) и на полюсе (theta=0)
    theta1 = np.pi/30
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
    data_m = []
    data_h = []
    data_J = []
    density = []
    pressure = []
    curv = []

    data_x_2 = []
    data_N_2 = []
    data_B_2 = []
    data_Om_2 = []
    data_A_2 = []
    data_h_2 = []
    data_vel = []
    density_2 = []
    pressure_2 = []
    curv_2 = []

    filename_solution_1=dir_name + 'e=' + str(e1)+ ','\
    +'omega='+str(Omega1)+'_res.txt'

    filename_solution_2=dir_name + 'e=' + str(e1)+ ','\
    +'omega='+str(Omega1)+'_res-eq.txt'

    G = open(filename_solution_1, 'w')
    G2 = open(filename_solution_2, 'w')

    for i in range (2000):
        x1 = R0 + 0.0005 * i * (R1 - R0)
        p1 = Point(x1,theta1)
        p2 = Point(x1,theta2)
        F0 = exp(u(p1.x(), p1.y())[0]) #функция N=exp(nu)
        F1 = exp(u(p1.x(), p1.y())[1])/F0 #функция B
        F2 = u(p1.x(),p1.y())[2]/Omega #Функция omega/Omega
        F3 = exp(u(p1.x(), p1.y())[3])/F0 #функция A
        F4 = nu0 + h0 - ln(F0) #энтальпия на полярной оси
        data_x.append(F3*x1*r_g/(10**5)) #координатное расстояние (1.475 - 1/2 грав. радиуса Солнца)
        data_N.append(F0)
        data_B.append(F1)
        data_Om.append(F2)
        data_A.append(F3)
        data_h.append(F4)
        density.append(numer_src_function0(Name_EOS,F4)[0])
        if density[i]>=core_density:
            x_core = x1
            R_core = F3*x1*r_g/(10**5)
        pressure.append(numer_src_function0(Name_EOS,F4)[1])
        curvature=8*np.pi*(numer_src_function0(Name_EOS,F4)[0]-3*numer_src_function0(Name_EOS,F4)[1])
        curv.append(curvature)
        m = 2*(pow(F1,0.5)-1)*x1
        data_m.append(m)
        G.write(str(F3*x1*r_g/(10**5)))
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
        G.write(str(4*curvature))
        G.write('\n')
        #то же, что и ранее, но для экваториальной плоскости
        F0 = exp(u(p2.x(), p2.y())[0])
        F1 = exp(u(p2.x(), p2.y())[1])/F0
        F2 = u(p2.x(),p2.y())[2]/Omega
        F3 = exp(u(p2.x(), p2.y())[3])/F0
        vel = (F1/F0) * x1 * Omega * (1 - F2) * key1
        F4 = nu0 + h0 - ln(F0) - 0.5 * ln(1 - vel * vel)
        data_x_2.append(F3*x1*1*r_g/(10**5))
        data_N_2.append(F0)
        data_B_2.append(F1)
        data_Om_2.append(F2)
        data_A_2.append(F3)
        data_h_2.append(F4)
        data_vel.append(vel)
        density_2.append(numer_src_function0(Name_EOS,F4)[0])
        pressure_2.append(numer_src_function0(Name_EOS,F4)[1])
        curvature=8*np.pi*(numer_src_function0(Name_EOS,F4)[0]-3*numer_src_function0(Name_EOS,F4)[1])
        curv_2.append(curvature)
        J_rot = (x1**3) * Omega * F2/2
        data_J.append(J_rot)
        if density_2[i]>=core_density:
            x_core = x1
            R_core = F3*x1*r_g/(10**5)
        G2.write(str(x1*F3*r_g/(10**5)))
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
        G2.write(str(4*curvature))
        G2.write('\n')

    G.close()
    G2.close()

    graphic_in(data_x,data_x_2,data_N,data_N_2,data_B,data_B_2,
                        data_Om,data_Om_2,data_A,data_A_2,
                        density, density_2, pressure, pressure_2,curv,curv_2,e1,Omega0)

    for i in range(2000):
        if data_h[i]<-0.0088:
            polar_radius=data_x[i]/data_B[i]
            break

    for i in range(2000):
        if data_h_2[i]<-0.0088:
            eq_radius=data_x_2[i]/data_B_2[i]
            circ_radius=data_x_2[i]
            eq_velocity=data_vel[i]
            break

    #решение задачи в промежуточной области. Все обозначения аналогичны использованным
    #ранее, только с постфиксом _int для сетки, пробных и неизвестных функций
    domain_int = Rectangle(Point(R1,0), Point(R_int, np.pi/2))
    mesh_int = generate_mesh(domain_int, 32)

    P2 = FiniteElement('CG', triangle, 3)
    element = MixedElement([P2, P2, P2, P2])

    V_int = FunctionSpace(mesh_int,element)
    
    u_int = Function(V_int)
    u_int_1, u_int_2, u_int_3, u_int_4 = split(u_int)
    v_int_1, v_int_2, v_int_3, v_int_4 = TestFunctions(V_int)

    def boundary_left(x, on_boundary):
        return on_boundary and near(x[0], R1, tol)
    def boundary_int(x, on_boundary):
        return on_boundary and near(x[0], R_int, tol)

    bcs_int=[]
    bcs_left=DirichletBC(V_int, u_B, boundary_left)
    bcs_int.append(bcs_left)
    bcs_right=DirichletBC(V_int, u_int_0, boundary_int)
    bcs_int.append(bcs_right)

    #линейная часть уравнений Пуассона с неймановскими условиями (для нашей задачи
    #можно опустить последнюю строчку, т.к. условия на производные нулевые)
    F0_int = -J1*u_int_1.dx(0)*v_int_1.dx(0)*dx\
    -J1*(1/H_theta)**2*u_int_1.dx(1)*v_int_1.dx(1)*dx\
    -J2*u_int_2.dx(0)*v_int_2.dx(0)*dx\
    -J2*(1/H_theta)**2*u_int_2.dx(1)*v_int_2.dx(1)*dx\
    -J3*u_int_3.dx(0)*v_int_3.dx(0)*dx\
    -J3*(1/H_theta)**2*u_int_3.dx(1)*v_int_3.dx(1)*dx\
    -J4*u_int_4.dx(0)*v_int_4.dx(0)*dx\
    -J4*(1/H_theta)**2*u_int_4.dx(1)*v_int_4.dx(1)*dx\
    #нелинейные слагаемые (по порядку уравнений, 2 строчки ~ одно уравнение,
    #3-е уравнение системы на 4 строчках)
    F_nl_int = J1*u_int_1.dx(0)*u_int_2.dx(0)*v_int_1*dx\
    +J1*(1/H_theta)**2*u_int_1.dx(1)*u_int_2.dx(1)*v_int_1*dx\
    +J2*u_int_2.dx(0)*u_int_2.dx(0)*v_int_2*dx\
    +J2*(1/H_theta)*u_int_2.dx(1)*u_int_2.dx(1)*v_int_2*dx\
    -4*J3*u_int_3.dx(0)*u_int_1.dx(0)*v_int_3*dx\
    -4*J3*(1/H_theta)**2*u_int_3.dx(1)*u_int_1.dx(1)*v_int_3*dx\
    +3*J3*u_int_3.dx(0)*u_int_2.dx(0)*v_int_3*dx\
    +3*J3*(1/H_theta)**2*u_int_3.dx(1)*u_int_2.dx(1)*v_int_3*dx\
    +J4*u_int_1.dx(0)*u_int_1.dx(0)*v_int_4*dx\
    +J4*(1/H_theta)**2*u_int_1.dx(1)*u_int_1.dx(1)*v_int_4*dx\
    #фактор N/B
    FF_int = exp(u_int_2-2*u_int_1)
    #слагаемые, связанные с вращением
    F_rot_int = -0.5*J3*FF_int*FF_int*u_int_3.dx(0)*u_int_3.dx(0)*v_int_1*dx\
    -0.5*J3*FF_int*FF_int*((1/H_theta)**2)*u_int_3.dx(1)*u_int_3.dx(1)*v_int_1*dx\
    -0.75*J2*FF_int*FF_int*u_int_3.dx(0)*u_int_3.dx(0)*v_int_4*dx\
    -0.75*J2*FF_int*FF_int*u_int_3.dx(1)*((1/H_theta)**2)*u_int_3.dx(1)*v_int_4*dx\

    F_int = F0_int+F_nl_int+F_rot_int

    print('**********************************************')
    print('SOLVING IN INTERMEDIATE DOMAIN. PLEASE WAIT')

    #решение вариационной задачи
    solve(F_int == 0, u_int, bcs_int)

    #формирование массивов искомых функций во внешней области. Решение считается
    #сферически симметричным, потому значение theta не имеет значения
    data_x_out = []
    theta0 = np.pi/2

    result_solution_int=dir_name + 'e=' + str(e1)+ ','\
    + 'omega=' + str(Omega1) + '_res-int' + '.txt'

    G = open(result_solution_int, 'w')

    for i in range (1000):
        x1 = R1 + 0.001 * i * (R_int - R1)
        p = Point(x1,theta0)
        F0 = exp(u_int(p.x(), p.y())[0])
        F1 = exp(u_int(p.x(),p.y())[1])/F0
        F2 = (u_int(p.x(), p.y())[2])/Omega
        F3 = exp(u_int(p.x(),p.y())[3])/F0
        m = 2*(pow(F1,0.5)-1)*x1
        J_rot = (x1**3) * Omega * F2/2
        data_x_out.append(F3*x1*r_g/(10**5))
        data_x.append(F3*x1*r_g/(10**5))
        data_m.append(m)
        data_J.append(J_rot)
        data_N.append(F0)
        data_B.append(F1)
        data_Om.append(F2)
        data_A.append(F3)
        curv.append(0)
        G.write(str(F3*x1*r_g/(10**5)))
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
        G.write('\n')

    G.close()


    # задание функционального пространства для функции источников
    SRC_int = FunctionSpace(mesh_int, 'CG', 1)

    #количество элементов пространства (оличество узлов, в которых определяется Функция
    #источника), размерность пространства
    n = SRC_int.dim()
    d = mesh_int.geometry().dim()

    #массив координат узлов
    F_dof_coordinates = SRC_int.tabulate_dof_coordinates().reshape(n,d)
    F_dof_coordinates.resize((n, d))

    #массив x-координат узлов и y-ооординат узлов
    F_dof_x = F_dof_coordinates[:, 0]
    F_dof_y = F_dof_coordinates[:, 1]


    Name10='init_files/' + 'e=' + str(e1)+ ',' + 'omega=' + str(Omega0) + '_int.txt'

    #запись пересчитанной энтальпии и фактора (B/N)**2 в узлы сетки в файлы
    G = open(Name10, 'w')
    p0=Point(R0,0)
    nu0=u(p0.x(), p0.y())[0]
    for i in range(n):
        a=[]
        p=Point(F_dof_x[i], F_dof_y[i])
        nu=u_int(p.x(), p.y())[0] #значение nu в узле
        scale=exp(u_int(p.x(), p.y())[3]-u_int(p.x(), p.y())[0]) #функция A
        scale2=exp(u_int(p.x(), p.y())[1]-2*u_int(p.x(), p.y())[0]) #функция B/N
        #пересчет скорости и энтальпии в узле
        velocity=scale2*F_dof_x[i]*sin(F_dof_y[i])*(key1*Omega-u_int(p.x(), p.y())[2])
        print(velocity)
        print(nu)
        H=h0+nu0-nu-0.5*ln(1-velocity**2)
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




    #решение задачи во внешней области. Все обозначения аналогичны использованным
    #ранее, только с постфиксом _out для сетки, пробных и неизвестных функций
    domain_out = Rectangle(Point(X1,0), Point(X2, np.pi/2))
    mesh_out = generate_mesh(domain_out, 32)

    P2 = FiniteElement('CG', triangle, 3)
    element = MixedElement([P2, P2, P2, P2])

    V_out = FunctionSpace(mesh_out,element)
    
    u_out = Function(V_out)
    u_out_1, u_out_2, u_out_3, u_out_4 = split(u_out)
    v_out_1, v_out_2, v_out_3, v_out_4 = TestFunctions(V_out)

    def boundary_left(x, on_boundary):
        return on_boundary and near(x[0], X1, tol)
    def boundary_inf(x, on_boundary):
        return on_boundary and near(x[0], X2, tol)

    bcs_out=[]
    bcs_left=DirichletBC(V_out, u_int_0, boundary_left)
    bcs_out.append(bcs_left)
    bcs_inf=DirichletBC(V_out, u_inf, boundary_inf)
    bcs_out.append(bcs_inf)

    #линейная часть уравнений Пуассона с неймановскими условиями (для нашей задачи
    #можно опустить последнюю строчку, т.к. условия на производные нулевые)
    F0_out = -J1_out*(1/H_r_out)**2*u_out_1.dx(0)*v_out_1.dx(0)*dx\
    -J1_out*(1/H_theta_out)**2*u_out_1.dx(1)*v_out_1.dx(1)*dx\
    -J2_out*(1/H_r_out)**2*u_out_2.dx(0)*v_out_2.dx(0)*dx\
    -J2_out*(1/H_theta_out)**2*u_out_2.dx(1)*v_out_2.dx(1)*dx\
    -J3_out*(1/H_r_out)**2*u_out_3.dx(0)*v_out_3.dx(0)*dx\
    -J3_out*(1/H_theta_out)**2*u_out_3.dx(1)*v_out_3.dx(1)*dx\
    -J4_out*(1/H_r_out)**2*u_out_4.dx(0)*v_out_4.dx(0)*dx\
    -J4_out*(1/H_theta_out)**2*u_out_4.dx(1)*v_out_4.dx(1)*dx\
    #нелинейные слагаемые (по порядку уравнений, 2 строчки ~ одно уравнение,
    #3-е уравнение системы на 4 строчках)
    F_nl_out = J1_out*(1/H_r_out)**2*u_out_1.dx(0)*u_out_2.dx(0)*v_out_1*dx\
    +J1_out*(1/H_theta_out)**2*u_out_1.dx(1)*u_out_2.dx(1)*v_out_1*dx\
    +J2_out*(1/H_r_out)**2*u_out_2.dx(0)*u_out_2.dx(0)*v_out_2*dx\
    +J2_out*(1/H_theta_out)*u_out_2.dx(1)*u_out_2.dx(1)*v_out_2*dx\
    -4*J3_out*(1/H_r_out)**2*u_out_3.dx(0)*u_out_1.dx(0)*v_out_3*dx\
    -4*J3_out*(1/H_theta_out)**2*u_out_3.dx(1)*u_out_1.dx(1)*v_out_3*dx\
    +3*J3_out*(1/H_r_out)**2*u_out_3.dx(0)*u_out_2.dx(0)*v_out_3*dx\
    +3*J3_out*(1/H_theta_out)**2*u_out_3.dx(1)*u_out_2.dx(1)*v_out_3*dx\
    +J4_out*(1/H_r_out)**2*u_out_1.dx(0)*u_out_1.dx(0)*v_out_4*dx\
    +J4_out*(1/H_theta_out)**2*u_out_1.dx(1)*u_out_1.dx(1)*v_out_4*dx\
    #фактор N/B
    FF = exp(u_out_2-2*u_out_1)
    #слагаемые, связанные с вращением
    F_rot_out = -0.5*J3_out*(1/H_r_out)**2*FF*FF*u_out_3.dx(0)*u_out_3.dx(0)*v_out_1*dx\
    -0.5*J3_out*FF*FF*((1/H_theta_out)**2)*u_out_3.dx(1)*u_out_3.dx(1)*v_out_1*dx\
    -0.75*J2_out*FF*FF*(1/H_r_out)**2*u_out_3.dx(0)*u_out_3.dx(0)*v_out_4*dx\
    -0.75*J2_out*FF*FF*u_out_3.dx(1)*((1/H_theta_out)**2)*u_out_3.dx(1)*v_out_4*dx\

    F_out = F0_out+F_nl_out+F_rot_out

    print('**********************************************')
    print('SOLVING IN OUTER DOMAIN. PLEASE WAIT')

    #решение вариационной задачи
    solve(F_out == 0, u_out, bcs_out)

    #формирование массивов искомых функций во внешней области. Решение считается
    #сферически симметричным, потому значение theta не имеет значения
    theta0 = np.pi/2

    result_solution_out=dir_name + 'e=' + str(e1)+ ','\
    + 'omega=' + str(Omega1) + '_res-out' + '.txt'

    G = open(result_solution_out, 'w')

    for i in range (1000):
        x1 = X1 + 0.001 * i * (X2 - X1)
        p = Point(x1,theta0)
        F0 = exp(u_out(p.x(), p.y())[0])
        F1 = exp(u_out(p.x(),p.y())[1])/F0
        F2 = (u_out(p.x(), p.y())[2])/Omega
        F3 = exp(u_out(p.x(),p.y())[3])/F0
        m=2*(pow(F1,0.5)-1)*R_int/(1-x1)
        J_rot=((R_int/(1-x1))**3)*F2*Omega/2
        data_x_out.append(R_int*r_g*10**(-5)/(1-x1))
        data_x.append(F3*R_int*r_g*10**(-5)/(1-x1))
        data_m.append(m)
        data_J.append(J_rot)
        data_N.append(F0)
        data_B.append(F1)
        data_Om.append(F2)
        data_A.append(F3)
        curv.append(0)
        G.write(str(R_int*F1*r_g*10**(-5)/(1-x1)))
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
        G.write('\n')

    G.close()

    graphic_module(data_x,data_x_2,data_N,data_N_2,data_B,data_B_2,
                        data_Om,data_Om_2,data_A,data_A_2,
                        data_h,data_h_2,data_m,data_J,data_vel,curv,curv_2, e1,Omega0)

    #вычисление производных искомых функций для выяснения сшивки решений
    #функциональные пространства для внешней и внутренней областей
    V1 = FunctionSpace(mesh, P1)
    V2 = FunctionSpace(mesh_int, P2)

    Q1 = FunctionSpace(mesh, 'DG', 1)
    Q2 = FunctionSpace(mesh_int, 'DG', 1)

    p=Point(R1,np.pi/2)
    p_int=Point(R1,np.pi/2)

    derivative=[]
    derivative_int=[]

    for i in range(4):
        u1 = project(u[i],V1)
        u1_int = project(u_int[i],V2)
        u1_dx = project(u1.dx(0), Q1) # u.dx(1) for y-derivative
        u1_int_dx = project(u1_int.dx(0), Q2) # u.dx(1) for y-derivative
        derivative.append(u1_dx(p.x(),p.y()))
        derivative_int.append(u1_int_dx(p_int.x(),p_int.y()))

    delta=0
    delta_rel=0

    for i in range(2):
        delta=(derivative[i]-derivative_int[i])**2+delta
        delta_rel=(1-derivative_int[i]/derivative[i])**2+delta_rel


    class My_domain(SubDomain):
        def inside(self,x,on_boundary):
            return x[0]<=x_core and x[1]>=0*np.pi/20

    class My_domain_1(SubDomain):
        def inside(self,x,on_boundary):
            return x[1]>=0*np.pi/20

    subdomain = MeshFunction('size_t',mesh,mesh.topology().dim(), 0)
    subdomain.set_all(0)
    my_domain = My_domain()
    my_domain.mark(subdomain,1 )

    subdomain_1 = MeshFunction('size_t',mesh,mesh.topology().dim(), 0)
    subdomain_1.set_all(0)
    my_domain_1 = My_domain_1()
    my_domain_1.mark(subdomain_1, 1)

    dx_sub = Measure('dx', domain=mesh, subdomain_id=1, subdomain_data=subdomain)

    dx1_sub = Measure('dx', domain=mesh, subdomain_id=1, subdomain_data=subdomain_1)

    P_2=Expression('pow(x[0]*sin(x[1]),3)*x[0]',degree=3)

    V00 = FunctionSpace(mesh, "Lagrange", 3)
    U00 = project(P_2, V00)

    J_ang=4*np.pi*assemble(f3*exp(3*u_2-4*u_1)*(Omega*key1-u_3)*U00*dx_sub)

    J_ang0=4*np.pi*assemble(f3*exp(3*u_2-4*u_1)*(Omega*key1-u_3)*U00*dx1_sub)

    J_ang0=4*np.pi*assemble(f3*exp(3*u_2-4*u_1)*(Omega*key1-u_3)*U00*dx)

    Mass_ADM=4*np.pi*assemble(f1*exp(u_2)*J1*dx)

    DM=4*np.pi*assemble(2*u_3*f3*exp(3*u_2-4*u_1)*(Omega*key1-u_3)*J3*dx)

    Mass_b=1.673*10**(-24)*4*np.pi*assemble(f6*exp(2*u_2-3*u_1+u_4)*J1*dx)*r_g**3*10**39/solar_mass

    GRV2_1=8*np.pi*assemble(f4*J4*dx)
    GRV2_2=-assemble((u_1.dx(0)**2+(u_1.dx(1)/H_theta)**2)*J4*dx)
    GRV2_2_int=-assemble(((u_int_1.dx(0))**2+(u_int_1.dx(1)/H_theta)**2)*J4*dx)
    GRV2_3=0.75*assemble((u_3.dx(0)**2+(u_3.dx(1)/H_theta)**2)*J3*exp(2*u_2-4*u_1)*dx)
    GRV2_3_int=0.75*assemble(((u_int_3.dx(0))**2+(u_int_3.dx(1)/H_theta)**2)*J3*FF_int*FF_int*dx)

    print("************************************************************")
    print("Производные на границе внутренней области =", derivative)
    print("Производные на границе внешней области =", derivative_int)
    print("квадратичное отклонение =", delta)
    print("относительное квадратичное отклонение =", delta_rel)
    print("GRV2=",(GRV2_1+GRV2_2+GRV2_2_int+GRV2_3+GRV2_3_int)*2)

    print("************************************************************")
    print("STAR PARAMETERS")
    print("gravitational mass =", Mass_ADM+DM)
    print("baryon mass =", Mass_b)
    print("frequency of rotation =", Omega0)
    print("central enthalpy =", h0)
    print("central energy density in MeV/fm^3 =", central_density*rho_0/(1.78E12))
    print("equatorial radius =", eq_radius)
    print("polar radius =",polar_radius)
    print("axis ratio =", polar_radius/eq_radius)
    print("circumferential equatorial radius =", circ_radius)
    print("compactness =", (Mass_ADM+DM)*r_g/((10**5)*circ_radius))
    print("velocity at the equator =", eq_velocity)
    print("************************************************************")
    print("MULTIPOLE COEFFICIENTS")
    print("J =", J_ang0)
    chi_0 = J_ang0/((Mass_ADM+DM)**2)
    print("chi =", chi_0)
    print("I (core) =", J_ang*1475**2*solar_mass*0.001/Omega)
    print("I (total) =", J_ang0*1475**2*solar_mass*0.001/Omega)
    I_0 = (J_ang0/Omega)/(Mass_ADM+DM)**3
    #мультипольные коэффициенты
    P_2 = Expression('1.5*pow(cos(x[1]),2)-0.5', degree=4) #полином Лежандра 2-го порядка

    DP_3 = Expression('7.5*pow(cos(x[1]),2)-1.5', degree=3) #производная полинома Лежандра 3-го порядка

    factor_1=Expression('(x[0])*sin(x[1])',degree=3)

    factor_1_out = R1*Expression('(1/(1-x[0]))*sin(x[1])',degree=3)

    factor_2 = Expression('x[0]*x[0]',degree=2)

    factor_2_out = R1**2*Expression('pow(1/(1-x[0]),2)',degree=3)

    factor_3_out = Expression('sin(x[1])',degree=3)

    #функции источников
    sigma_nu = 4*(np.pi)*f1\
    +0.5*factor_1**2*f0*u_3.dx(0)*u_3.dx(0)+0.5*factor_1**2*f0*((1/H_theta)**2)*u_3.dx(1)*u_3.dx(1)\
    -u_1.dx(0)*u_2.dx(0)-((1/H_theta)**2)*u_1.dx(1)*u_2.dx(1)

    sigma_nu_out = 0.5*factor_1_out**2*(1/H_r_out)**2*FF**2*u_out_3.dx(0)*u_out_3.dx(0)\
    +0.5*factor_1_out**2*FF**2*((1/H_theta_out)**2)*u_out_3.dx(1)*u_out_3.dx(1)\
    -(1/H_r_out)**2*u_out_1.dx(0)*u_out_2.dx(0)- ((1/H_theta_out)**2)*u_out_1.dx(1)*u_out_2.dx(1)

    sigma_b = 16*(np.pi)*f2*factor_1*exp(u_2)

    sigma_omega = -16*(np.pi)*f3*(Omega-u_3)*factor_1\
    +(4*u_3.dx(0)*u_1.dx(0)+4*((1/H_theta)**2)*u_3.dx(1)*u_1.dx(1))*factor_1\
    -(3*u_3.dx(0)*u_2.dx(0)+3*(1/H_theta)**2*u_3.dx(1)*u_2.dx(1))*factor_1\

    if Omega0==0:
        q_2 = 0
        q_2_out = 0
        b_0 = -0.25
        b_2 = 0
        omega_2 = 0

    if Omega0!=0:
        q_2 = (Mass_ADM+DM)**(-3)*assemble(sigma_nu*P_2*factor_2*J1*dx)
        q_2_out = (Mass_ADM+DM)**(-3)*assemble(sigma_nu_out*factor_2_out*P_2*J1_out*factor_3_out*dx)
        b_0 = (Mass_ADM+DM)**(-2)*np.sqrt(2/np.pi)*assemble(sigma_b*J1*dx)
        b_0 = -0.25
        omega_2 = (1/12)*(Mass_ADM+DM)**(-4)*assemble(sigma_omega*DP_3*factor_2*factor_1*J1*dx)

    M_2 = -(1/3)*(1+4*b_0+3*(q_2+q_2_out))*(Mass_ADM+DM)**(3)

    S_3 = -(3/10)*(2*chi_0+8*chi_0*b_0-5*omega_2)*(Mass_ADM+DM)**(4)

    if Omega0==0:
        m_2 = 0
        s_3 = 0
    if Omega0!=0:
        m_2 = - M_2 / ((Mass_ADM+DM)**(3)*chi_0**2)
        s_3 = - S_3 / ((Mass_ADM+DM)**(4)*chi_0**3)

    print("I =", I_0)

    print("M2 =", M_2)

    print("m2 = ", m_2)

    print("S3 =", S_3)

    print("s3 =", s_3)

    print("b0 =", b_0)

    print("q2 =", q_2)

    print("q2_out =", q_2_out)

    print("omega2 =", omega_2)

    output = {}

    result_solution_3='init_files/' + 'e=' + str(e1)+ ','\
    + 'omega=' + str(Omega1) + '_sum' + '.txt'

    with open(result_solution_3, "w") as init_file:
        output["gravitational mass"] = str(Mass_ADM+DM)
        output["baryon mass"] = str(Mass_b)
        output["mass"] = str(mass)
        output["m_r"] = str(moment_of_rotation)
        output["m_r2"] = str(moment_of_rotation2)
        output["omega"] = str(Omega0)
        output["enthalpy"] = str(h0)
        output["density"] = str(central_density*rho_0/(1.78E12))
        output["eq_rad"] = str(eq_radius)
        output["polar_rad"] = str(polar_radius)
        output["r/a"] = str(polar_radius/eq_radius)
        output["R_eq"] = str(circ_radius)
        output["comp"] = str((Mass_ADM+DM)*r_g/((10**5)*circ_radius))
        output["J_ang"] = str(J_ang0)
        output["Kerr"] = str(J_ang0/((Mass_ADM+DM)**2))
        output["moment of inertia"] = str(J_ang0*1475**2*solar_mass*0.001/Omega)
        output["radius of core"] = str(R_core)
        output["R1"] = str(R1)
        output["resolution"] = str(args[9])
        output["I"] = str(I_0)
        output["M2"] = str(m_2)
        output["S3"] = str(s_3)
        output["b0"] = str(b_0)
        output["q2"] = str(q_2 + q_2_out)
        output["omega_2"] = str(omega_2)

        json.dump(output, init_file, indent=4, separators=(',', ': '))

