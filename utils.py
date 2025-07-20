import os
from copy import deepcopy, copy

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from scipy.signal import find_peaks

from math import pi, sqrt, sin, cos, radians

from constants import PLOT_DATA_ROUND, DPI_VALIE, PLOT_DISPLAY_SIZE, PLOT_LEGEND_FONT_SIZE, PLOT_MARKET_SIZE,\
                      PLOT_LINE_WIDTH, PLOT_TITLE_FONT_SIZE, PLOT_ANOTATE_FONT_SIZE, PLOT_ASIX_FONT_SIZE, I, E


def get_request_data(request):
    return dict(request.json if request.is_json else (request.form.items() or {}))


def display_plot(arguments, labels, color_line, title, annotate_step, points_count, alpha=None, show=None, axis=None):
    show = show or range(len(arguments))
    alpha = alpha or [1 for i in range(len(arguments))]
    plt.figure(figsize=PLOT_DISPLAY_SIZE, dpi = DPI_VALIE)
    for i in show:
        if i == "":
            continue
        plt.plot(*np.round(arguments[i], PLOT_DATA_ROUND), color_line[i], label=labels[i], markersize=PLOT_MARKET_SIZE, linewidth=PLOT_LINE_WIDTH, alpha=alpha[i])
        plt.rc('legend', fontsize=PLOT_LEGEND_FONT_SIZE)
        if annotate_step[i]:
            for j in range(0, len(arguments[i][0])-(1 if len(arguments[i][0]) != points_count else 0), annotate_step[i]):
                mid_x, mid_y = (sum(arguments[i][0])/len(arguments[i][0])), (sum(arguments[i][1])/len(arguments[i][1]))
                scale_x, scale_y = (abs(max(arguments[i][0])) + abs(min(arguments[i][0])))/2, (abs(max(arguments[i][1])) + abs(min(arguments[i][1])))/2
                try:
                    if arguments[i][0][j] != 0:
                        x_add = abs(arguments[i][0][j])/arguments[i][0][j]/25*abs(mid_x-arguments[i][0][j])
                    else:
                        x_add = 0
                except:
                    x_add = 0
                try:
                    if arguments[i][1][j] != 0:
                        y_add = abs(arguments[i][1][j])/arguments[i][1][j]/25*abs(mid_y-arguments[i][1][j])
                    else:
                        y_add = 0
                except:
                    y_add = 0
                plt.annotate(j+1, (arguments[i][0][j] - 0.025 * scale_x + x_add, arguments[i][1][j] - 0.015 * scale_x + y_add), fontsize=PLOT_ANOTATE_FONT_SIZE)
                # plt.annotate(j+1, (arguments[i][0][j], arguments[i][1][j]), fontsize=plot_annotate_font_size)
    plt.suptitle(title, fontsize=PLOT_TITLE_FONT_SIZE)

    if axis:
        print(axis)
        if axis == 2:
            plt.xlim([-0.25, 2.25])
            plt.ylim([-1.25, 1.25])
            plt.grid()
        else:
            plt.xlim([-2.5, 2.5])
            plt.ylim([-1.5, 1.5])

    # plt.xticks([])
    # plt.yticks([])
    plt.tick_params(labelsize=PLOT_ASIX_FONT_SIZE)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    return plt

def vector_cords(M, X, Y):
    d = np.array([])

    for i in range(M):
        d = np.append(d, np.array([X[(i+1) % M]-X[i % M], Y[(i+1) % M]-Y[i % M]]))
    d = np.append(d, [d[0], d[1]])
    d = d.reshape(M+1, 2)
    return d

def vector_cords_not_loop(M, X, Y):
    d = np.array([])

    for i in range(M):
        d = np.append(d, np.array([X[i+1]-X[i], Y[i+1]-Y[i]]))
    d = d.reshape(M, 2)
    return d


def to_angle(x_a, y_a, x_b, y_b):
    sin_phi = (x_a * -y_b - y_a * -x_b) / (np.sqrt(x_a ** 2 + y_a ** 2) * np.sqrt(x_b ** 2 + y_b ** 2))
    cos_phi = (x_a * x_b + y_a * y_b) / (np.sqrt(x_a ** 2 + y_a ** 2) * np.sqrt(x_b ** 2 + y_b ** 2))
    if np.isnan(sin_phi) or np.isnan(cos_phi):
        return 0, 1
        print("ALARM!!!!")
    return sin_phi, cos_phi


def align_value_count(M, d, curve_type):
    # psis_sin = np.array([])
    psis = np.array([])

    if curve_type == "not_loop":
        M -= 1

    for i in range(M):
        psi = to_angle(d[i][0], d[i][1], d[i+1][0], d[i+1][1])
        if psi[0] > 0 and psi[1] > 0:
            # psis_sin = np.append(psis_sin, psi[0])
            psis = np.append(psis, np.arcsin(psi[0]))
        elif psi[0] > 0 and psi[1] < 0:
            # psis_sin = np.append(psis_sin, psi[0])
            psis = np.append(psis, pi - np.arcsin(psi[0]))
        elif psi[0] < 0 and psi[1] < 0:
            # psis_sin = np.append(psis_sin, psi[0])
            psis = np.append(psis, -pi - np.arcsin(psi[0]))
        elif psi[0] < 0 and psi[1] > 0:
            # psis_sin = np.append(psis_sin, psi[0])
            psis = np.append(psis, np.arcsin(psi[0]))
        elif round(psi[0], 5) == 0 and round(psi[1], 5) == 1:
            # psis_sin = np.append(psis_sin, pi/2)
            psis = np.append(psis, 0)
        elif round(psi[0], 5) == 0 and round(psi[1], 5) == -1:
            # psis_sin = np.append(psis_sin, -pi/2)
            psis = np.append(psis, pi)
            print("AAAAAAAAAA")
        elif round(psi[0], 5) == 1 and round(psi[1], 5) == 0:
            # psis_sin = np.append(psis_sin, pi)
            psis = np.append(psis, pi/2)
        elif round(psi[0], 5) == -1 and round(psi[1], 5) == 0:
            # psis_sin = np.append(psis_sin, pi)
            psis = np.append(psis, -pi/2)
        else:
            print(psi, "\nERROR!!!")

    return psis


def len_value_count(M, d):
    S = np.array([])
    for i in range(M):
        S = np.append(S, np.sqrt(d[i][0]**2 + d[i][1]**2))
    return S


def C_coef_value_count(M, S, C_proportion_coef):
    return (6/((sum(S)/M)**3))*C_proportion_coef


def C_coef_value_count(M, S):
    c_coef = np.array([])
    for i in range(M - 1):
        c_coef = np.append(c_coef, (S[i]/2 + S[i+1]/2))
    return c_coef


def matrix_coefs(M, S, psis, C, point_type, equation_type, P_align_coef=None, extra_psis=None, aligns=None):
    dims = 8 * M
    matrix = np.zeros((dims, dims))
    coefs = np.zeros((dims))

    if equation_type == "not_loop":
        for i in range(M):
            # Рівняння зв'язку
            matrix[i*8+2, i*8], matrix[i*8+2, i*8+1], \
            matrix[i*8+2, i*8+2], matrix[i*8+2, i*8+3] = \
                (1, S[i], S[i]**2/(2*I*E), S[i]**3/(6*I*E))

            matrix[i*8+3, i*8], matrix[i*8+3, i*8+1], \
            matrix[i*8+3, i*8+2], matrix[i*8+3, i*8+3] = \
                (0, 1, S[i]/(I*E), S[i]**2/(2*I*E))

            matrix[i*8+4, i*8], matrix[i*8+4, i*8+1], \
            matrix[i*8+4, i*8+2], matrix[i*8+4, i*8+3] = \
                (0, 0, 1, S[i])

            matrix[i*8+5, i*8], matrix[i*8+5, i*8+1], \
            matrix[i*8+5, i*8+2], matrix[i*8+5, i*8+3] = \
                (0, 0, 0, 1)

            matrix[i*8+2, i*8+4], matrix[i*8+3, i*8+5], matrix[i*8+4, i*8+6], matrix[i*8+5, i*8+7] = -1, -1, -1, -1

            if i < M - 1:
                matrix[i*8+6, i*8+4], matrix[i*8+7, i*8+5], matrix[i*8+8, i*8+6], matrix[i*8+9, i*8+7] = 1, 1, 1, 1
                matrix[i*8+6, (i+1)*8], matrix[i*8+7, (i+1)*8+1], matrix[i*8+8, (i+1)*8+2], matrix[i*8+9, (i+1)*8+3] = -1, -1, -1, -1
                if point_type[i] == 0:
                    matrix[i*8+9, i*8+8] = -C[i]
                elif point_type[i] == 1:
                    matrix[i*8+9, i*8+7] = 0
                    matrix[i*8+9, (i+1)*8+3] = 0
                    matrix[i*8+9, i*8+4] = 1
                coefs[i*8+7] = psis[i]

                if P_align_coef is not None and point_type[i] == 0:
                    coefs[i*8+9] = -C[i]*P_align_coef[i]

        # coefs[1] = (radians(aligns[0]) + aligns[1] * extra_psis[0])
        # coefs[-1] = (radians(aligns[2]) + aligns[3] * extra_psis[-1])
        # print(degrees(coefs[1]), degrees(coefs[-1]))

        matrix[0][0], matrix[1][2] = 1, 1
        matrix[-2][-4], matrix[-1][-2] = 1, 1
    else:
        for i in range(M):
            # Рівняння зв'язку
            matrix[i*8, i*8], matrix[i*8, i*8+1], \
            matrix[i*8, i*8+2], matrix[i*8, i*8+3] = \
                (1, S[i], S[i]**2/(2*I*E), S[i]**3/(6*I*E))

            matrix[i*8+1, i*8], matrix[i*8+1, i*8+1], \
            matrix[i*8+1, i*8+2], matrix[i*8+1, i*8+3] = \
                (0, 1, S[i]/(I*E), S[i]**2/(2*I*E))

            matrix[i*8+2, i*8], matrix[i*8+2, i*8+1], \
            matrix[i*8+2, i*8+2], matrix[i*8+2, i*8+3] = \
                (0, 0, 1, S[i])

            matrix[i*8+3, i*8], matrix[i*8+3, i*8+1], \
            matrix[i*8+3, i*8+2], matrix[i*8+3, i*8+3] = \
                (0, 0, 0, 1)

            matrix[i*8, i*8+4], matrix[i*8+1, i*8+5], matrix[i*8+2, i*8+6], matrix[i*8+3, i*8+7] = -1, -1, -1, -1

            if i < M - 1:
                matrix[i*8+4, i*8+4], matrix[i*8+5, i*8+5], matrix[i*8+6, i*8+6], matrix[i*8+7, i*8+7] = 1, 1, 1, 1
                matrix[i*8+4, (i+1)*8], matrix[i*8+5, (i+1)*8+1], matrix[i*8+6, (i+1)*8+2], matrix[i*8+7, (i+1)*8+3] = -1, -1, -1, -1
                if point_type[i] == 0:
                    matrix[i*8+7, i*8+8] = -C[i]
                elif point_type[i] == 1:
                    matrix[i*8+7, i*8+7] = 0
                    matrix[i*8+7, (i+1)*8+3] = 0
                    matrix[i*8+7, i*8+4] = 1

            else:
                matrix[i*8+4, i*8+4], matrix[i*8+5, i*8+5], matrix[i*8+6, i*8+6], matrix[i*8+7, i*8+7] = 1, 1, 1, 1
                matrix[i*8+4, 0], matrix[i*8+5, 1], matrix[i*8+6, 2], matrix[i*8+7, 3] = -1, -1, -1, -1
                if point_type[i] == 0:
                    matrix[i*8+7, 0] = -C[i]
                elif point_type[i] == 1:
                    matrix[i*8+7, i*8+7] = 0
                    matrix[i*8+7, 3] = 0
                    matrix[i*8+7, i*8+4] = 1


            coefs[i*8+5] = psis[i]
            if P_align_coef is not None and point_type[i] == 0:
                coefs[i*8+7] = -C[i]*P_align_coef[i]

    # display_table(matrix, bad_data = False, revert=True)

    return matrix, coefs


def len_calc(k, X, Y, x, y):
    return abs((y-Y)/(sqrt(1+k**2))-(k*(x-X))/(sqrt(1+k**2)))


def P_coef_count(M, d, X, Y, X_n, Y_n, equation_type):
    P_align_coef = []
    P_align_coef_new = []

    for i in range(M):
        position = (X_n[i+1] - X_n[i]) * (Y[i+1] - Y_n[i]) - (Y_n[i+1] - Y_n[i]) * (X[i+1] - X[i])
        if position < 0:
            # sign = -1
            sign = 1
        elif position > 0:
            # sign = 1
            sign = -1
        else:
            print("000000")
            sign = 0

        dist = sign * np.sqrt((X_n[i+1] - X[i+1]) ** 2 + (Y_n[i+1] - Y[i+1]) ** 2)
        P_align_coef_new.append(dist)

    if equation_type == "not_loop":
        M -= 1

    if M != 1:
        for i in range(M):
            if str(np.arcsin(to_angle(d[i][0], d[i][1], X[i+1]-X_n[i+1], Y[i+1]-Y_n[i+1])[0])) == 'nan':
                print()

            psi_0 = np.sign(to_angle(d[i][0], d[i][1], X[i+1]-X_n[i+1], Y[i+1]-Y_n[i+1])[0])
            psi_1 = np.sign(to_angle(d[i+1][0], d[i+1][1], X[i+1]-X_n[i+1], Y[i+1]-Y_n[i+1])[0])
            k_0 = (Y_n[(i+1)]-Y_n[i])/(X_n[(i+1)]-X_n[i])
            k_1 = (Y_n[(i+2)]-Y_n[(i+1)])/(X_n[(i+2)]-X_n[(i+1)])
            len_0 = len_calc(k_0, X_n[i+1], Y_n[i+1], X[i+1], Y[i+1])
            len_1 = len_calc(k_1, X_n[i+1], Y_n[i+1], X[i+1], Y[i+1])
            P_align_coef.append(psi_0 * len_0 if len_0 < len_1 else psi_1 * len_1)
            # print(P_align_coef_new[i], P_align_coef[i])
    else:
        P_align_coef = None
    return P_align_coef


def vector_normalization(d, S, solution, curve_type):
    a_norm = []
    b_norm = []
    c_l_norm = []
    d_l_norm = []
    c_n_norm = []
    d_n_norm = []

    for i in range(len(S)):
        a_norm.append(d[i][0]/S[i])
        b_norm.append(d[i][1]/S[i])

    if curve_type == "loop":
        d = d[:-1]

    for i in range(len(d)):
        matrix_rotate = [[cos(-pi/2), -sin(-pi/2)], [sin(-pi/2), cos(-pi/2)]]
        vektors = (np.dot(matrix_rotate, [a_norm[i], b_norm[i]]))
        c_l_norm.append(vektors[0]), d_l_norm.append(vektors[1])

        align = solution[8*i+1]
        matrix_rotate = [[cos(-pi/2-align), -sin(-pi/2-align)], [sin(-pi/2-align), cos(-pi/2-align)]]
        vektors = (np.dot(matrix_rotate, [a_norm[i], b_norm[i]]))
        c_n_norm.append(vektors[0]), d_n_norm.append(vektors[1])

    align = solution[-3]
    matrix_rotate = [[cos(-pi/2-align), -sin(-pi/2-align)], [sin(-pi/2-align), cos(-pi/2-align)]]
    vektors = (np.dot(matrix_rotate, [a_norm[-1], b_norm[-1]]))
    c_n_norm.append(vektors[0]), d_n_norm.append(vektors[1])

    return a_norm, b_norm, c_l_norm, d_l_norm, c_n_norm, d_n_norm


def midle_point_params_vector(M, S, solution, list_of_patrs, psis):
    dims = 4
    matrix = np.zeros((dims, dims))
    vector = np.zeros((dims))
    sol_half = []

    for i in range(M):
        for k in list_of_patrs:
            s = S[i]*k

            matrix[0, 0], matrix[0, 1], \
            matrix[0, 2], matrix[0, 3] = \
                (1, s, s**2/(2*I*E), s**3/(6*I*E))

            matrix[1, 0], matrix[1, 1], \
            matrix[1, 2], matrix[1, 3] = \
                (0, 1, s/(I*E), s**2/(2*I*E))

            matrix[2, 0], matrix[2, 1], \
            matrix[2, 2], matrix[2, 3] = \
                (0, 0, 1, s)

            matrix[3, 0], matrix[3, 1], \
            matrix[3, 2], matrix[3, 3] = \
                (0, 0, 0, 1)

            vector[0], vector[1], vector[2], vector[3] = solution[i*8], solution[i*8+1], solution[i*8+2], solution[i*8+3]
            sol_half.append(np.dot(matrix, vector))
    return sol_half


def midle_point_count(M, list_of_patrs, X, Y, S, a_norm, b_norm, sol_half):
    B_j = []
    c_n_norm_B_j=[]
    d_n_norm_B_j=[]

    for i in range(M):
        for k in range(len(list_of_patrs)):
            B_j.append([X[i] + S[i] * list_of_patrs[k] * a_norm[i], Y[i] + S[i] * list_of_patrs[k] * b_norm[i]])

            index = len(list_of_patrs)*i+k
            matrix_rotate = [[cos(-pi/2-sol_half[index][1]), -sin(-pi/2-sol_half[index][1])],
                             [sin(-pi/2-sol_half[index][1]), cos(-pi/2-sol_half[index][1])]]
            vektors = (np.dot(matrix_rotate, [a_norm[i], b_norm[i]]))
            c_n_norm_B_j.append(vektors[0]), d_n_norm_B_j.append(vektors[1])

    return B_j, c_n_norm_B_j, d_n_norm_B_j


def new_position_count(M, S, X, Y, solution, c_l_norm, c_n_norm, c_n_norm_j, d_l_norm, d_n_norm, d_n_norm_j, sol_half, list_of_patrs, B_j, curve_type):
    M_j = []
    M_j_coreg = []
    D_j = []
    D_j_coreg = []
    X_disp = []
    Y_disp = []
    X__disp = []
    Y__disp = []
    znam = []

    for i in range(M):
        M_j.append([sum(S[:i]), solution[8*i+2]])
        X_ = 1 + solution[8*i+1]*sin(solution[8*i+1]) + solution[8*i]*cos(solution[8*i+1])*solution[8*i+2]
        Y_ = solution[8*i+1]*cos(solution[8*i+1]) - solution[8*i]*sin(solution[8*i+1])*solution[8*i+2]
        X__ = (solution[8*i+2]*sin(solution[8*i+1]) + 2*solution[8*i+1]*cos(solution[8*i+1])*solution[8*i+2]
               - solution[8*i]*sin(solution[8*i+1])*(solution[8*i+2]**2) + solution[8*i]*cos(solution[8*i+1])*solution[8*i+3])
        Y__ = (solution[8*i+2]*cos(solution[8*i+1]) - 2*solution[8*i+1]*sin(solution[8*i+1])*solution[8*i+2]
               - solution[8*i]*cos(solution[8*i+1])*(solution[8*i+2]**2) - solution[8*i]*sin(solution[8*i+1])*solution[8*i+3])

        X_disp.append(X_)
        Y_disp.append(Y_)
        X__disp.append(X__)
        Y__disp.append(Y__)
        znam.append(((sqrt(X_**2 + Y_**2))**3))

        D_j.append([X[i] + solution[8 * i] * c_l_norm[i], Y[i] + solution[8 * i] * d_l_norm[i]])

        if list_of_patrs:
            M_j_coreg.append([sum(S[:i]), (-(X__*Y_ - Y__*X_))/((sqrt(X_**2 + Y_**2))**3), S[i]*list_of_patrs[0]])
            D_j_coreg.append([X[i] + solution[8*i] * c_n_norm[i], Y[i] + solution[8*i] * d_n_norm[i]])
            for k in range(len(list_of_patrs)):
                index = len(list_of_patrs)*i+k
                M_j.append([M_j[i*len(list_of_patrs)+k+i][0]+S[i]*list_of_patrs[0], sol_half[index][2]])
                X_ = 1 + sol_half[index][1]*sin(sol_half[index][1]) + sol_half[index][0]*cos(sol_half[index][1])*sol_half[index][2]
                Y_ = sol_half[index][1]*cos(sol_half[index][1]) - sol_half[index][0]*sin(sol_half[index][1])*sol_half[index][2]
                X__ = (sol_half[index][2]*sin(sol_half[index][1]) + 2*sol_half[index][1]*cos(sol_half[index][1])*sol_half[index][2]
                       - sol_half[index][0]*sin(sol_half[index][1])*(sol_half[index][2]**2) + sol_half[index][0]*cos(sol_half[index][1])*sol_half[index][3])
                Y__ = (sol_half[index][2]*cos(sol_half[index][1]) - 2*sol_half[index][1]*sin(sol_half[index][1])*sol_half[index][2]
                       - sol_half[index][0]*cos(sol_half[index][1])*(sol_half[index][2]**2) - sol_half[index][0]*sin(sol_half[index][1])*sol_half[index][3])

                X_disp.append(X_)
                Y_disp.append(Y_)
                X__disp.append(X__)
                Y__disp.append(Y__)
                znam.append(((sqrt(X_**2 + Y_**2))**3))

                M_j_coreg.append([M_j[i*len(list_of_patrs)+k+i][0]+S[i]*list_of_patrs[0], (-(X__*Y_ - Y__*X_))/((sqrt(X_**2 + Y_**2))**3), S[i]*list_of_patrs[0]])
                D_j.append([B_j[index][0] + sol_half[index][0] * c_l_norm[i], B_j[index][1] + sol_half[index][0] * d_l_norm[i]])
                D_j_coreg.append([B_j[index][0] + sol_half[index][0] * c_n_norm_j[index], B_j[index][1] + sol_half[index][0] * d_n_norm_j[index]])
    M_j.append([sum(S), solution[-2]])
    X_ = 1 + solution[-3]*sin(solution[-3]) + solution[-4]*cos(solution[-3])*solution[-2]
    Y_ = solution[-3]*cos(solution[-3]) - solution[-4]*sin(solution[-3])*solution[-2]
    X__ = (solution[-2]*sin(solution[-3]) + 2*solution[-3]*cos(solution[-3])*solution[-2]
           - solution[-4]*sin(solution[-3])*(solution[-2]**2) + solution[-4]*cos(solution[-3])*solution[-1])
    Y__ = (solution[-2]*cos(solution[-3]) - 2*solution[-3]*sin(solution[-3])*solution[-2]
           - solution[-4]*cos(solution[-3])*(solution[-2]**2) - solution[-4]*sin(solution[-3])*solution[-1])

    X_disp.append(X_)
    Y_disp.append(Y_)
    X__disp.append(X__)
    Y__disp.append(Y__)
    znam.append(((sqrt(X_**2 + Y_**2))**3))

    view_solution = solution.reshape(len(solution) // 8, 8).transpose()
    disp_sol_half = np.array(sol_half).transpose()

    points_in_parts = disp_sol_half.shape[1] // view_solution.shape[1]

    cutted_solution = view_solution[:4, :]
    sol_half_parts = np.array_split(disp_sol_half, view_solution.shape[1], axis=1)

    combined = [np.concatenate((cutted_solution[:, i:i + 1], sol_half_parts[i]), axis=1) for i in range(view_solution.shape[1])]
    full_solution = np.concatenate(combined, axis=1)

    # display_plot_plotly(
    #     [
    #         [
    #             np.array([[i, X_disp[i]] for i in range(len(X_disp))]).transpose(),
    #             "markers+lines", "X'", "#E60000", {}, True
    #         ],
    #         [
    #             np.array([[i, Y_disp[i]] for i in range(len(Y_disp))]).transpose(),
    #             "markers+lines", "Y'", "#E6B400", {}, True
    #         ],
    #         [
    #             np.array([[i, X__disp[i]] for i in range(len(X__disp))]).transpose(),
    #             "markers+lines", "X''", "#288E45", {}, True
    #         ],
    #         [
    #             np.array([[i, Y__disp[i]] for i in range(len(Y__disp))]).transpose(),
    #             "markers+lines", "Y''", "#0014E6", {}, True
    #         ],
    #     ],
    #     filename=f"smooth_contour/Derivatives"
    # )

    # display_plot_plotly(
    #     [
    #         [
    #             np.array([[i, full_solution[0][i]] for i in range(len(full_solution[0]))]).transpose(),
    #             "markers+lines", "W full", "#B22222", {}, True
    #         ],
    #         [
    #             np.array([[(points_in_parts + 1) * i, solution[::8][i]] for i in range(len(solution[::8]))]).transpose(),
    #             "markers+lines", "W", "#FF0000", {}, True
    #         ],
    #         [
    #             np.array([[i, full_solution[1][i]] for i in range(len(full_solution[1]))]).transpose(),
    #             "markers+lines", "Aligns full", "#CC8400", {}, True
    #         ],
    #         [
    #             np.array([[(points_in_parts + 1) * i, solution[1::8][i]] for i in range(len(solution[1::8]))]).transpose(),
    #             "markers+lines", "Aligns", "#FFA500", {}, True
    #         ],
    #         [
    #             np.array([[i, full_solution[2][i]] for i in range(len(full_solution[2]))]).transpose(),
    #             "markers+lines", "M full", "#005500", {}, True
    #         ],
    #         [
    #             np.array([[(points_in_parts + 1) * i, solution[2::8][i]] for i in range(len(solution[2::8]))]).transpose(),
    #             "markers+lines", "M", "#008000", {}, True
    #         ],
    #         [
    #             np.array([[i, full_solution[3][i]] for i in range(len(full_solution[3]))]).transpose(),
    #             "markers+lines", "Q full", "#00008B", {}, True
    #         ],
    #         [
    #             np.array([[(points_in_parts + 1) * i, solution[3::8][i]] for i in range(len(solution[3::8]))]).transpose(),
    #             "markers+lines", "Q", "#0000FF", {}, True
    #         ],
    #     ],
    #     filename=f"smooth_contour/Solution"
    # )

    # display_plot_plotly(
    #     [
    #         [
    #             np.array([[i, full_solution[1][i]] for i in range(len(full_solution[1]))]).transpose(),
    #             "markers+lines", "Aligns (solution + sol_half)", "#FF0000", {}, True
    #         ],
    #         [
    #             np.array([[(points_in_parts + 1) * i, solution[1::8][i]] for i in
    #                       range(len(solution[1::8]))]).transpose(),
    #             "markers+lines", "Aligns (solution each 8)", "#FFA500", {}, True
    #         ],
    #         [
    #             np.array([[i * (points_in_parts // 2 + 1), solution[1::4][i]] for i in
    #                       range(len(solution[1::4]))]).transpose(),
    #             "markers+lines", "Aligns (solution each 4)", "#008000", {}, True
    #         ],
    #         [
    #             np.array([[i + i // points_in_parts + 1, disp_sol_half[1][i]] for i in range(len(disp_sol_half[1]))]).transpose(),
    #             "markers+lines", "Aligns (sol_half)", "#0000FF", {}, True
    #         ],
    #     ],
    #     filename=f"smooth_contour/Aligns"
    # )

    D_j.append([X[-1] + solution[-4] * c_l_norm[-1], Y[-1] + solution[-4] * d_l_norm[-1]])
    if list_of_patrs:
        M_j_coreg.append([sum(S), (-(X__*Y_ - Y__*X_))/((sqrt(X_**2 + Y_**2))**3), S[-1]*list_of_patrs[0]])
        D_j_coreg.append([X[-1] + solution[-4] * c_n_norm[-1], Y[-1] + solution[-4] * d_n_norm[-1]])

    # display_plot_plotly(
    #     [
    #         [
    #             np.array([[i, M_j[i][1]] for i in range(len(M_j))]).transpose(),
    #             "markers+lines", "M_j", "#E60000", {}, True
    #         ],
    #         [
    #             np.array([[i, M_j_coreg[i][1]] for i in range(len(M_j_coreg))]).transpose(),
    #             "markers+lines", "M_j_coreg", "#0014E6", {}, True
    #         ],
    #     ],
    #     filename=f"smooth_contour/Moments"
    # )

    return map(np.transpose, map(np.array, [M_j, M_j_coreg, D_j, D_j_coreg]))


def find_all_local_maxima(arr):
    maxima = []
    for i in range(0, len(arr) - 1):  # Не берем первый и последний элементы
        if i == 0:
            if arr[i] > arr[i - 2] and arr[i] > arr[i + 1]:
                maxima.append(arr[i])
            continue
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            maxima.append(arr[i])  # Сохраняем индекс локального максимума
    return np.array(maxima)


def find_all_local_minima(arr):
    minima = []
    for i in range(0, len(arr) - 1):  # Не берем первый и последний элементы
        if i == 0:
            if arr[i] < arr[i - 2] and arr[i] < arr[i + 1]:
                minima.append(arr[i])
            continue
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            minima.append(arr[i])  # Сохраняем индекс локального максимума
    return np.array(minima)


def find_local_maxima(arr):
    candidates = []
    min_distance = 3500

    # Находим все локальные максимумы
    for i in range(0, len(arr) - 1):
        if i == 0:
            if arr[i] > arr[i - 2] and arr[i] > arr[i + 1]:
                candidates.append(i)
            continue

        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            candidates.append(i)

    # Отбираем пики с учётом минимального расстояния
    selected = []
    for idx in sorted(candidates, key=lambda i: arr[i], reverse=True):  # сортируем по высоте
        if all(abs(idx - prev) >= min_distance for prev in selected):
            selected.append(idx)

    return np.array(selected)


def find_local_minima(arr):
    candidates = []
    min_distance = 3500

    # Находим все локальные максимумы
    for i in range(0, len(arr) - 1):
        if i == 0:
            if arr[i] < arr[i - 2] and arr[i] < arr[i + 1]:
                candidates.append(i)
            continue

        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            candidates.append(i)

    # Отбираем пики с учётом минимального расстояния
    selected = []
    for idx in sorted(candidates, key=lambda i: arr[i], reverse=False):  # сортируем по высоте
        if all(abs(idx - prev) >= min_distance for prev in selected):
            selected.append(idx)

    return np.array(selected)


def sort_points_clockwise_starting_from_first(pts):
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_indices = np.argsort(-angles)

    # Сдвигаем массив так, чтобы первая точка осталась первой
    first_index_in_sorted = np.where(sorted_indices == 0)[0][0]
    sorted_indices = np.roll(sorted_indices, -first_index_in_sorted)

    return pts[sorted_indices]

def max_area_quad(points):
    N = len(points)

    if N == 12:
        index_groups = [[i for i in range(0, N, 3)]]
    elif N == 10:
        index_groups = [
            [0, 3, 6, 7],
            [0, 3, 4, 7],
            [0, 1, 4, 7],
            [0, 3, 6, 9],
        ]
    elif N == 8:
        index_groups = [
            [0, 3, 6, 7],
            [0, 3, 4, 5],
            [0, 1, 2, 5],
            [0, 1, 4, 7],
        ]
    else:
        raise ValueError("Unsupported number of points")

    max_area = 0
    best_group = None
    for group in index_groups:
        polygon = Polygon([points[i] for i in group])
        area = polygon.area
        if area > max_area:
            max_area = area
            best_group = group

    return np.array([points[i] for i in best_group])

def find_max_area_quadrilateral(points):
    sorted_pts = sort_points_clockwise_starting_from_first(points)
    quad = max_area_quad(sorted_pts)

    return quad


def display_plot_plotly(data, equal=False, filename=None, background_image=None):
    import sys
    import os

    sys.path.append(os.path.abspath("../plots_storage"))
    from plots import display_plot

    title = filename.split("/")[-1]

    display_plot(data, filename=filename, html=True, s_json=True, title=title, equal=equal, background_image=background_image)
    # fig = go.Figure()
    #
    # for d in data:
    #     plot, markers, label, color, style = d
    #     fig.add_trace(
    #         go.Scatter(
    #             x=plot[0],
    #             y=plot[1],
    #             mode=markers,
    #             name=label,
    #             showlegend=bool(label),
    #             marker_color=color,
    #             text=[str(i + 1) for i in range(len(plot[0]))],
    #             textposition='bottom center',  # or 'bottom right', etc.
    #             line= style["line"] if "line" in style else None,  # thinner line, dashed style
    #             marker=dict(size=10),
    #         )
    #     )
    #
    # if filename:
    #     fig.write_image(f"plots/{filename}.png", width=1500, height=1500, scale=1)
    # else:
    #     fig.show()
    #
    # del fig


# def find_near_point(x, y, x_new, y_new, D_j_coreg):
#     x_new_array = np.array([])
#     y_new_array = np.array([])
#
#     old_points = np.empty((0, 2))
#     replace_points = np.empty((0, 2))
#     indexes = np.array([], dtype=int)
#
#     for index in range(len(x)):
#         current_point = [x[index], y[index]]
#         mirror_point = [x_new[index], y_new[index]]
#
#         position_index = np.where(D_j_coreg.transpose() == mirror_point)[0][0]
#
#         start_index, end_index = position_index - 200, position_index + 200
#         if start_index < 0:
#             interval_points = np.hstack([D_j_coreg[:, start_index - 1:], D_j_coreg[:, :end_index + 1]])
#         elif end_index > len(D_j_coreg[0]):
#             interval_points = np.hstack([D_j_coreg[:, :end_index - len(D_j_coreg[0]) + 1], D_j_coreg[:, start_index:]])
#         else:
#             interval_points = D_j_coreg[:, start_index:end_index + 1]
#
#         # new_points = np.array([interval_points[0][5], interval_points[1][5]])
#         # distance = np.sqrt((current_point[0] - new_points[0]) ** 2 + (current_point[1] - new_points[1]) ** 2)
#         new_points = np.array([])
#
#         if interval_points[:, 200] in np.vstack((x_new_array, y_new_array)).transpose():
#             distance = 1000
#         else:
#             distance = np.sqrt((current_point[0] - interval_points[0][5]) ** 2 + (current_point[1] - interval_points[1][5]) ** 2)
#
#         for i_point in interval_points.transpose():
#             local_distance = np.sqrt((current_point[0] - i_point[0]) ** 2 + (current_point[1] - i_point[1]) ** 2)
#             if local_distance < distance and i_point not in np.vstack((x_new_array, y_new_array)).transpose():
#                 distance = local_distance
#                 new_points = i_point
#         if len(new_points):
#             x_new_array = np.append(x_new_array, new_points[0])
#             y_new_array = np.append(y_new_array, new_points[1])
#             indexes = np.append(indexes, index)
#             old_points = np.vstack((old_points, mirror_point))
#             replace_points = np.vstack((replace_points, new_points))
#         else:
#             x_new_array = np.append(x_new_array, mirror_point[0])
#             y_new_array = np.append(y_new_array, mirror_point[1])
#
#     return x_new_array, y_new_array, indexes, old_points, replace_points

def find_near_point(x, y, x_new, y_new, D_j_coreg):
    current_points = np.column_stack((x, y))
    mirror_points = np.column_stack((x_new, y_new))
    D_j_coreg_T = D_j_coreg.T
    total_len = D_j_coreg.shape[1]
    used_points = set()
    search_radius = 200

    x_new_array = []
    y_new_array = []

    old_points = []
    replace_points = []
    changed_indexes = []

    position_indices = [np.where((D_j_coreg_T == point).all(axis=1))[0][0] for point in mirror_points]

    x_new_array.append(mirror_points[0][0])
    y_new_array.append(mirror_points[0][1])
    used_points.add(tuple(D_j_coreg_T[0]))
    used_points.add(tuple(D_j_coreg_T[-1]))

    for idx, point_index in enumerate(position_indices[1: -1], 1):
        start = 0 if point_index - search_radius < 0 else point_index - search_radius
        end = total_len if point_index + search_radius > total_len else point_index + search_radius
        indices_range = np.arange(start, end)
        candidate_points = D_j_coreg_T[indices_range]

        mask = [tuple(p) not in used_points for p in candidate_points]
        candidate_points = candidate_points[mask]

        distances = np.linalg.norm(candidate_points - current_points[idx], axis=1)
        chosen_point = candidate_points[np.argmin(distances)]

        if not np.array_equal(chosen_point, mirror_points[idx]):
            changed_indexes.append(idx)
            old_points.append(mirror_points[idx])
            replace_points.append(chosen_point)

        used_points.add(tuple(chosen_point))
        x_new_array.append(chosen_point[0])
        y_new_array.append(chosen_point[1])

    x_new_array.append(D_j_coreg_T[-1][0])
    y_new_array.append(D_j_coreg_T[-1][1])

    ressss = np.column_stack((x_new_array, y_new_array))
    unique_ressss = np.unique(ressss, axis=0)

    if ressss.shape[0] != unique_ressss.shape[0]:
        print("AAALLLAAARRRMMM!!!!")

        _, idx, counts = np.unique(ressss, return_index=True, return_counts=True, axis=0)
        duplicate_values = ressss[np.sort(idx[counts > 1])]
        print("Дублирующиеся значения:", duplicate_values)
        duplicate_indices = [i for i, val in enumerate(ressss) if val in duplicate_values]
        print("Индексы дублирующихся элементов:", duplicate_indices)

    return (
        np.array(x_new_array),
        np.array(y_new_array),
        np.array(changed_indexes),
        np.array(old_points),
        np.array(replace_points)
    )


# def find_near_point(x, y, x_new, y_new, D_j_coreg):
#     x_new_array = []
#     y_new_array = []
#
#     old_points = []
#     replace_points = []
#     indexes = []
#
#     D_j_coreg_T = D_j_coreg.T  # Shape: (N, 2)
#     used_points = set()
#
#     for index in range(len(x) - 1):
#         current_point = np.array([x[index], y[index]])
#         mirror_point = np.array([x_new[index], y_new[index]])
#
#         # Find index of mirror_point in D_j_coreg_T (avoid full np.where comparison)
#         # Use np.all to find rows that match the mirror_point
#         match = np.where(np.all(D_j_coreg_T == mirror_point, axis=1))[0]
#         if match.size == 0:
#             x_new_array.append(mirror_point[0])
#             y_new_array.append(mirror_point[1])
#             continue
#         position_index = match[0]
#
#         total_len = D_j_coreg.shape[1]
#         start_index = (position_index - 2000) % total_len
#         end_index = (position_index + 2000) % total_len
#
#         if start_index <= end_index:
#             interval_points = D_j_coreg[:, start_index:end_index + 1]
#         else:
#             interval_points = np.hstack((D_j_coreg[:, start_index:], D_j_coreg[:, :end_index + 1]))
#
#         # Vectorized distance computation
#         candidate_points = interval_points.T
#         mask = [tuple(p) not in used_points for p in candidate_points]
#         candidate_points = candidate_points[mask]
#
#         if candidate_points.shape[0] == 0:
#             chosen_point = mirror_point
#         else:
#             distances = np.linalg.norm(candidate_points - current_point, axis=1)
#             min_idx = np.argmin(distances)
#             chosen_point = candidate_points[min_idx]
#
#         if not np.array_equal(chosen_point, mirror_point):
#             indexes.append(index)
#             old_points.append(mirror_point)
#             replace_points.append(chosen_point)
#
#         used_points.add(tuple(chosen_point))
#         x_new_array.append(chosen_point[0])
#         y_new_array.append(chosen_point[1])
#
#     return (
#         np.array(x_new_array),
#         np.array(y_new_array),
#         np.array(indexes),
#         np.array(old_points),
#         np.array(replace_points)
#     )



# def order_points(b_x, b_y, x, y):
#     big = np.column_stack((b_x, b_y))
#     small = np.column_stack((x, y))
#
#     # Find the indices in big_contour that match small_contour points
#     indices = [np.where((big == point).all(axis=1))[0][0] for point in small]
#
#     # Sort the small contour by the order of points in big_contour
#     sorted_small_contour = small[np.argsort(indices)]
#
#     sorted_small_contour[-1] = sorted_small_contour[0]
#
#     return sorted_small_contour[:, 0], sorted_small_contour[:, 1]

def order_points(b_x, b_y, x, y, x_inp, y_inp):
    big = np.column_stack((b_x, b_y))
    small = np.column_stack((x, y))
    small_inp = np.column_stack((x_inp, y_inp))

    # Находим индексы точек small в big
    indices = [np.where((big == point).all(axis=1))[0][0] for point in small]

    # Сортируем small по индексам в big
    sorted_indices = np.argsort(indices)
    sorted_small = small[sorted_indices]
    sorted_small_inp = small_inp[sorted_indices]
    # sorted_big_indices = np.array(indices)[sorted_indices]

    # # Определим, с какого индекса начинать (по позиции первой точки small)
    # start_point = small[0]
    # start_index = np.where((sorted_small == start_point).all(axis=1))[0][0]
    #
    # # Циклический сдвиг
    # rotated_small = np.roll(sorted_small, -start_index, axis=0)
    # rotated_small_inp = np.roll(sorted_small_inp, -start_index, axis=0)

    # rotated_small = np.append(rotated_small, [rotated_small[0]], axis=0)
    # rotated_small_inp = np.append(rotated_small_inp, [rotated_small_inp[0]], axis=0)

    return sorted_small[:, 0], sorted_small[:, 1], sorted_small_inp[:, 0], sorted_small_inp[:, 1]


# def ccw(a, b, c):
#     return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
#
# def segments_intersect(a, b, c, d):
#     return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)
#
# def order_points(x, y):
#     points = np.vstack([x, y])
#     for i in range(points.shape[1]):
#         j = 0
#         while j < 40:
#             if segments_intersect(
#                 points[:, i],
#                 points[:, (i + 1) % (points.shape[1])],
#                 points[:, (i + j + 2) % (points.shape[1])],
#                 points[:, (i + j +  3) % (points.shape[1])]
#             ):
#                 points[:, [(i + 1) % (points.shape[1]), (i + j + 2) % (points.shape[1])]] = points[:, [(i + j + 2) % (points.shape[1]), (i + 1) % (points.shape[1])]]
#                 j = 0
#             else:
#                 j += 1
#
#     return points[0], points[1]


def is_convex_quad(points, indices):
    quad = points[indices]

    # Убедимся, что 4 уникальные точки
    if len(np.unique(quad, axis=0)) != 4:
        return False

    # Вычисляем векторные повороты (z-компонент векторного произведения)
    def cross_z(a, b, c):
        ab = b - a
        bc = c - b
        return ab[0] * bc[1] - ab[1] * bc[0]

    signs = []
    for i in range(4):
        a = quad[i]
        b = quad[(i + 1) % 4]
        c = quad[(i + 2) % 4]
        cross = cross_z(a, b, c)
        signs.append(np.sign(cross))

    # Все повороты одного знака → выпукло
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)

def is_convex_quad(points, indices):
    quad = points[indices]
    if len(np.unique(quad, axis=0)) != 4:
        return False

    def cross_z(a, b, c):
        ab = b - a
        bc = c - b
        return ab[0] * bc[1] - ab[1] * bc[0]

    signs = []
    for i in range(4):
        a = quad[i]
        b = quad[(i + 1) % 4]
        c = quad[(i + 2) % 4]
        signs.append(np.sign(cross_z(a, b, c)))

    return all(s > 0 for s in signs) or all(s < 0 for s in signs)

def polygon_area(quad):
    x = quad[:, 0]
    y = quad[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def delete_near_points(arr, candidates, order):
    min_distance = 3500
    selected = []
    for idx in sorted(candidates, key=lambda i: arr[i], reverse=order):  # сортируем по высоте
        if all(abs(idx - prev) >= min_distance for prev in selected):
            selected.append(idx)

    return selected


def get_corner_points_candidate(M_j, D_j_coreg, direction, general_l, puzzle_index):
    arr = M_j[1]
    arr = np.insert(arr, 0, M_j[1][-2])
    max_val = np.max(arr)
    min_val = np.min(arr)

    if not direction:
        high_peaks, _ = find_peaks(arr)
        high_peaks = [i-1 for i in high_peaks if arr[i] > 0.5 * max_val]
        high_peaks = delete_near_points(arr, high_peaks, True)

        low_peaks, _ = find_peaks(-arr)
        low_peaks = [i-1 for i in low_peaks if arr[i] < 0.5 * min_val]
        low_peaks = delete_near_points(arr, low_peaks, False)
    else:
        high_peaks, _ = find_peaks(-arr)
        high_peaks = [i-1 for i in high_peaks if arr[i] < 0.5 * min_val]
        high_peaks = delete_near_points(arr, high_peaks, False)

        low_peaks, _ = find_peaks(arr)
        low_peaks = [i-1 for i in low_peaks if arr[i] > 0.5 * max_val]
        low_peaks = delete_near_points(arr, low_peaks, True)

    # sort_low_peaks = sorted(low_peaks)
    # n = len(arr)
    # cyclic_peaks = sort_low_peaks + [low_peaks[0] + n]
    # min_distance = n
    # for i in range(len(sort_low_peaks)):
    #     idx1 = cyclic_peaks[i]
    #     idx2 = cyclic_peaks[i + 1]
    #     dist = (idx2 - idx1)
    #     if dist < min_distance:
    #         min_distance = dist
    #         start_index = i % 2

    all_extrema = sorted(set(high_peaks + low_peaks))
    n = len(all_extrema)
    points = D_j_coreg[:, all_extrema].T
    valid_quads = []

    for i in range(n):
        indices = [(i + j) % n for j in range(4)]
        if is_convex_quad(points, indices):
            area = polygon_area(points[indices])
            valid_quads.append((indices, area))

    valid_quads.sort(key=lambda x: x[1])
    delete_indexes = set()
    [delete_indexes.update(i[0]) for i in valid_quads[:len(low_peaks) // 2]]
    corner_points = np.delete(np.array(all_extrema), list(delete_indexes))

    display_plot_plotly(
        [
            [
                [M_j[0], M_j[1]],
                "lines", "Моменти", "#FF00FF", {}, True
            ],
            [
                M_j[:, high_peaks] if not direction else M_j[:, low_peaks],
                "markers", "Top", "#FF0F00", {}, True
            ],
            [
                M_j[:, low_peaks] if not direction else M_j[:, high_peaks],
                "markers", "Low", "#5800FF", {}, True
            ],
            [
                M_j[:, corner_points],
                "markers", "Corners", "#FFCC00", {}, True
            ],
        ],
        filename=f"smooth_contour/d_{general_l}/{puzzle_index}/{'straight' if direction else 'reverse'}/corner_candidats_moments"
    )

    return corner_points

def check_dir(path):
    path = "/".join(path.split("/")[:-1])
    if not os.path.exists(path):
        os.makedirs(path)




