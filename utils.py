import matplotlib.pyplot as plt
import numpy as np

from math import pi, sqrt, sin, cos, radians, cosh, sinh

from constants import PLOT_DATA_ROUND, DPI_VALIE, PLOT_DISPLAY_SIZE, PLOT_LEGEND_FONT_SIZE, PLOT_MARKET_SIZE,\
                      PLOT_LINE_WIDTH, PLOT_TITLE_FONT_SIZE, PLOT_ANOTATE_FONT_SIZE, PLOT_ASIX_FONT_SIZE, I, E


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
                plt.axis('equal')
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
    return sin_phi, cos_phi

def klotoid_align_value_count(d, pi_coef = 0, klotoid=False, index=1, clock="clockwise"):
    psis = np.array([])

    for i in range(2):
        psi = to_angle(d[i][0], d[i][1], 1, 0)
        #TODO  FIX
        if clock == "clockwise":
            cof_1 = 1
            cof_2 = 0
            minis_cof = -1
        else:
            cof_1 = 0
            cof_2 = 1
            minis_cof = 1

        if psi[0] > 0 and psi[1] > 0:
            print(1, end=" ")
            if klotoid and i == index:
                if pi_coef % 2 == cof_1:
                    if np.arcsin(psi[0]) > 0:
                        pi_coef -= 1
                    else:
                        pi_coef += 1
                psis = np.append(psis, pi_coef * pi - minis_cof * np.arcsin(psi[0]))
            else:
                psis = np.append(psis, np.arcsin(psi[0]))

        elif psi[0] > 0 and psi[1] < 0:
            print(2, end=" ")
            if klotoid and i == index:
                psis = np.append(psis, pi_coef * pi + pi - np.arcsin(psi[0]))
            else:
                psis = np.append(psis, pi - np.arcsin(psi[0]))

        elif psi[0] < 0 and psi[1] < 0:
            print(3, end=" ")
            if klotoid and i == index:
                if pi_coef % 2 == cof_2:
                    if np.arcsin(psi[0]) < 0:
                        pi_coef += 1
                    else:
                        pi_coef -= 1
                psis = np.append(psis, pi_coef * pi + minis_cof * np.arcsin(psi[0]))
            else:
                psis = np.append(psis, -pi - np.arcsin(psi[0]))

        elif psi[0] < 0 and psi[1] > 0:
            print(4, end=" ")
            if klotoid and i == index:
                psis = np.append(psis, (3 * pi) / 2 + pi/2 + np.arcsin(psi[0]))
            else:
                psis = np.append(psis, np.arcsin(psi[0]))
        elif round(psi[0], 5) == 0 and round(psi[1], 5) == 1:
            psis = np.append(psis, np.arcsin(psi[0]))
        elif round(psi[0], 5) == 0 and round(psi[1], 5) == -1:
            psis = np.append(psis, np.arcsin(psi[0]))
        elif round(psi[0], 5) == 1 and round(psi[1], 5) == 0:
            psis = np.append(psis, np.arcsin(psi[0]))
        elif round(psi[0], 5) == -1 and round(psi[1], 5) == 0:
            psis = np.append(psis, np.arcsin(psi[0]))
        else:
            print(psi, "\nERROR!!!")

    return list(map(round, psis, [6, 6])), pi_coef

def align_value_count(M, d, curve_type):
    psis_sin = np.array([])
    psis = np.array([])

    if curve_type == "not_loop":
        M -= 1

    for i in range(M):
        psi = to_angle(d[i][0], d[i][1], d[i+1][0], d[i+1][1])
        if psi[0] > 0 and psi[1] > 0:
            psis_sin = np.append(psis_sin, psi[0])
            psis = np.append(psis, np.arcsin(psi[0]))
        elif psi[0] > 0 and psi[1] < 0:
            psis_sin = np.append(psis_sin, psi[0])
            psis = np.append(psis, pi - np.arcsin(psi[0]))
        elif psi[0] < 0 and psi[1] < 0:
            psis_sin = np.append(psis_sin, psi[0])
            psis = np.append(psis, -pi - np.arcsin(psi[0]))
        elif psi[0] < 0 and psi[1] > 0:
            psis_sin = np.append(psis_sin, psi[0])
            psis = np.append(psis, np.arcsin(psi[0]))
        elif round(psi[0], 5) == 0 and round(psi[1], 5) == 1:
            psis_sin = np.append(psis_sin, psi[0])
            psis = np.append(psis, np.arcsin(psi[0]))
        elif round(psi[0], 5) == 0 and round(psi[1], 5) == -1:
            psis_sin = np.append(psis_sin, psi[0])
            psis = np.append(psis, np.arcsin(psi[0]))
        elif round(psi[0], 5) == 1 and round(psi[1], 5) == 0:
            psis_sin = np.append(psis_sin, psi[0])
            psis = np.append(psis, np.arcsin(psi[0]))
        elif round(psi[0], 5) == -1 and round(psi[1], 5) == 0:
            psis_sin = np.append(psis_sin, psi[0])
            psis = np.append(psis, np.arcsin(psi[0]))
        else:
            print(psi, "\nERROR!!!")

    return psis_sin, psis


def len_value_count(M, d):
    S = np.array([])
    for i in range(M):
        S = np.append(S, np.sqrt(d[i][0]**2 + d[i][1]**2))
    return S


def C_coef_value_count(M, S, C_proportion_coef):
    return (6/((sum(S)/M)**3))*C_proportion_coef


def matrix_coefs(M, S, psis, C, point_type, equation_type, P_align_coef=None, extra_psis=None, aligns=None, iter=None):
    dims = 8 * M
    matrix = np.zeros((dims, dims))
    coefs = np.zeros((dims))

    if equation_type == "not_loop":
        # k = 1/sum(S)
        k = (1/1000)
        chsh = False
        additional_align = 0

        if iter > 4:
            chsh = False
            additional_align = 13 * ((iter) // 4)
            # additional_align = 10 * (iter - 4)

        if iter > 5:
            k = k * (1.001 ** (iter - 5))

        # print(iter, k)

        for i in range(M):
            # Рівняння зв'язку
            matrix[i*8+2, i*8], matrix[i*8+2, i*8+1], \
            matrix[i*8+2, i*8+2], matrix[i*8+2, i*8+3] = \
                ((1, S[i], S[i] ** 2 / (2 * I * E), S[i] ** 3 / (6 * I * E))
                if not chsh else
                (1, S[i], (cosh(k*S[i]) - 1)/k**2, (sinh(k*S[i]) - k*S[i])/k**3))
                # (1, S[i], (1 - cos(k * S[i])) / k ** 2, (k * S[i] - sin(k * S[i])) / k ** 3))



            matrix[i*8+3, i*8], matrix[i*8+3, i*8+1], \
            matrix[i*8+3, i*8+2], matrix[i*8+3, i*8+3] = \
                ((0, 1, S[i] / (I * E), S[i] ** 2 / (2 * I * E))
                if not chsh else
                (0, 1, (k*sinh(k*S[i]))/k**2, (cosh(k*S[i]) - 1)/k**2))
                # (0, 1, k * sin(k * S[i]) / k ** 2, (1 - cos(k * S[i])) / k ** 2))

            matrix[i*8+4, i*8], matrix[i*8+4, i*8+1], \
            matrix[i*8+4, i*8+2], matrix[i*8+4, i*8+3] = \
                ((0, 0, 1, S[i])
                if not chsh else
                (0, 0, cosh(k*S[i]), sinh(k*S[i])/k))
                # (0, 0, cos(k * S[i]), sin(k * S[i]) / k))


            matrix[i*8+5, i*8], matrix[i*8+5, i*8+1], \
            matrix[i*8+5, i*8+2], matrix[i*8+5, i*8+3] = \
                ((0, 0, 0, 1)
                if not chsh else
                (0, 0, k*sinh(k*S[i]), cosh(k*S[i])))
                # (0, 0, -k * sin(k * S[i]), cos(k * S[i])))

            matrix[i*8+2, i*8+4], matrix[i*8+3, i*8+5], matrix[i*8+4, i*8+6], matrix[i*8+5, i*8+7] = -1, -1, -1, -1

            if i < M - 1:
                matrix[i*8+6, i*8+4], matrix[i*8+7, i*8+5], matrix[i*8+8, i*8+6], matrix[i*8+9, i*8+7] = 1, 1, 1, 1
                matrix[i*8+6, (i+1)*8], matrix[i*8+7, (i+1)*8+1], matrix[i*8+8, (i+1)*8+2], matrix[i*8+9, (i+1)*8+3] = -1, -1, -1, -1
                if point_type[i+1] == 0:
                    matrix[i*8+9, i*8+8] = -C
                elif point_type[i+1] == 1:
                    matrix[i*8+9, i*8+7] = 0
                    matrix[i*8+9, (i+1)*8+3] = 0
                    matrix[i*8+9, i*8+4] = 1
                coefs[i*8+7] = psis[i]

            if P_align_coef is not None:
                coefs[i*8+7] = -C*P_align_coef[i]

        # --------------klotoid--------------
        coefs[1] = (radians(aligns[0]) + aligns[1] * extra_psis[0])
        coefs[-2] = (radians(aligns[2]) + aligns[3] * extra_psis[-1])

        matrix[0][0], matrix[1][1] = 1, 1
        matrix[-2][-3], matrix[-1][-2] = 1, 1
        # --------------klotoid--------------

        # coefs[1] = (radians(aligns[0]) + aligns[1] * extra_psis[0])
        # # coefs[1] = extra_psis[0]
        # coefs[-1] = (radians(aligns[2]) + aligns[3] * extra_psis[-1])
        # # print(degrees(coefs[1]), degrees(coefs[-1]))
        # matrix[0][0], matrix[1][1] = 1, 1
        # matrix[-2][-4], matrix[-1][-3] = 1, 1

        print(coefs[1], coefs[-1])
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
                    matrix[i*8+7, i*8+8] = -C
                elif point_type[i] == 1:
                    matrix[i*8+7, i*8+7] = 0
                    matrix[i*8+7, (i+1)*8+3] = 0
                    matrix[i*8+7, i*8+4] = 1

            else:
                matrix[i*8+4, i*8+4], matrix[i*8+5, i*8+5], matrix[i*8+6, i*8+6], matrix[i*8+7, i*8+7] = 1, 1, 1, 1
                matrix[i*8+4, 0], matrix[i*8+5, 1], matrix[i*8+6, 2], matrix[i*8+7, 3] = -1, -1, -1, -1
                if point_type[i] == 0:
                    matrix[i*8+7, 0] = -C
                elif point_type[i] == 1:
                    matrix[i*8+7, i*8+7] = 0
                    matrix[i*8+7, 3] = 0
                    matrix[i*8+7, i*8+4] = 1


            coefs[i*8+5] = psis[i]
            if P_align_coef is not None:
                coefs[i*8+7] = -C*P_align_coef[i]

    # display_table(matrix, bad_data = False, revert=True)

    return matrix, coefs


def len_calc(k, X, Y, x, y):
    return abs((y-Y)/(sqrt(1+k**2))-(k*(x-X))/(sqrt(1+k**2)))


def P_coef_count(M, d, X, Y, X_n, Y_n):
    P_align_coef = []

    if M != 1:
        for i in range(M):
            psi_0 = np.sign(np.arcsin(to_angle(d[i][0], d[i][1], X[i+1]-X_n[i+1], Y[i+1]-Y_n[i+1])[0]))
            psi_1 = np.sign(np.arcsin(to_angle(d[i+1][0], d[i+1][1], X[i+1]-X_n[i+1], Y[i+1]-Y_n[i+1])[0]))
            k_0 = (Y_n[(i+1)%M]-Y_n[i%M])/(X_n[(i+1)%M]-X_n[i%M])
            k_1 = (Y_n[(i+2)%M]-Y_n[(i+1)%M])/(X_n[(i+2)%M]-X_n[(i+1)%M])
            len_0 = len_calc(k_0, X_n[i+1], Y_n[i+1], X[i+1], Y[i+1])
            len_1 = len_calc(k_1, X_n[i+1], Y_n[i+1], X[i+1], Y[i+1])
            P_align_coef.append(psi_0 * len_0 if len_0 > len_1 else psi_1 * len_1)
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


def midle_point_params_vector(M, S, solution, list_of_patrs):
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

    for i in range(M):
        M_j.append([sum(S[:i]), solution[8*i+2]])
        X_ = 1 - solution[8*i+1]*sin(solution[8*i+1]) - solution[8*i]*cos(solution[8*i+1])*solution[8*i+2]
        Y_ = solution[8*i+1]*cos(solution[8*i+1]) - solution[8*i]*sin(solution[8*i+1])*solution[8*i+2]
        X__ = (-solution[8*i+2]*sin(solution[8*i+1]) - 2*solution[8*i+1]*cos(solution[8*i+1])*solution[8*i+2]
               + solution[8*i]*sin(solution[8*i+1])*(solution[8*i+2]**2) - solution[8*i]*cos(solution[8*i+1])*solution[8*i+3])
        Y__ = (solution[8*i+2]*cos(solution[8*i+1]) - 2*solution[8*i+1]*sin(solution[8*i+1])*solution[8*i+2]
               - solution[8*i]*cos(solution[8*i+1])*(solution[8*i+2]**2) - solution[8*i]*sin(solution[8*i+1])*solution[8*i+3])
        # print((-(X__*Y_ - Y__*X_))/((sqrt(X_**2 + Y_**2))**3), solution[8*i+2], cos(solution[8*i+1]), 2*solution[8*i+1], sin(solution[8*i+1]), solution[8*i+2])
        M_j_coreg.append([sum(S[:i]), (-(X__*Y_ - Y__*X_))/((sqrt(X_**2 + Y_**2))**3), S[i]*list_of_patrs[0]])
        D_j.append([X[i] + solution[8*i] * c_l_norm[i], Y[i] + solution[8*i] * d_l_norm[i]])
        D_j_coreg.append([X[i] + solution[8*i] * c_n_norm[i], Y[i] + solution[8*i] * d_n_norm[i]])
        for k in range(len(list_of_patrs)):
            index = len(list_of_patrs)*i+k
            M_j.append([M_j[i*len(list_of_patrs)+k+i][0]+S[i]*list_of_patrs[0], sol_half[index][2]])
            X_ = 1 - sol_half[index][1]*sin(sol_half[index][1]) - sol_half[index][0]*cos(sol_half[index][1])*sol_half[index][2]
            Y_ = sol_half[index][1]*cos(sol_half[index][1]) - sol_half[index][0]*sin(sol_half[index][1])*sol_half[index][2]
            X__ = (-sol_half[index][2]*sin(sol_half[index][1]) - 2*sol_half[index][1]*cos(sol_half[index][1])*sol_half[index][2]
                   + sol_half[index][0]*sin(sol_half[index][1])*(sol_half[index][2]**2) - sol_half[index][0]*cos(sol_half[index][1])*sol_half[index][3])
            Y__ = (sol_half[index][2]*cos(sol_half[index][1]) - 2*sol_half[index][1]*sin(sol_half[index][1])*sol_half[index][2]
                   - sol_half[index][0]*cos(sol_half[index][1])*(sol_half[index][2]**2) - sol_half[index][0]*sin(sol_half[index][1])*sol_half[index][3])
            M_j_coreg.append([M_j[i*len(list_of_patrs)+k+i][0]+S[i]*list_of_patrs[0], (-(X__*Y_ - Y__*X_))/((sqrt(X_**2 + Y_**2))**3), S[i]*list_of_patrs[0]])
            D_j.append([B_j[index][0] + sol_half[index][0] * c_l_norm[i], B_j[index][1] + sol_half[index][0] * d_l_norm[i]])
            D_j_coreg.append([B_j[index][0] + sol_half[index][0] * c_n_norm_j[index], B_j[index][1] + sol_half[index][0] * d_n_norm_j[index]])
    M_j.append([sum(S), solution[-2]])
    X_ = 1 - solution[-3]*sin(solution[-3]) - solution[-4]*cos(solution[-3])*solution[-2]
    Y_ = solution[-3]*cos(solution[-3]) - solution[-4]*sin(solution[-3])*solution[-2]
    X__ = (-solution[-2]*sin(solution[-3]) - 2*solution[-3]*cos(solution[-3])*solution[-2]
           + solution[-4]*sin(solution[-3])*(solution[-2]**2) - solution[-4]*cos(solution[-3])*solution[-1])
    Y__ = (solution[-2]*cos(solution[-3]) - 2*solution[-3]*sin(solution[-3])*solution[-2]
           - solution[-4]*cos(solution[-3])*(solution[-2]**2) - solution[-4]*sin(solution[-3])*solution[-1])
    M_j_coreg.append([sum(S), (-(X__*Y_ - Y__*X_))/((sqrt(X_**2 + Y_**2))**3), S[-1]*list_of_patrs[0]])
    D_j.append([X[-1] + solution[-4] * c_l_norm[-1], Y[-1] + solution[-4] * d_l_norm[-1]])
    D_j_coreg.append([X[-1] + solution[-4] * c_n_norm[-1], Y[-1] + solution[-4] * d_n_norm[-1]])



    return map(np.transpose, map(np.array, [M_j, M_j_coreg, D_j, D_j_coreg]))
