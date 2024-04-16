import json
from io import BytesIO
import base64
from math import radians

from flask import render_template, send_file
import numpy as np

from constants import display_aligns_table
from controllers import _Controller
from utils import (
    vector_cords,
    vector_cords_not_loop,
    align_value_count,
    len_value_count,
    C_coef_value_count,
    P_coef_count,
    matrix_coefs,
    vector_normalization,
    midle_point_params_vector,
    midle_point_count,
    new_position_count,
    display_plot,
)


class Calculate(_Controller):
    def _post(self):
        data = json.loads(self.request_data["data"])

        iterations = int(self.request_data["iter_count"])

        C_start = float(self.request_data['start_value'])
        C_end = float(self.request_data['end_value'])
        C_last = float(self.request_data.get('last_value', 0.0))

        parts = int(self.request_data.get('subitems', 50))
        list_of_patrs = [i/parts for i in range(1, parts)]

        curve_type = self.request_data.get('curve_type', "not_loop")
        save_data = bool(int(self.request_data['save_data']))
        file_name = self.request_data.get('file_name', "").split(".")[0]

        if curve_type == "not_loop":
            aligns = [
                int(self.request_data.get('align_1')),
                int(self.request_data.get('direction_1', 1)),
                int(self.request_data.get('align_2')),
                int(self.request_data.get('direction_2', 1)),
            ]
        else:
            aligns = None

        if iterations > 1:
            C_step = (C_end/C_start)**(1/(iterations-1))
            # C_step = (C_end/C_start)**(1/(iterations+special_iteration-1))

        if curve_type == "not_loop":
            file_dataset_len = len(data) - 1
        else:
            file_dataset_len = len(data)

        if curve_type == "loop":
            data = np.array([*data, data[0]])
        else:
            data = np.array(data)

        x = x_base = data[:, 0]
        y = y_base = data[:, 1]
        point_type = data[:, 2]

        response_images = []

        for iteration in range(iterations):
            if curve_type == "loop":
                d = vector_cords(file_dataset_len, x, y)
            else:
                d = vector_cords_not_loop(file_dataset_len, x, y)
            psis_sin, psis = align_value_count(file_dataset_len, d, curve_type)

            if curve_type == "not_loop":
                d_abs = np.array([d[0], [1, 0], d[-1], [1, 0]])
                psis_sin_abs, psis_abs = align_value_count(4, d_abs, curve_type)
            else:
                psis_sin_abs, psis_abs = None, None

            # if display_aligns_table:
            #     display_table((psis_sin, np.degrees(psis), psis), columns_name = ["SIN в радіанах", "Градуси", "Радіани"])
            #     pass
            S_input = len_value_count(file_dataset_len, d)

            # print(S_input)

            if iteration == 0:
                C_ris = C_start
                C = C_coef_value_count(file_dataset_len, S_input, C_start)
                matrix, coefs = matrix_coefs(file_dataset_len, S_input, psis, C, point_type, curve_type, extra_psis=psis_abs, aligns=aligns, iter=iteration)
            elif iteration > 0 and iteration < iterations:
                C_ris *= C_step
                C *= C_step
                if curve_type == "loop":
                    P_align_coef = P_coef_count(file_dataset_len, d, x_base, y_base, x, y)
                else:
                    P_align_coef = None
                matrix, coefs = matrix_coefs(file_dataset_len, S_input, psis, C, point_type, curve_type, P_align_coef=P_align_coef, extra_psis=psis_abs, aligns=aligns, iter=iteration)
            else:
                C_ris /= C_step
                C /= C_step
                P_align_coef = P_coef_count(file_dataset_len, d, x_base, y_base, x, y)
                matrix, coefs = matrix_coefs(file_dataset_len, S_input, psis, C, point_type[iteration], curve_type, P_align_coef=P_align_coef, aligns=aligns)

            solution = np.linalg.solve(matrix, coefs)


            # ---- current_task --------
            extra_psis = [solution[1], solution[-3]]
            # if round(solution[1] - radians(30), 5) == 0 and radians(solution[-3] - radians(30)) == 0:
            #     break
            # ---- current_task --------


            # if display_solution_table:
            #     display_table(np.transpose(solution.reshape(points_count, 8)), rows_name=["W_0", "θ_0", "M_0", "Q_0", "W_l", "θ_l", "M_l", "Q_l"], bad_data = False, revert=True)
            a_norm, b_norm, c_l_norm, d_l_norm, c_n_norm, d_n_norm = vector_normalization(d, S_input, solution, curve_type)
            sol_half = midle_point_params_vector(file_dataset_len, S_input, solution, list_of_patrs)
            B_j, c_n_norm_B_j, d_n_norm_B_j = midle_point_count(file_dataset_len, list_of_patrs, x, y, S_input, a_norm, b_norm, sol_half)
            M_j, M_j_coreg, D_j, D_j_coreg = new_position_count(file_dataset_len, S_input, x, y, solution, c_l_norm, c_n_norm, c_n_norm_B_j, d_l_norm, d_n_norm, d_n_norm_B_j, sol_half, list_of_patrs, B_j, curve_type)

            x = D_j_coreg[0, ::parts]
            y = D_j_coreg[1, ::parts]



            if iteration == (iterations - 1):
                if save_data:
                    with open(f"{file_name}_data.txt", "w") as f:
                        for i in np.transpose(D_j_coreg):
                            f.write(f"{i[0]} {i[1]}\n")

                    # моменты
                    M_x = M_j_coreg[0]
                    M_y = M_j_coreg[1]
                    with open(f"{file_name}_moments.txt", "w") as f:
                        for i in range(len(M_y)):
                            f.write(f"{M_x[i]} {M_y[i]}\n")

                qulity = 0
                for i in range(len(M_j_coreg[1])):
                    qulity += M_j_coreg[1][i]**2*M_j_coreg[2][i]

                print("Якість", qulity)
                print("Довжина", M_j_coreg[0][-1])

            points_data = [[x_base, y_base], [D_j_coreg[0], D_j_coreg[1]], [x, y]]
            colours = ['ob', '-y', '']
            labels = ["Input points", "Сontinuous contour", ""]
            annotate_step = [0, 0, (file_dataset_len//10+1)]
            alpha = [1, 1, 0]
            spline_points = [[], []]
            imagine_points = [[], []]
            fixed_points = [[], []]
            for i in range(file_dataset_len+1):
                if point_type[i] == 0:
                    spline_points[0].append(x[i])
                    spline_points[1].append(y[i])
                elif point_type[i] == 1:
                    fixed_points[0].append(x[i])
                    fixed_points[1].append(y[i])
                elif point_type[i] == 2:
                    imagine_points[0].append(x[i])
                    imagine_points[1].append(y[i])

            if spline_points[0]:
                points_data.append(spline_points)
                colours.append('oy')
                labels.append('')
                annotate_step.append(0)
                alpha.append(1)
            if fixed_points[0]:
                points_data.append(fixed_points)
                colours.append('or')
                labels.append('Fixed points')
                annotate_step.append(0)
                alpha.append(1)
            if imagine_points[0]:
                points_data.append(imagine_points)
                colours.append('oC7')
                labels.append('Imaginary points')
                annotate_step.append(0)
                alpha.append(1)

            plot = display_plot(points_data, labels=labels, color_line=colours,
                                title="", annotate_step=annotate_step, points_count=len(x), alpha=alpha)

            # plot = display_plot([[M_j_coreg[0], M_j_coreg[1]]], labels = ['Моменти'], color_line = ['-m'],
            #                     title="", annotate_step=annotate_step, points_count=len(x), alpha=alpha)

            flike = BytesIO()
            plot.savefig(flike, format='png')
            flike.seek(0)
            image_png = flike.getvalue()
            graph = base64.b64encode(image_png).decode('utf-8')
            flike.close()

            response_images.append(graph)

            # if curve_type != "loop" and iteration == 0:
            #     x = D_j_coreg[0, ::parts//6]
            #     y = D_j_coreg[1, ::parts//6]
            #
            #
            #     for i in range(len(x) - len(point_type)):
            #         point_type = np.insert(point_type, 1, 2)
            #     file_dataset_len = (len(x) - 1)

            if curve_type != "loop" and iteration <= 4 and (len(x) - len(x_base)) < 25:
                for im in range(3, int(parts**0.5)+1):
                    if parts % im == 0:
                        im_points_count = im
                        break
                else:
                    im_points_count = parts

                x = D_j_coreg[0, ::parts//im_points_count]
                y = D_j_coreg[1, ::parts//im_points_count]

                for i in range(int((len(x) - len(point_type)) / (im_points_count - 1))):
                    for j in range(im_points_count - 1):
                        point_type = np.insert(point_type, im_points_count*i+1, 2)

                file_dataset_len = (len(x) - 1)

        return {"plots": response_images, "quality": round(qulity, 5), "length": round(M_j_coreg[0][-1], 5)}

    def _get(self):
        return render_template("main_page.html")
