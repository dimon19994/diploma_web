import json
import time

from flask import render_template
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

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
    display_plot_plotly,
    find_near_point,
    order_points,
    check_dir,
)


class Calculate(_Controller):
    def _post(self):
        interval = 200
        straight = bool(int(self.request_data.get('straight', True)))

        data = json.loads(self.request_data["data"])

        if straight:
            data = data
        else:
            data = data[::-1]

        iterations = int(self.request_data["iter_count"])

        general_l_imput = float(self.request_data.get('general_l'))
        general_l = general_l_imput or 15
        L = 50
        d_4 = L**4

        parts = interval
        list_of_patrs = [i/parts for i in range(1, parts)]

        curve_type = self.request_data.get('curve_type', "not_loop")
        save_data = bool(int(self.request_data['save_data']))
        file_name = self.request_data.get('file_name', "").split(".")[0]
        side = int(self.request_data['side'])

        aligns = None

        puzzle_and_direction = f"{self.request_data.get('puzzle_index')}_{'straight' if straight else 'reverse'}"
        puzzle_index = self.request_data.get('puzzle_index')
        direction = 'straight' if straight else 'reverse'

        data = np.unique(np.array(data), axis=0)

        x_base = data[:, 0]
        y_base = data[:, 1]
        point_type = data[:, 2]

        first_iter_skip = 20

        if "test" in file_name:
            x = x_base
            y = y_base
        else:
            x = x_base[::first_iter_skip]
            y = y_base[::first_iter_skip]

            if (len(x_base) - 1) % first_iter_skip != 0:
                x = np.append(x, x_base[-1])
                y = np.append(y, y_base[-1])

        if curve_type == "not_loop":
            file_dataset_len = len(x) - 1
        else:
            file_dataset_len = len(x)

        if curve_type == "loop":
            x = np.array([*x, x[0]])
            y = np.array([*y, y[0]])
            x_base = np.array([*x_base, x_base[0]])
            y_base = np.array([*y_base, y_base[0]])
        else:
            x = np.array(x)
            y = np.array(y)

        response_images = []

        SHOW_NEW_TYPE_PLOTS = True

        for iteration in range(iterations):
            print(iteration)

            start = time.time()

            if curve_type == "loop":
                d = vector_cords(file_dataset_len, x, y)
            else:
                d = vector_cords_not_loop(file_dataset_len, x, y)
            psis = align_value_count(file_dataset_len, d, curve_type)

            if curve_type == "not_loop":
                d_abs = np.array([d[0], [1, 0], d[-1], [1, 0]])
                psis_abs = align_value_count(4, d_abs, curve_type)
            else:
                psis_abs = None, None

            # if display_aligns_table:
            #     display_table((psis_sin, np.degrees(psis), psis), columns_name = ["SIN в радіанах", "Градуси", "Радіани"])
            #     pass
            S_input = len_value_count(file_dataset_len, d)

            if iteration == 0:
                C = C_coef_value_count(file_dataset_len, S_input) / d_4
                matrix, coefs = matrix_coefs(file_dataset_len, S_input, psis, C, point_type, curve_type, extra_psis=psis_abs, aligns=aligns)
            elif iteration > 0 and iteration < iterations:

                C = C_coef_value_count(file_dataset_len, S_input) / d_4
                # if curve_type == "loop":
                P_align_coef = P_coef_count(file_dataset_len, d, x_base, y_base, x, y, curve_type)
                # else:
                #     P_align_coef = None
                matrix, coefs = matrix_coefs(file_dataset_len, S_input, psis, C, point_type, curve_type, P_align_coef=P_align_coef)
            else:
                # C_ris /= C_step
                # C /= C_step
                C = C_coef_value_count(file_dataset_len, S_input) / d_4
                P_align_coef = P_coef_count(file_dataset_len, d, x_base, y_base, x, y, curve_type)
                matrix, coefs = matrix_coefs(file_dataset_len, S_input, psis, C, point_type[iteration], curve_type, P_align_coef=P_align_coef)

            # with open(f"{curve_type}_{iteration}.csv", "w") as f:
            #     f.write("\n".join(["\t".join(map(str, i)).replace(".", ",") for i in matrix]))

            # with open(f"{curve_type}_{iteration}_coefs.csv", "w") as f:
            #     f.write("\n".join([str(i).replace(".", ",") for i in coefs]))


            # solution = np.linalg.solve(matrix, coefs)
            solution = spsolve(csr_matrix(matrix), coefs)

            solution_time = time.time()
            print(f"Время выполнения (solution_time): {solution_time - start:.4f} секунд")


            display_solution = solution.reshape(solution.shape[0]//8, 8).transpose()
            # "\n".join([" ".join(map(str, i)).replace(".", ",") for i in display_solution])

            # moments = np.append(solution[10::8], solution[2]) - solution[6::8]


            # ---- current_task --------
            extra_psis = [solution[1], solution[-3]]
            # if round(solution[1] - radians(30), 5) == 0 and radians(solution[-3] - radians(30)) == 0:
            #     break
            # ---- current_task --------


            # if display_solution_table:
            #     display_table(np.transpose(solution.reshape(points_count, 8)), rows_name=["W_0", "θ_0", "M_0", "Q_0", "W_l", "θ_l", "M_l", "Q_l"], bad_data = False, revert=True)
            a_norm, b_norm, c_l_norm, d_l_norm, c_n_norm, d_n_norm = vector_normalization(d, S_input, solution, curve_type)
            sol_half = midle_point_params_vector(file_dataset_len, S_input, solution, list_of_patrs, psis)
            B_j, c_n_norm_B_j, d_n_norm_B_j = midle_point_count(file_dataset_len, list_of_patrs, x, y, S_input, a_norm, b_norm, sol_half)
            M_j, M_j_coreg, D_j, D_j_coreg = new_position_count(file_dataset_len, S_input, x, y, solution, c_l_norm, c_n_norm, c_n_norm_B_j, d_l_norm, d_n_norm, d_n_norm_B_j, sol_half, list_of_patrs, B_j, curve_type)

            # M_j[1] = M_j[1] * -1

            # if iteration == 0:
            #     x = D_j_coreg[0, ::parts]
            #     y = D_j_coreg[1, ::parts]
            # else:
            #     x = D_j_coreg[0, ::parts]
            #     y = D_j_coreg[1, ::parts]

            coreg_time = time.time()
            print(f"Время выполнения (coreg_time): {coreg_time - solution_time:.4f} секунд")

            if iteration == (iterations - 1):
                if save_data:
                    # if straight:
                    #     # display_plot_plotly(
                    #     #     [
                    #     #         [
                    #     #             [M_j[0], M_j[1]],
                    #     #             "lines", "Моменти", "#FF00FF", {}, True
                    #     #         ],
                    #     #     ],
                    #     #     filename=f"smooth_contour/d_{general_l}/{puzzle_index}/{direction}/pre_moments"
                    #     # )
                    #     # minima_indices = find_local_minima(M_j[1])
                    #     # top_12_indices = minima_indices[np.argsort(M_j[1][minima_indices])]
                    #     # # top_8_indices = minima_indices[np.argsort(M_j[1][minima_indices])[:4]]
                    #     # # [554 210 407 778]
                    #     #
                    #     # points_top = np.vstack((D_j_coreg[0][top_12_indices], D_j_coreg[1][top_12_indices])).transpose()
                    #     # corner_points = find_max_area_quadrilateral(points_top)
                    #     #
                    #     # indexes = [i for i in range(len(top_12_indices)) if points_top[i] in corner_points]
                    #     # top_4_indices = top_12_indices[indexes]
                    #
                    #     minima_indices = find_local_minima(M_j[1])
                    #     top_4_indices = np.array([
                    #         minima_indices[np.argmin(np.abs(minima_indices - val))]
                    #         for val in top_4_candidates
                    #     ])
                    #     top_4_indices.sort()
                    #
                    #     max_side_index = np.argmax(D_j_coreg[1])
                    #     for i in range(len(top_4_indices)):
                    #         if max_side_index < top_4_indices[i]:
                    #             top_4_indices = np.roll(top_4_indices, len(top_4_indices) - i)
                    #             break
                    #
                    #     # plot = display_plot([points_top.transpose(),
                    #     #                      np.vstack((D_j_coreg[0][top_4_indices], D_j_coreg[1][top_4_indices])), [x, y]],
                    #     #                     labels=['top 8', "top 4", "all"],
                    #     #                     color_line=['og', '-oy', "-m"],
                    #     #                     title="", annotate_step=[1, 1, 100], points_count=8)
                    #     # plot.show()
                    #
                    #     display_plot_plotly(
                    #         [
                    #             [
                    #                 np.vstack((D_j_coreg[0][top_4_indices], D_j_coreg[1][top_4_indices])),
                    #                 "lines+markers", "top 4", "#FF4500", {}, True
                    #             ],
                    #             [
                    #                 D_j_coreg,
                    #                 "lines", "all", "#054907", {}, True
                    #             ],
                    #         ],
                    #         equal=True,
                    #         filename=f"smooth_contour/d_{general_l}/{puzzle_index}/{direction}/top_points_on_contur"
                    #     )
                    #
                    #
                    #     display_corner_points = M_j[:, top_4_indices][:2]
                    # else:
                    #     # display_plot_plotly(
                    #     #     [
                    #     #         [
                    #     #             [M_j[0], M_j[1]],
                    #     #             "lines", "Моменти", "#FF00FF", {}, True
                    #     #         ],
                    #     #     ],
                    #     #     filename=f"smooth_contour/d_{general_l}/{puzzle_index}/{direction}/pre_moments"
                    #     # )
                    #     # with open("jpt_file.csv", "w") as f:
                    #     #     for indddd in range(len(D_j_coreg[0])):
                    #     #         f.write(f"{D_j_coreg[0][indddd]},{D_j_coreg[1][indddd]},{M_j[1][indddd]}\n")
                    #     #
                    #     # maxima_indices = find_local_maxima(M_j[1])
                    #     # top_12_indices = maxima_indices[np.argsort(M_j[1][maxima_indices])][::-1]
                    #     # # top_8_indices = maxima_indices[np.argsort(M_j[1][maxima_indices])[-4:]]
                    #     #
                    #     # points_top = np.vstack((D_j_coreg[0][top_12_indices], D_j_coreg[1][top_12_indices])).transpose()
                    #     # corner_points = find_max_area_quadrilateral(points_top)
                    #     #
                    #     # indexes = [i for i in range(len(top_12_indices)) if points_top[i] in corner_points]
                    #     # top_4_indices = top_12_indices[indexes]
                    #
                    #     maxima_indices = find_local_maxima(M_j[1])
                    #     top_4_indices = np.array([
                    #         maxima_indices[np.argmin(np.abs(maxima_indices - val))]
                    #         for val in top_4_candidates
                    #     ])
                    #     top_4_indices.sort()
                    #
                    #     max_side_index = np.argmax(D_j_coreg[1])
                    #     for i in range(len(top_4_indices)):
                    #         if max_side_index < top_4_indices[i]:
                    #             top_4_indices = np.roll(top_4_indices, len(top_4_indices) - i)
                    #             break
                    #
                    #     # plot = display_plot([points_top.transpose(),
                    #     #                      np.vstack((D_j_coreg[0][top_4_indices], D_j_coreg[1][top_4_indices])), [x, y]],
                    #     #                     labels=['top 8', "top 4", "all"],
                    #     #                     color_line=['og', '-oy', "-m"],
                    #     #                     title="", annotate_step=[1, 1, 100], points_count=8)
                    #     # plot.show()
                    #
                    #     display_plot_plotly(
                    #         [
                    #             [
                    #                 np.vstack((D_j_coreg[0][top_4_indices], D_j_coreg[1][top_4_indices])),
                    #                 "lines+markers", "top 4", "#FF4500", {}, True
                    #             ],
                    #             [
                    #                 D_j_coreg,
                    #                 "lines", "all", "#054907", {}, True
                    #             ],
                    #         ],
                    #         equal=True,
                    #         filename=f"smooth_contour/d_{general_l}/{puzzle_index}/{direction}/top_points_on_contur"
                    #     )
                    #
                    #     display_corner_points = M_j[:, top_4_indices][:2]

                    display_plot_plotly(
                        [
                            [
                                [M_j[0], M_j[1]],
                                "lines", "Моменти", "#FF00FF", {}, True
                            ],
                        ],
                        filename=f"smooth_contour/d_{general_l}/{puzzle_index}/{side}/{direction}/moments"
                    )

                    if straight:
                        file_path = f"./processed_pazzle_data/{general_l}/puzzle/straight/{file_name}_{side + 1}_data.txt"
                        file_path_contur = f"./processed_pazzle_data/{general_l}/conturs/straight/{file_name}_{side + 1}_data.txt"
                    else:
                        file_path = f"./processed_pazzle_data/{general_l}/puzzle/reverse/{file_name}_{side + 1}_data.txt"
                        file_path_contur = f"./processed_pazzle_data/{general_l}/conturs/reverse/{file_name}_{side + 1}_data.txt"

                    check_dir(file_path)
                    check_dir(file_path_contur)

                    with open(file_path, "w") as f:
                        # if top_4_indices[i] < top_4_indices[i - 1]:
                        #     p1 = np.vstack((M_j[0, top_4_indices[i - 1]:] - M_j[0][top_4_indices[i - 1]], M_j[1, top_4_indices[i - 1]:]))
                        #     p2 = np.vstack((M_j[0, :top_4_indices[i]] + p1[0][-1], M_j[1, :top_4_indices[i]]))[:, 1:]
                        #     ddd = np.hstack((p1, p2))
                        # else:
                        #     ddd = np.vstack((M_j[0, top_4_indices[i - 1]:top_4_indices[i]] - M_j[0][top_4_indices[i - 1]],
                        #                      M_j[1, top_4_indices[i - 1]:top_4_indices[i]]))
                        #
                        # f.write("\n".join([f"{i[0]} {i[1]}" for i in ddd.transpose()]))
                        np.savetxt(file_path, M_j, delimiter=' ', fmt='%d')

                    with open(file_path_contur, "w") as f:
                        # if top_4_indices[i] < top_4_indices[i - 1]:
                        #     p1 = np.vstack((D_j_coreg[0][top_4_indices[i - 1]:], D_j_coreg[1][top_4_indices[i - 1]:]))
                        #     p2 = np.vstack((D_j_coreg[0][:top_4_indices[i]], D_j_coreg[1][:top_4_indices[i]]))[:, 1:]
                        #     ddd = np.hstack((p1, p2))
                        # else:
                        #     ddd = np.vstack((D_j_coreg[0][top_4_indices[i - 1]:top_4_indices[i]],
                        #                      D_j_coreg[1][top_4_indices[i - 1]:top_4_indices[i]]))
                        #
                        # f.write("\n".join([f"{i[0]} {i[1]}" for i in ddd.transpose()]))
                        np.savetxt(file_path_contur, D_j_coreg, delimiter=' ', fmt='%d')

                    # with open(f"{file_name}_data.txt", "w") as f:
                    #     for i in np.transpose(D_j):
                    #         f.write(f"{i[0]} {i[1]}\n")
                    #
                    # # моменты
                    # M_x = M_j[0]
                    # M_y = M_j[1]
                    # with open(f"{file_name}_moments.txt", "w") as f:
                    #     for i in range(len(M_y)):
                    #         f.write(f"{M_x[i]} {M_y[i]}\n")

                qulity = 0
                m_j_dif = np.array([*[M_j[0][i + 1] - M_j[0][i] for i in range(M_j.shape[1] - 1)], M_j[0][-1] - M_j[0][-2]])
                M_j = np.vstack((M_j, m_j_dif))
                for i in range(len(M_j[1])):
                    qulity += M_j[1][i]**2*M_j[2][i]

                print("Якість", qulity)
                print("Довжина", M_j[0][-1])
                continue


            OLD_DISPLAY = False
            if OLD_DISPLAY:
                DISPLAY_FULL = True

                if DISPLAY_FULL:
                    points_data = [[x_base, y_base], [D_j[0], D_j[1]], [x, y]]
                else:
                    if iteration == 0:
                        # points_data = [[x_base[899:1080], y_base[899:1080]], [D_j[0][899:1080], D_j[1][899:1080]], [x[45:54], y[45:54]]]
                        points_data = [[x_base[899:980], y_base[899:980]], [D_j[0][9000:9800], D_j[1][9000:9800]], [x[225:245], y[225:245]]]
                    else:
                        # points_data = [[x_base[899:1080], y_base[899:1080]], [D_j[0][17980:21600], D_j[1][17980:21600]], [x[899:1080], y[899:1080]]]
                        points_data = [[x_base[899:980], y_base[899:980]], [D_j[0][36000:39200], D_j[1][36000:39200]], [x[899:980], y[899:980]]]
                colours = ['-ob', '-oy', '']
                labels = ["Input points", "Сontinuous contour", ""]
                annotate_step = [0, 0, (file_dataset_len//10+1)]
                alpha = [1, 1, 0]
                spline_points = [[], []]
                imagine_points = [[], []]
                fixed_points = [[], []]
                # for i in range(file_dataset_len+1):
                disp_range = range(file_dataset_len+1) if DISPLAY_FULL else range(45, 55) if iteration ==0 else range(899, 1080)
                for i in disp_range:
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
                    pass
                    # points_data.append(spline_points)
                    # colours.append('oy')
                    # labels.append('')
                    # annotate_step.append(0)
                    # alpha.append(1)
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

            if curve_type == "loop" or curve_type == "not_loop" :  # and iteration == 0:
                if iteration == 0:

                    start_1_iteration = time.time()

                    new_x_list = []
                    new_y_list = []
                    used_points = set()
                    D_j_coreg_len = D_j_coreg.shape[1]

                    for index in range(len(x_base)):
                        if index % first_iter_skip == 0:
                            i = (index * 10) % D_j_coreg_len
                            point = (D_j_coreg[0, i], D_j_coreg[1, i])
                        else:
                            current_point = np.array([x_base[index], y_base[index]])
                            start = 0 if (index * 10 - 500) < 0 else (index * 10 - 500)
                            end = D_j_coreg_len if (index * 10 + 500) > D_j_coreg_len else (index * 10 + 500)

                            interval = D_j_coreg[:, start:end]

                            candidates = interval.T
                            distances = np.linalg.norm(candidates - current_point, axis=1)

                            # Mask out used points
                            mask = [tuple(p) not in used_points for p in candidates]
                            if not any(mask):
                                point = (x_base[index], y_base[index])  # fallback if all are used
                            else:
                                valid_candidates = candidates[mask]
                                valid_distances = distances[mask]
                                best_idx = np.argmin(valid_distances)
                                point = tuple(valid_candidates[best_idx])

                        new_x_list.append(point[0])
                        new_y_list.append(point[1])
                        used_points.add(point)

                    missing_ponts_time = time.time()
                    print(f"Время выполнения (missing_ponts_time): {missing_ponts_time - start_1_iteration:.4f} секунд")

                    x_with_skipped = np.array(new_x_list)
                    y_with_skipped = np.array(new_y_list)

                    if SHOW_NEW_TYPE_PLOTS:
                        display_plot_plotly(
                            [
                                [[
                                    np.array([[x_base[q], x_with_skipped[q], None] for q in range(len(x_base))]).flatten(),
                                    np.array([[y_base[q], y_with_skipped[q], None] for q in range(len(y_base))]).flatten()
                                ], "lines", "Springs", "#D7101F", {"line": {"width": 2, "dash": 'dash'}}, True],
                                [[x_base, y_base], "lines+markers", "All input", "#349950", {"line": {"width": 2}}, True],
                                [D_j_coreg, "markers", "All new points", "#015AC8", {}, True],
                                [[x_with_skipped, y_with_skipped], "markers", "Bace new points", "#C2A4FF", {}, True],
                            ],
                            equal=True,
                            filename=f"smooth_contour/d_{general_l}/{puzzle_index}/{side}/{direction}/after_1_iter_find_skipped"
                        )

                    x_near, y_near, indexes, old_positions, new_positions = find_near_point(x_base, y_base, x_with_skipped, y_with_skipped, D_j_coreg)

                    find_near_point_time = time.time()
                    print(f"Время выполнения (find_near_point_time): {find_near_point_time - missing_ponts_time:.4f} секунд")

                    if SHOW_NEW_TYPE_PLOTS:
                        display_plot_plotly(
                            [
                                [[
                                    np.array([[x_base[q], x_near[q], None] for q in range(len(x_near))]).flatten(),
                                    np.array([[y_base[q], y_near[q], None] for q in range(len(x_near))]).flatten()
                                ], "lines", "Springs", "#D7101F", {"line": {"width": 2, "dash": 'dash'}}, True],
                                [[
                                    np.array([[old_positions[q][0], new_positions[q][0], None] for q in range(len(old_positions))]).flatten(),
                                    np.array([[old_positions[q][1], new_positions[q][1], None] for q in range(len(old_positions))]).flatten()
                                ], "lines", "Move", "#00008B", {"line": {"width": 2, "dash": 'dash'}}, True],
                                [[
                                    x_base,
                                    y_base
                                ], "lines+markers", "All input", "#349950", {"line": {"width": 2}}, True],
                                [D_j_coreg, "markers", "All new points", "#015AC8", {}, True],
                                [[
                                    x_near,
                                    y_near
                                ], "markers", "Bace new points", "#C2A4FF", {}, True],
                                [new_positions.T if 0 not in new_positions.T.shape else [[], []], "markers", "Pont new position", "#D7101F", {}, True],
                                [old_positions.T if 0 not in old_positions.T.shape else [[], []], "markers", "Pont old position", "#D08D00", {}, True],
                            ],
                            equal=True,
                            filename=f"smooth_contour/d_{general_l}/{puzzle_index}/{side}/{direction}/after_1_iter_find_new_near_points"
                        )

                    x, y, x_base, y_base = order_points(D_j_coreg[0], D_j_coreg[1], x_near, y_near, x_base, y_base)

                    if SHOW_NEW_TYPE_PLOTS:
                        display_plot_plotly(
                            [
                                [[
                                    np.array([[x_base[q], x[q], None] for q in range(len(x_base))]).flatten(),
                                    np.array([[y_base[q], y[q], None] for q in range(len(y_base))]).flatten()
                                ], "lines", "Springs", "#D7101F", {"line": {"width": 2, "dash": 'dash'}}, True],
                                [[
                                    x_base,
                                    y_base
                                ], "lines+markers", "All input", "#349950", {"line": {"width": 3}}, True],
                                [D_j_coreg, "markers", "All new points", "#015AC8", {}, True],
                                [[x, y], "markers", "Bace new points", "#C2A4FF", {}, True],
                            ],
                            equal=True,
                            filename=f"smooth_contour/d_{general_l}/{puzzle_index}/{side}/{direction}/after_1_iter_after_ordering"
                        )

                    order_points_time = time.time()
                    print(f"Время выполнения (order_points_time): {order_points_time - find_near_point_time:.4f} секунд")

                    parts = 40
                    list_of_patrs = [i / parts for i in range(1, parts)]

                    L = L / 2
                    d_4 = L ** 4

                else:
                    # if iteration == 2:
                    #     top_4_candidates = get_corner_points_candidate(M_j, D_j_coreg, straight, general_l, puzzle_index)
                    #     print(f"{top_4_candidates/40=}")
                    #
                    #     display_plot_plotly(
                    #         [
                    #             [
                    #                 np.vstack((D_j_coreg[0][top_4_candidates], D_j_coreg[1][top_4_candidates])),
                    #                 "lines+markers", "top 4", "#FF4500", {}, True
                    #             ],
                    #             [
                    #                 D_j_coreg,
                    #                 "lines", "all", "#054907", {}, True
                    #             ],
                    #         ],
                    #         equal=True,
                    #         filename=f"smooth_contour/d_{general_l}/{puzzle_index}/{direction}/top_points_candidats_on_contur"
                    #     )

                    start_n_iteration = time.time()

                    x_ooold = x
                    y_ooold = y

                    x = D_j_coreg[0, ::parts]
                    y = D_j_coreg[1, ::parts]

                    find_near_point_start = time.time()
                    x, y, indexes, old_positions, new_positions = find_near_point(x_base, y_base, x, y, D_j_coreg)
                    find_near_point_end = time.time()
                    print(f"Время выполнения (find_near_point_end): {find_near_point_end - find_near_point_start:.4f} секунд")


                    if SHOW_NEW_TYPE_PLOTS:
                        display_plot_plotly(
                            [
                                [[
                                    np.array([[x_base[q], x[q], None] for q in range(len(x))]).flatten(),
                                    np.array([[y_base[q], y[q], None] for q in range(len(y))]).flatten()
                                ], "lines", "Springs", "#D7101F", {"line": {"width": 2, "dash": 'dash'}}, True],

                                [[
                                    x_ooold,
                                    y_ooold
                                ], "lines+markers", "Prew iter", "#349950", {"line": {"width": 3}}, True],
                                [[
                                    x_base,
                                    y_base,
                                ], "lines+markers", "Input", "#D08D00", {}, True],
                                [[
                                    x_base,
                                    y_base,
                                ], "lines", "Input", "#D08D00", {}, True],
                                [[
                                    x,
                                    y
                                ], "lines+markers", "New iter", "#015AC8", {}, True],
                                [[
                                    x,
                                    y
                                ], "lines", "New iter", "#015AC8", {}, True],
                            ],
                            equal=True,
                            filename=f"smooth_contour/d_{general_l}/{puzzle_index}/{side}/{direction}/after_{iteration+1}_iter_find_new_near_points",
                        )

                    order_points_start = time.time()
                    x, y, x_base, y_base = order_points(D_j_coreg[0], D_j_coreg[1], x, y, x_base, y_base)
                    order_points_end = time.time()
                    print(f"Время выполнения (order_points): {order_points_end - order_points_start:.4f} секунд")

                    if SHOW_NEW_TYPE_PLOTS:
                        display_plot_plotly(
                            [
                                [[
                                    np.array([[x_base[q], x[q], None] for q in range(len(x_base))]).flatten(),
                                    np.array([[y_base[q], y[q], None] for q in range(len(y_base))]).flatten()
                                ], "lines", "Springs", "#D7101F", {"line": {"width": 2, "dash": 'dash'}}, True],

                                [[
                                    x_ooold,
                                    y_ooold
                                ], "lines+markers", "Prew iter", "#349950", {"line": {"width": 3}}, True],
                                [[
                                    x_base,
                                    y_base,
                                ], "lines+markers", "Input", "#D08D00", {}, True],
                                [[
                                    x_base,
                                    y_base,
                                ], "lines", "Input", "#D08D00", {}, True],
                                [[
                                    x,
                                    y
                                ], "lines+markers", "New iter", "#015AC8", {}, True],
                                [[
                                    x,
                                    y
                                ], "lines", "New iter", "#015AC8", {}, True],
                            ],
                            equal=True,
                            filename=f"smooth_contour/d_{general_l}/{puzzle_index}/{side}/{direction}/after_{iteration+1}_iter_find_missing_spring",
                            background_image = f"/Users/dmyrto_koltsov/PycharmProjects/PDF/my_data/{file_name}.jpg"
                        )

                    order_points_n_time = time.time()
                    print(f"Время выполнения (general_n_time): {order_points_n_time - start_n_iteration:.4f} секунд")

                    if L / 2 < general_l:
                        L = general_l
                    else:
                        L = L / 2

                    d_4 = L ** 4


                # for im in range(2, int(parts**0.5)+1):
                #     if parts % im == 0:
                #         im_points_count = im
                #         break
                # else:
                #     im_points_count = parts
                #
                # x = D_j_coreg[0, ::parts//im_points_count]
                # y = D_j_coreg[1, ::parts//im_points_count]
                #
                # for i in range(len(x) - len(point_type)):
                #     point_type = np.insert(point_type, 1, 2)

                point_type = np.zeros(shape=(len(x)))
                file_dataset_len = len(x) - 1
                # parts = 0
                # list_of_patrs = []
            else:
                pass
                # display_plot_plotly([
                #     [[x[1337:1345], y[1337:1345]], "lines", "input"],
                #     [[x_base[1337:1345], y_base[1337:1345]], "lines+markers", "base"],
                #     # [[D_j_coreg[0, 1337:1345], D_j_coreg[1, 1337:1345]], "lines+markers", "new"],
                # ], False)



        return {"plots": response_images}

    def _get(self):
        return render_template("main_page.html")
