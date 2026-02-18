import json
import time

from flask import render_template

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from controllers import _Controller
from utils import *
from constants import MATERIALS_PATH



class Calculate(_Controller):
    def _post(self):
        interval = 200
        display_corner_points = np.array([])
        straight = bool(int(self.request_data.get('straight', True)))

        data = json.loads(self.request_data["data"])

        if straight:
            data = data
        else:
            data = data[::-1]

        iterations = int(self.request_data["iter_count"])

        general_l_imput = float(self.request_data.get('general_l'))
        general_l = general_l_imput or 15
        start_L = float(self.request_data.get('start_l', 20))
        L = start_L
        d_4 = L**4
        scale_coef = 1.2

        # parts = int(self.request_data.get('subitems', 50))
        parts = interval
        list_of_patrs = [i/parts for i in range(1, parts)]

        curve_type = self.request_data.get('curve_type', "not_loop")
        save_data = bool(int(self.request_data['save_data']))
        file_name = self.request_data.get('file_name', "").split(".")[0]

        equal = bool(int(self.request_data['equal'])) if "equal" in self.request_data else None

        if curve_type == "not_loop":
            aligns = [
                int(self.request_data.get('align_1', 0)),
                int(self.request_data.get('direction_1', 1)),
                int(self.request_data.get('align_2', 0)),
                int(self.request_data.get('direction_2', 1)),
            ]
        else:
            aligns = None

        puzzle_and_direction = f"{self.request_data.get('puzzle_index')}_{'straight' if straight else 'reverse'}"
        puzzle_index = self.request_data.get('puzzle_index')
        direction = 'straight' if straight else 'reverse'
        corner_move = int(self.request_data.get('corner_move', 0))

        # if iterations > 1:
        #     C_step = (C_end/C_start)**(1/(iterations-1))
        #     # C_step = (C_end/C_start)**(1/(iterations+special_iteration-1))

        data = np.array(data)

        x_base = data[:, 0]
        y_base = data[:, 1]
        point_type = data[:, 2]

        # x = x_base[::4]
        # y = y_base[::4]

        skip = 1
        gate = 500

        x = x_base[::skip]
        y = y_base[::skip]

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
        display_point_positions = []
        display_corner_points_positions = []

        # scale = 0.33
        # scale = 0.5
        scale = 1
        real_corner_point = None

        SHOW_NEW_TYPE_PLOTS = True

        display_plot_plotly(
            [
                [
                    [x_base, y_base],
                    "lines+markers", "iteration input points", "black", {}, True
                ]
            ],
            equal=equal if equal is not None else True,
            save_path=f"{MATERIALS_PATH}smooth_contour/{file_name.rsplit('_', 1)[0]}/plots/d_{general_l}/{puzzle_index}/{direction}/",
            filename=f"input_points_"
        )

        for iteration in range(iterations):
            print(f"{iteration=}, {L=}")

            start = time.time()

            # display_plot_plotly(
            #     [
            #         [
            #             [x, y],
            #             "lines+markers", "Points", "#FF00FF", {}
            #         ],
            #     ],
            #     # equal=True,
            #     # filename="after_1_iter_find_missing"
            # )

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

            # print(S_input)

            if iteration == 0:
                # C_ris = C_start
                # C = C_coef_value_count(file_dataset_len, S_input, C_start)
                C = C_coef_value_count(file_dataset_len, S_input) / d_4
                matrix, coefs = matrix_coefs(file_dataset_len, S_input, psis, C, point_type, curve_type, extra_psis=psis_abs, aligns=aligns)
            elif iteration > 0 and iteration < iterations:
                # C_ris *= C_step
                # C *= C_step

                # plot = display_plot([[x, y]], labels=['1 iter'],
                #                     color_line=['-m'],
                #                     title="", annotate_step=[1], points_count=len(x))
                # plot.show()

                C = C_coef_value_count(file_dataset_len, S_input) / d_4
                if curve_type == "loop":
                    P_align_coef = P_coef_count(file_dataset_len, d, x_base, y_base, x, y)
                else:
                    # P_align_coef = None
                    P_align_coef = P_coef_count(file_dataset_len, d, x_base, y_base, x, y, roll_=False)
                matrix, coefs = matrix_coefs(file_dataset_len, S_input, psis, C, point_type, curve_type, P_align_coef=P_align_coef, extra_psis=psis_abs, aligns=aligns)
            else:
                # C_ris /= C_step
                # C /= C_step
                C = C_coef_value_count(file_dataset_len, S_input) / d_4
                P_align_coef = P_coef_count(file_dataset_len, d, x_base, y_base, x, y)
                matrix, coefs = matrix_coefs(file_dataset_len, S_input, psis, C, point_type[iteration], curve_type, P_align_coef=P_align_coef, aligns=aligns)

            # solution = np.linalg.solve(matrix, coefs)
            solution = spsolve(csr_matrix(matrix), coefs)

            solution_time = time.time()
            print(f"Время выполнения (solution_time): {solution_time - start:.4f} секунд")


            display_solution = solution.reshape(solution.shape[0]//8, 8).transpose()

            # moments = np.append(solution[10::8], solution[2]) - solution[6::8]


            # ---- current_task --------
            extra_psis = [solution[1], solution[-3]]
            # ---- current_task --------


            a_norm, b_norm, c_l_norm, d_l_norm, c_n_norm, d_n_norm = vector_normalization(d, S_input, solution, curve_type, display_corner_points, psis)

            x_norm_disp, y_norm_disp, x_norm_real_disp, y_norm_real_disp = get_norm_vectors(x, y, c_n_norm, d_n_norm, solution[::8], 10)

            sol_half = midle_point_params_vector(file_dataset_len, S_input, solution, list_of_patrs, psis)
            B_j, c_n_norm_B_j, d_n_norm_B_j = midle_point_count(file_dataset_len, list_of_patrs, x, y, S_input, a_norm, b_norm, sol_half, display_corner_points, solution, psis)
            M_j, M_j_coreg, D_j, D_j_coreg = new_position_count(
                file_dataset_len, S_input, x, y, solution, c_l_norm, c_n_norm, c_n_norm_B_j,
                d_l_norm, d_n_norm, d_n_norm_B_j, sol_half, list_of_patrs, B_j, curve_type, display_corner_points,
                scale=scale
            )

            display_plot_plotly(
                [
                    [
                        [x, y],
                        "lines+markers", "iteration input points", "black", {}, True
                    ],
                    [
                        [x_norm_disp, y_norm_disp],
                        "lines", "norm_vectors", "#D7101F", {}, False
                    ],
                    [
                        [x_norm_real_disp, y_norm_real_disp],
                        "lines", "norm_vectors_true_len", "blue", {}, True
                    ],
                    [
                        [D_j_coreg[0], D_j_coreg[1]],
                        "lines+markers", "З корегуванням", "green", {}, True
                    ],
                    [[
                        x[display_corner_points.astype(int)],
                        y[display_corner_points.astype(int)]
                    ], "markers", "Imagine corners", "red", {"marker": {"size": 10}}, True],
                ],
                equal=equal if equal is not None else True,
                save_path=f"{MATERIALS_PATH}smooth_contour/{file_name.rsplit('_', 1)[0]}/plots/d_{general_l}/{puzzle_index}/{direction}/",
                filename=f"after_{iteration+1}_new_points"
            )

            coreg_time = time.time()
            print(f"Время выполнения (coreg_time): {coreg_time - solution_time:.4f} секунд")

            if iteration == (iterations - 1):
                qulity = 0
                m_j_dif = np.array([*[M_j[0][i + 1] - M_j[0][i] for i in range(M_j.shape[1] - 1)], M_j[0][-1] - M_j[0][-2]])
                M_j = np.vstack((M_j, m_j_dif))
                for i in range(len(M_j[1])):
                    qulity += M_j[1][i]**2*M_j[2][i]

                print("Якість", qulity)
                print("Довжина", M_j[0][-1])

                if len(display_corner_points) > 0 :
                    top_4_candidates = display_corner_points
                else:
                    top_4_candidates = get_corner_points_candidate(M_j, D_j_coreg, straight, general_l, puzzle_index, file_name, full=True) // 40

                if save_data:
                    display_plot_plotly(
                        [
                            [
                                [M_j[0], M_j[1]],
                                "lines", "Моменти", "#FF00FF", {}, True
                            ],
                            [
                                M_j[:, display_corner_points.astype(int) * 40],
                                "markers", "Кутові точки", "#000000", {}, True
                            ],
                            [
                                M_j[:, top_4_candidates * 40],
                                "markers+text", "Локальні екстремуми", "brown", {"marker": {"size": 10}, "textposition": "middle right"}, True
                            ],
                        ],
                        save_path=f"{MATERIALS_PATH}smooth_contour/{file_name.rsplit('_', 1)[0]}/plots/d_{general_l}/{puzzle_index}/{direction}/",
                        filename=f"moments_{iteration+1}{('_'+str(corner_move)) if iteration == (iterations - 1) else ''}",
                    )

                pics_force = []
                force_indexes = []

                # top_4_candidates = np.append(top_4_candidates, 99)

                for tp in top_4_candidates:
                    pic_force, left, right = find_force(tp, P_align_coef, C, S_input, L * 3 / 4)

                    force_indexes.extend([zu % len(C)  for zu in range(left, right + 1)])
                    force_indexes.append(None)

                    pics_force.append(pic_force)
                    print(f"index = {tp + 1}, Робота = {pics_force[-1]:.5f}")

                if SHOW_NEW_TYPE_PLOTS:
                    display_plot_plotly(
                        [
                            [[
                                np.array([[x_base[q], D_j_coreg[0][q * 40], None] for q in range(len(x))]).flatten(),
                                np.array([[y_base[q], D_j_coreg[1][q * 40], None] for q in range(len(y))]).flatten()
                            ], "lines", "Springs", "#D7101F", {"line": {"width": 2, "dash": 'dash'}}, True],
                            [[
                                x_base,
                                y_base,
                            ], "lines+markers", "Input", "#D08D00", {}, True],
                            [[
                                D_j_coreg[0],
                                D_j_coreg[1]
                            ], "lines+markers", "New iter", "#015AC8", {}, True],
                            [[
                                D_j_coreg[0],
                                D_j_coreg[1]
                            ], "lines", "New iter", "#015AC8", {}, True],
                            # [[
                            #     D_j_coreg[0, ::parts][force_indexes],
                            #     D_j_coreg[1, ::parts][force_indexes]
                            # ], "lines", "Force_compare_pints", "green", {}, True],
                            [[
                                np.array([x_base[iiii] if iiii is not None else None for iiii in force_indexes]),
                                np.array([y_base[iiii] if iiii is not None else None for iiii in force_indexes])
                            ], "lines+markers", "Force compare pints", "green", {}, True],
                            [[
                                x_base[top_4_candidates],
                                y_base[top_4_candidates],
                            ], "markers", "Corner pints", "purple", {"marker": {"size": 10}}, True],
                        ],
                        equal=equal if equal is not None else True,
                        save_path=f"{MATERIALS_PATH}smooth_contour/{file_name.rsplit('_', 1)[0]}/plots/d_{general_l}/{puzzle_index}/{direction}/",
                        filename=f"after_{iteration + 1}_iter_force_points{('_'+str(corner_move)) if iteration == (iterations - 1) else ''}",
                    )

                if len(display_corner_points) > 0:
                    print(display_corner_points, real_corner_point)
                    for lll in display_corner_points:
                        force, _, _ = find_force(lll, P_align_coef, C, S_input, start_L * 3 / 4, imagine=True)
                        print("Korner Робота", force)

                    if real_corner_point:
                        vallll = int(real_corner_point)
                        # force = sum(
                        #     np.array(P_align_coef)[vallll - 20: vallll + 21] ** 2 * C[vallll - 20: vallll + 21]) - \
                        #         P_align_coef[vallll] ** 2 * C[vallll]

                        force, _, _ = find_force(vallll, P_align_coef, C, S_input, start_L * 3 / 4, imagine=True)
                        print("Real Робота", force)

                    force_full = sum(np.array(P_align_coef) ** 2 * C) - sum(np.array(P_align_coef)[display_corner_points] ** 2 * C[display_corner_points])
                    print("Робота повна", force_full)
                    print("Кути рад", psis[display_corner_points - 1])
                    print("Кути", np.degrees(psis[display_corner_points - 1]))

                    if iteration == (iterations - 1):
                        continue


            # if curve_type == "loop":  # and iteration == 0:
            if curve_type == "not_loop":  # and iteration == 0:
                if iteration == 0:

                    start_1_iteration = time.time()

                    new_x_list = []
                    new_y_list = []
                    used_points = set()
                    D_j_coreg_len = D_j_coreg.shape[1]

                    for index in range(len(x_base)):
                        if index % skip == 0 or "test_data" in file_name:
                            i = (index * (parts // skip)) % D_j_coreg_len
                            point = (D_j_coreg[0, i], D_j_coreg[1, i])
                        else:
                            current_point = np.array([x_base[index], y_base[index]])
                            start = (index * (parts // skip) - gate) % D_j_coreg_len
                            end = (index * (parts // skip) + gate) % D_j_coreg_len

                            if start < end:
                                interval = D_j_coreg[:, start:end]
                            else:
                                interval = np.hstack((D_j_coreg[:, start:], D_j_coreg[:, :end]))

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
                                [
                                    [x_base, y_base],
                                    "lines+markers", "All input", "#349950", {"line": {"width": 2}}, True],
                                [D_j_coreg, "markers", "All new points", "#015AC8", {}, True],
                                [
                                    [x_with_skipped, y_with_skipped],
                                    "markers", "Bace new points", "#C2A4FF", {}, True],
                            ],
                            equal=equal if equal is not None else True,
                            save_path=f"{MATERIALS_PATH}smooth_contour/{file_name.rsplit('_', 1)[0]}/plots/d_{general_l}/{puzzle_index}/{direction}/",
                            filename="after_1_iter_find_skipped"
                        )

                    x_near, y_near, indexes, old_positions, new_positions = find_near_point(x_base, y_base, x_with_skipped, y_with_skipped, D_j_coreg, curve_type)

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
                                [new_positions.T, "markers", "Pont new position", "#D7101F", {}, True],
                                [old_positions.T, "markers", "Pont old position", "#D08D00", {}, True],
                            ],
                            equal=equal if equal is not None else True,
                            save_path=f"{MATERIALS_PATH}smooth_contour/{file_name.rsplit('_', 1)[0]}/plots/d_{general_l}/{puzzle_index}/{direction}/",
                            filename="after_1_iter_find_new_near_points"
                        )

                    x, y, x_base, y_base, _, _ = order_points(D_j_coreg[0], D_j_coreg[1], x_near, y_near, x_base, y_base, curve_type=curve_type)

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
                            equal=equal if equal is not None else True,
                            save_path=f"{MATERIALS_PATH}smooth_contour/{file_name.rsplit('_', 1)[0]}/plots/d_{general_l}/{puzzle_index}/{direction}/",
                            filename=f"after_1_iter_after_ordering"
                        )

                    order_points_time = time.time()
                    print(f"Время выполнения (order_points_time): {order_points_time - find_near_point_time:.4f} секунд")

                    parts = 40
                    list_of_patrs = [i / parts for i in range(1, parts)]

                    # L = L / scale_coef
                    # d_4 = L ** 4

                else:
                    start_n_iteration = time.time()

                    x_ooold = x
                    y_ooold = y

                    x = D_j_coreg[0, ::parts]
                    y = D_j_coreg[1, ::parts]

                    D_j_coreg_without_corners = np.copy(D_j_coreg)

                    if len(display_corner_points) > 0:
                        print(f"ggggg {iteration=}")
                        x, y, D_j_coreg_without_corners, x_base, y_base, point_type, display_corner_points_positions = delete_corner_points(x, y, x_base, y_base, D_j_coreg.T, point_type, display_corner_points)

                        if SHOW_NEW_TYPE_PLOTS:
                            display_plot_plotly(
                                [
                                    [D_j_coreg_without_corners, "lines+markers", "New iter", "#015AC8", {}, True],
                                ],
                                equal=equal if equal is not None else True,
                                save_path=f"{MATERIALS_PATH}smooth_contour/{file_name.rsplit('_', 1)[0]}/plots/d_{general_l}/{puzzle_index}/{direction}/",
                                filename=f"after_{iteration + 1}_without_imagine_points",
                            )

                        if iteration == 12:
                            print("removed extra corner points")
                            display_corner_points_positions = display_corner_points_positions[2::5]

                    find_near_point_start = time.time()
                    x, y, indexes, old_positions, new_positions = find_near_point(x_base, y_base, x, y, D_j_coreg_without_corners, curve_type)

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
                                [
                                    D_j_coreg_without_corners, "lines+markers", "D_j_coreg", "pink", {}, True
                                ],
                            ],
                            equal=equal if equal is not None else True,
                            save_path=f"{MATERIALS_PATH}smooth_contour/{file_name.rsplit('_', 1)[0]}/plots/d_{general_l}/{puzzle_index}/{direction}/",
                            filename=f"after_{iteration+1}_iter_find_new_near_points",
                        )

                    find_near_point_end = time.time()
                    print(f"Время выполнения (find_near_point_end): {find_near_point_end - find_near_point_start:.4f} секунд")


                    # x, y, x_base, y_base, point_type, display_corner_points = near_find_check(x, y, D_j_coreg[0], D_j_coreg[1], x_base, y_base, point_type, display_corner_points)
                    # if SHOW_NEW_TYPE_PLOTS:
                    #     display_plot_plotly(
                    #         [
                    #             [[
                    #                 np.array([[x_base[q], x[q], None] for q in range(len(x))]).flatten(),
                    #                 np.array([[y_base[q], y[q], None] for q in range(len(y))]).flatten()
                    #             ], "lines", "Springs", "#D7101F", {"line": {"width": 2, "dash": 'dash'}}, True],
                    #
                    #             [[
                    #                 x_ooold,
                    #                 y_ooold
                    #             ], "lines+markers", "Prew iter", "#349950", {"line": {"width": 3}}, True],
                    #             [[
                    #                 x_base,
                    #                 y_base,
                    #             ], "lines+markers", "Input", "#D08D00", {}, True],
                    #             [[
                    #                 x_base,
                    #                 y_base,
                    #             ], "lines", "Input", "#D08D00", {}, True],
                    #             [[
                    #                 x,
                    #                 y
                    #             ], "lines+markers", "New iter", "#015AC8", {}, True],
                    #             [[
                    #                 x,
                    #                 y
                    #             ], "lines", "New iter", "#015AC8", {}, True],
                    #             [
                    #                 D_j_coreg, "lines+markers", "D_j_coreg", "pink", {}, True
                    #             ],
                    #         ],
                    #         equal=True,
                    #         save_path=f"{MATERIALS_PATH}smooth_contour/{file_name.rsplit('_', 1)[0]}/plots/d_{general_l}/{puzzle_index}/{direction}/",
                    #         filename=f"after_{iteration+1}_iter_find_new_near_points_after_corner_fix",
                    #     )


                    # if len(display_corner_points):
                    #     display_point_positions = np.vstack((x[display_corner_points - 1], y[display_corner_points - 1])).T
                    #     for cp in display_corner_points[::-1]:
                    #         x = np.delete(x, cp)
                    #         y = np.delete(y, cp)
                    #         x_base = np.delete(x_base, cp)
                    #         y_base = np.delete(y_base, cp)
                    #         point_type = np.delete(point_type, cp)

                    order_points_start = time.time()
                    x, y, x_base, y_base, remove_pints, new_points = order_points(D_j_coreg_without_corners[0], D_j_coreg_without_corners[1], x, y, x_base, y_base, curve_type=curve_type)

                    if len(remove_pints) > 0:
                        for ze in range(len(remove_pints)):
                            additional_move = 1
                            print("add move:", additional_move)

                            x_base = np.delete(x_base, remove_pints[ze])
                            x_base = insert_corner_points(x_base, new_points[ze] - additional_move)

                            y_base = np.delete(y_base, remove_pints[ze])
                            y_base = insert_corner_points(y_base, new_points[ze] - additional_move)

                            point_type = np.delete(point_type, remove_pints[ze])
                            point_type = insert_corner_points(point_type, new_points[ze] - additional_move, 3)

                            inndddd = np.where(display_corner_points == remove_pints[ze])[0][0]
                            display_corner_points[inndddd] = new_points[ze]

                    if len(display_corner_points) > 0:
                        x, y, x_base, y_base, point_type, display_corner_points = insert_new_corner_points(D_j_coreg[0], D_j_coreg[1], x, y, x_base, y_base, point_type, display_corner_points_positions)

                    order_points_end = time.time()
                    print(f"Время выполнения (order_points): {order_points_end - order_points_start:.4f} секунд")

                    if len(display_point_positions):
                        new_display_corner_points = np.array([ind for ind, ppp in enumerate(np.vstack((x, y)).T) if ppp in display_point_positions])
                        print(f"{display_corner_points=}, {new_display_corner_points=}")

                        display_corner_points = new_display_corner_points

                        for i, point in enumerate(display_corner_points[::-1]):
                            x = insert_corner_points(x, int(point))
                            y = insert_corner_points(y, int(point))
                            x_base = insert_corner_points(x_base, int(point))
                            y_base = insert_corner_points(y_base, int(point))

                            point_type = insert_corner_points(point_type, int(point) - 1, 3)

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
                                [[
                                    x[display_corner_points.astype(int)],
                                    y[display_corner_points.astype(int)]
                                ], "markers", "Imagine corners", "black", {}, True],
                            ],
                            equal=equal if equal is not None else True,
                            save_path=f"{MATERIALS_PATH}smooth_contour/{file_name.rsplit('_', 1)[0]}/plots/d_{general_l}/{puzzle_index}/{direction}/",
                            filename=f"after_{iteration+1}_iter_find_missing_spring",
                            background_image = f"/Users/dmyrto_koltsov/PycharmProjects/PDF/my_data/{file_name}.jpg"
                        )

                    order_points_n_time = time.time()
                    print(f"Время выполнения (general_n_time): {order_points_n_time - start_n_iteration:.4f} секунд")

                    if iteration == 9:
                        top_4_candidates = get_corner_points_candidate(M_j, D_j_coreg, straight, general_l, puzzle_index, file_name, full=True)

                        display_plot_plotly(
                            [
                                [
                                    np.vstack((D_j_coreg[0][top_4_candidates], D_j_coreg[1][top_4_candidates])),
                                    "lines+markers", "top 4", "#FF4500", {}, True
                                ],
                                [
                                    D_j_coreg,
                                    "lines", "all", "#054907", {}, True
                                ],
                            ],
                            equal=equal if equal is not None else True,
                            save_path=f"{MATERIALS_PATH}smooth_contour/{file_name.rsplit('_', 1)[0]}/plots/d_{general_l}/{puzzle_index}/{direction}/",
                            filename="top_points_candidats_on_contur"
                        )

                        top_4_candidates = top_4_candidates / 40
                        print(f"{top_4_candidates=}")

                        if (len(x) - 1) in top_4_candidates:
                            top_4_candidates = np.delete(top_4_candidates, 3)
                            top_4_candidates = np.insert(top_4_candidates, 0, 0)

                        # if "test_data" in file_name:
                        if "test_data" in file_name or "all_good" in file_name:
                            real_corner_point = top_4_candidates[1] + 2
                            top_4_candidates[1] = top_4_candidates[1] + corner_move

                        for i, point in enumerate(top_4_candidates[::-1]):
                            x = insert_corner_points(x, int(point))
                            y = insert_corner_points(y, int(point))
                            x_base = insert_corner_points(x_base, int(point))
                            y_base = insert_corner_points(y_base, int(point))

                            point_type = insert_corner_points(point_type, int(point) - 1, 3)

                        display_corner_points = np.where(point_type == 3)[0]

                        display_corner_points = display_corner_points + 1


                        # display_corner_points = np.array([cor + i - 1 for i, cor in enumerate(top_4_candidates)])
                        # minus_index = np.where(display_corner_points == -1)[0]
                        # if len(minus_index) > 0:
                        #     display_corner_points[minus_index[0]] = len(x_base) - 2
                        print(f"{display_corner_points=}")

                        if SHOW_NEW_TYPE_PLOTS:
                            display_plot_plotly(
                                [
                                    [[
                                        np.array([[x_base[q], x[q], None] for q in range(len(x_base))]).flatten(),
                                        np.array([[y_base[q], y[q], None] for q in range(len(y_base))]).flatten()
                                    ], "lines", "Springs", "#D7101F", {"line": {"width": 2, "dash": 'dash'}}, True],
                                    [[
                                        x_base,
                                        y_base,
                                    ], "lines+markers", "Input", "#D08D00", {}, True],
                                    [[
                                        x,
                                        y
                                    ], "lines+markers", "New iter", "#015AC8", {}, True],
                                    [[
                                        x[display_corner_points.astype(int)],
                                        y[display_corner_points.astype(int)]
                                    ], "markers", "Imagine corners", "black", {}, True],
                                ],
                                equal=equal if equal is not None else True,
                                save_path=f"{MATERIALS_PATH}smooth_contour/{file_name.rsplit('_', 1)[0]}/plots/d_{general_l}/{puzzle_index}/{direction}/",
                                filename=f"after_{iteration + 1}_iter_with_corners_points",
                                # background_image = f"/Users/dmyrto_koltsov/PycharmProjects/PDF/my_data/{file_name}.jpg"
                            )

                    if iteration > 0:
                        if iteration in [1, 2]:
                            pass
                        elif iteration < 5:
                            L = L / scale_coef
                        else:
                            L = general_l

                        d_4 = L ** 4


                        # if L / scale_coef < general_l and iteration > 4:
                        #     L = general_l
                        # else:
                        #     L = L / scale_coef
                        #
                        # d_4 = L ** 4

                file_dataset_len = len(x_base) - 1
            else:
                pass



        return {"plots": response_images, "S": M_j[0, ::40].tolist()}

    def _get(self):
        return render_template("main_page.html")
