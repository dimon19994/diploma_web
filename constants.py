#--set_input_vaiable--
# Значення жорсткості та згину
E, I = 1.0, 1.0


iterations = 30
# minus_list = [-2, -3, -4, -5, -6,]
# added_points_iter = 3
# revers_iter = 14
LANGUAGE = "en"

#----display_config----
# table_display_round = 4
PLOT_DATA_ROUND = 4
PLOT_DISPLAY_SIZE = (10, 10)
PLOT_MARKET_SIZE = 8
PLOT_LINE_WIDTH = 2
PLOT_LEGEND_FONT_SIZE = 20
PLOT_TITLE_FONT_SIZE = 20
PLOT_ANOTATE_FONT_SIZE = 15
PLOT_ASIX_FONT_SIZE = 10
DPI_VALIE = 75
#----/display_config----


#----C_coef----
C_start = 100
C_end = 1/30
C_last = 10
if iterations > 1:
    # C_step = (C_end/C_start)**(1/(iterations-1))
    C_step = (C_end/C_start)**(1/(iterations-1))
#----/C_coef----



parts = 50
list_of_patrs = [i/parts for i in range(1, parts)]
#--/set_input_vaiable-------------


display_aligns_table = 0
display_solution_table = 0
display_iteration_info = 1
display_result_plot = 1
display_moment_plot = 1

show_opor = 1
show_res_dots = 0
# if file_number == -7:
#     if fig == "elips":
#         axis_elips = 1
# else:
axis_elips = 0


POINTS_TYPE = {
    0: "movable",
    1: "fixed",
    2: "imagine"
}