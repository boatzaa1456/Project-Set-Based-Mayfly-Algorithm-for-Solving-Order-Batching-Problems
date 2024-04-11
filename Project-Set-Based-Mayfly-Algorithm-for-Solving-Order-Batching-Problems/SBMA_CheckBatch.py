import random
import time
import numpy as np
import matplotlib.pyplot as plt
from evaluate_all_sols_check import *
from evaluate_all_sols import *
import itertools
import pandas as pd
import concurrent.futures
import copy
import math
from numpy.polynomial.polynomial import Polynomial
from evaluate_all_sols import *
from SB_SupportFunction import *
from SBMA_Fucntion import *
# from numba import jit, cuda
value_heavy = 40

def mayfly(name_path_input, num_gen, pop_size, *parameters):
    a1, a2, a3, gmax, gmin, alpha, seed = parameters
    random.seed(seed)
    df_item_pool, df_item_sas_random = read_input(name_path_input)
    num_item = df_item_pool.shape[0]
    half_pop_size = pop_size // 2
    E_all = list(itertools.permutations(range(num_item), 2))
    sub_E_list = [[arc for arc in E_all if arc[0] == item or arc[1] == item] for item in range(num_item)]
    # สร้าง set ของ item ที่ถือว่าเป็น item หนัก
    num_item = len(df_item_pool)
    heavy_item_set = set(df_item_pool[df_item_pool['weight'] >= value_heavy].index)
    # Initialize gbest and pbest values for males and females
    gbest_value, gbest_sol, gbest_arc_sol_cut = 100000, [], []
    male_pbest_value, female_pbest_value = [100000] * half_pop_size, [100000] * half_pop_size
    male_pbest_sol, female_pbest_sol = [[] for _ in range(half_pop_size)], [[] for _ in range(half_pop_size)]
    male_pbest_arc_sols, female_pbest_arc_sols = [[] for _ in range(half_pop_size)], [[] for _ in range(half_pop_size)]
    male_pbest_arc_sols_cut, female_pbest_arc_sols_cut = [[] for _ in range(half_pop_size)], [[] for _ in range(half_pop_size)]

    def process_mayfly_population(population):
        evaluations = []
        cur_sols = []
        cur_sols_value = []
        cur_arc_sols = []
        arc_sols_cut = []
        velocity_dict = []

        for mayfly in population:
            random.shuffle(mayfly)
            evaluation = evaluate_all_sols(mayfly, df_item_pool, heavy_item_set, name_path_input)
            evaluations.append(evaluation)
            cur_sols.append(extract_and_flatten(evaluation))
            cur_sols_value.append(evaluation[1])
            cur_arc_sols.append(sol_from_list_to_arc(cur_sols[-1]))
            arc_sols_cut.append(cut_arc_sol(cur_arc_sols[-1]))
            velocity_dict.append(init_velocity_sol(arc_sols_cut[-1]))

        return evaluations, cur_sols, cur_sols_value, cur_arc_sols, arc_sols_cut, velocity_dict

    # Initialize male and female populations
    male_mayfly_population = [[i for i in range(num_item)] for _ in range(half_pop_size)]
    female_mayfly_population = [[i for i in range(num_item)] for _ in range(half_pop_size)]

    # Parallel processing for male and female populations
    with concurrent.futures.ThreadPoolExecutor() as executor:
        male_future = executor.submit(process_mayfly_population, male_mayfly_population)
        male_results = male_future.result()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        female_future = executor.submit(process_mayfly_population, female_mayfly_population)
        female_results = female_future.result()

    gbest_each_gen = []
    male_each_gen = []
    female_each_gen = []
    female_attractor = []
    male_evaluations, male_cur_sols, male_cur_sols_value, male_cur_arc_sols, male_arc_sols_cut, male_velocity_dict = male_results
    female_evaluations, female_cur_sols, female_cur_sols_value, female_cur_arc_sols, female_arc_sols_cut, female_velocity_dict = female_results

    for gen in range(num_gen):
        # ---------------------- Mayfly Male Section ----------------------
        for sol in range(half_pop_size):
            male_current_value = male_cur_sols_value[sol]
            male_current_sol = male_cur_sols[sol]
            male_current_arc_sol = sol_from_list_to_arc(male_current_sol)
            male_current_arc_sol_cut = cut_arc_sol(male_current_arc_sol)
            # Update personal best if current solution is better or equal
            better_than_gbest = False
            if male_current_value <= male_pbest_value[sol]:
                male_pbest_value[sol] = male_current_value
                male_pbest_sol[sol] = male_current_sol[:]
                male_pbest_arc_sols[sol] = [tuple(item) for item in male_current_arc_sol]
                male_pbest_arc_sols_cut[sol] = [tuple(item) for item in male_current_arc_sol_cut]

                # Update global best if current personal best is better
                if male_pbest_value[sol] <= gbest_value:
                    gbest_value = male_pbest_value[sol]
                    gbest_sol = male_pbest_sol[sol]
                    gbest_arc_sol_cut = male_pbest_arc_sols_cut[sol]
                    mnew_pos = nuptial_dance(male_cur_sols[sol][:], female_attractor)
                    mevaluation = evaluate_all_sols(mnew_pos, df_item_pool, heavy_item_set, name_path_input)
                    male_cur_sols_value[sol] = mevaluation[1]
                    male_cur_sols[sol] = mnew_pos[:]
                    male_cur_arc_sols[sol] = sol_from_list_to_arc(mnew_pos[:])
                    male_arc_sols_cut[sol] = cut_arc_sol([tuple(item) for item in male_cur_arc_sols[sol]])
                    male_velocity_dict[sol] = init_velocity_sol([tuple(item) for item in male_arc_sols_cut[sol]])
                    male_each_gen.append(mevaluation[1])
                    better_than_gbest = True

            if not better_than_gbest:
                mcoef_velocity = coef_times_velocity(gravity_calculation(gmax, gmin, gen, num_gen), male_velocity_dict[sol])
                mpbest_diff = position_minus_position(male_pbest_arc_sols_cut[sol], male_arc_sols_cut[sol])
                mgbest_diff = position_minus_position(gbest_arc_sol_cut, male_arc_sols_cut[sol])
                coef_pbest_diff = coef_times_position(a1, mpbest_diff)
                coef_gbest_diff = coef_times_position(a2, mgbest_diff)
                madded_best_diff = add_velocity(coef_gbest_diff, coef_pbest_diff)
                madded_velocity = add_velocity(mcoef_velocity, madded_best_diff)
                mvelocity_check_incon = check_velocity_inconsistency(madded_velocity)
                mcut_set = creat_cut_set(mvelocity_check_incon, alpha)
                mnew_pos = sol_position_update(mcut_set, male_arc_sols_cut[sol], sub_E_list, male_cur_sols[sol][0],male_pbest_sol[sol][0], gbest_sol[0])[0]
                mevaluation = evaluate_all_sols(mnew_pos, df_item_pool, heavy_item_set, name_path_input)
                male_cur_sols_value[sol] = mevaluation[1]
                male_cur_sols[sol] = mnew_pos[:]
                male_velocity_dict[sol] = mvelocity_check_incon
                male_each_gen.append(mevaluation[1])


        # ---------------------- Female Mayfly Section ----------------------
            female_current_value = female_cur_sols_value[sol]
            female_current_sol = female_cur_sols[sol]
            female_current_arc_sol = sol_from_list_to_arc(female_current_sol)
            female_current_arc_sol_cut = cut_arc_sol(female_current_arc_sol)
            # Update personal best if current solution is better or equal
            better_than_gbest = False
            if female_current_value <= female_pbest_value[sol]:
                female_pbest_value[sol] = female_current_value
                female_pbest_sol[sol] = female_current_sol[:]
                female_pbest_arc_sols[sol] = [tuple(item) for item in female_current_arc_sol]
                female_pbest_arc_sols_cut[sol] = [tuple(item) for item in female_current_arc_sol_cut]
                female_attractor = female_pbest_sol[sol]

                # Update global best if current personal best is better
                if female_pbest_value[sol] <= gbest_value:
                    gbest_value = female_pbest_value[sol]
                    gbest_sol = female_pbest_sol[sol]
                    gbest_arc_sol_cut = female_pbest_arc_sols_cut[sol]
                    fnew_pos = random_flight(female_cur_sols[sol][:])
                    fevaluation = evaluate_all_sols(fnew_pos, df_item_pool, heavy_item_set, name_path_input)
                    female_cur_sols_value[sol] = fevaluation[1]
                    female_cur_sols[sol] = fnew_pos[:]
                    female_cur_arc_sols[sol] = sol_from_list_to_arc(fnew_pos[:])
                    female_arc_sols_cut[sol] = cut_arc_sol([tuple(item) for item in female_cur_arc_sols[sol]])
                    female_velocity_dict[sol] = init_velocity_sol([tuple(item) for item in female_arc_sols_cut[sol]])
                    female_each_gen.append(fevaluation[1])
                    better_than_gbest = True

            if not better_than_gbest:
                fcoef_velocity = coef_times_velocity(gravity_calculation(gmax, gmin, gen, num_gen),female_velocity_dict[sol])
                female_diff = position_minus_position(male_arc_sols_cut[sol], female_arc_sols_cut[sol])
                coef_female_diff = coef_times_position(a3, female_diff)
                fadded_velocity = add_velocity(fcoef_velocity, coef_female_diff)
                fvelocity_check_incon = check_velocity_inconsistency(fadded_velocity)
                fcut_set = creat_cut_set(fvelocity_check_incon, alpha)
                fnew_pos = sol_position_update(fcut_set, female_arc_sols_cut[sol], sub_E_list, female_cur_sols[sol][0],female_pbest_sol[sol][0], gbest_sol[0])[0]
                fevaluation = evaluate_all_sols(fnew_pos, df_item_pool, heavy_item_set, name_path_input)


                female_cur_sols_value[sol] = fevaluation[1]
                female_cur_sols[sol] = fnew_pos[:]
                female_velocity_dict[sol] = fvelocity_check_incon
                female_each_gen.append(fevaluation[1])

        gbest_each_gen.append(gbest_value)
    best_solution = evaluate_all_sols_check(gbest_sol, df_item_pool, heavy_item_set, name_path_input)
    return min(gbest_each_gen),best_solution,gbest_each_gen


# start_time = time.time()
# num_gen = 10
# pop_size = 4
# a1 = 0.5
# a2 = 2
# a3 = 1
# gmax = 0.9
# gmin = 0.3
# alpha = 0.5
# random_seed = 9999
#
# gbest_each_gen = []
# male_each_gen = []
# female_each_gen = []
#
# name_path_input = '1R-50I-200C-2P'
# df_item_pool = read_input(name_path_input)
# gbest_each_gen, male_each_gen, female_each_gen,best_solution = mayfly(name_path_input, num_gen, pop_size, a1, a2, a3, gmax, gmin,
#                                 alpha, random_seed)
# male_each_gen = chunk_list(male_each_gen, pop_size // 2)
# female_each_gen = chunk_list(female_each_gen, pop_size // 2)
# # End the timer
# end_time = time.time()
# time_taken = end_time - start_time
#
# # Convert time_taken to hours, minutes, and seconds
# hours = int(time_taken // 3600)
# minutes = int((time_taken % 3600) // 60)
# seconds = time_taken % 60
# gbest_value = min(gbest_each_gen)
# gbest_each_gen = gbest_each_gen[:num_gen]
#
# # Display final results
# hours, remainder = divmod(time_taken, 3600)
# minutes, seconds = divmod(remainder, 60)
# print(f"Time Taken: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")
# print(f"Time Taken (second) : {time_taken:.2f}")
# print("----" * 50)
# #
# # # คำนวณค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน
# # average_male = [np.mean(gen) for gen in male_each_gen]
# # average_female = [np.mean(gen) for gen in female_each_gen]
# # std_deviation_male = [np.std(gen) for gen in male_each_gen]
# # std_deviation_female = [np.std(gen) for gen in female_each_gen]
# rounds = np.arange(1, len(gbest_each_gen) + 1)
# #
# # # ใช้ Polynomial regression สำหรับเส้นแนวโน้ม
# # p_male = Polynomial.fit(rounds - 1, average_male, deg=3)
# # p_female = Polynomial.fit(rounds - 1, average_female, deg=3)
# #
# # # สร้างข้อมูลสำหรับเส้นแนวโน้ม
# # x_new = np.linspace(0, len(male_each_gen) - 1, num=len(male_each_gen))
# # y_new_male = p_male(x_new)
# # y_new_female = p_female(x_new)
# #
# plt.figure(figsize=(16, 8))
# #
# # วาดกราฟข้อมูล
# plt.plot(rounds, gbest_each_gen, '-o', color='red', label='GBest Value', markersize=4, linewidth=1.5)
# #
# # # วาดกราฟข้อมูล
# # plt.plot(rounds, average_male, '-^', color='blue', label='Average Male Value', markersize=4, linewidth=1.5)
# # plt.plot(rounds, average_female, '-s', color='green', label='Average Female Value', markersize=4, linewidth=1.5)
# #
# # # วาดเส้นแนวโน้ม
# # plt.plot(x_new + 1, y_new_male, 'b--', label='Male Trendline')
# # plt.plot(x_new + 1, y_new_female, 'g--', label='Female Trendline')
# #
# # #แสดงความเคลื่อนไหวด้วยส่วนเบี่ยงเบนมาตรฐาน
# # plt.fill_between(rounds, np.array(average_male) - np.array(std_deviation_male),
# #                  np.array(average_male) + np.array(std_deviation_male), color='blue', alpha=0.2,
# #                  label='Std.Deviation of Male Value')
# # plt.fill_between(rounds, np.array(average_female) - np.array(std_deviation_female),
# #                  np.array(average_female) + np.array(std_deviation_female), color='green', alpha=0.2,
# #                  label='Std.Deviation of Female Value')
# #
# plt.xlabel(f"Time Taken: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")
# plt.ylabel('Values')
# plt.title(
#     f'{name_path_input} - {pop_size} Population Size - {num_gen} Generations - Seed {random_seed} - Tadiness: {gbest_value}')
#
# # Add the legend
# plt.legend()
# plt.tight_layout()
#
# # Create an inset in the plot for parameter descriptions
# param_descriptions = (
#     "Parameters:\n"
#     f"a1 = {a1}\n"
#     f"a2 = {a2}\n"
#     f"a3 = {a3}\n"
#     f"gmax= {gmax}\n"
#     f"gmin= {gmin}\n"
#     f"alpha = {alpha}\n"
#     # f"sub_size = {sub_size}\n"
#     # f"mutation_rate = {mutation_rate}"
# )
# # Position the text box in figure coords, and set the box style
# text_box = plt.text(0.16, 0.04, param_descriptions, transform=plt.gcf().transFigure, fontsize=11,
#                     verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
#
# plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9)
#
# # Show the plot with the parameter descriptions
# plt.show()
#
