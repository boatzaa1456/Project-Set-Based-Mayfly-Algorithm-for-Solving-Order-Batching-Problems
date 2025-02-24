import concurrent.futures
import itertools
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from evaluate_all_sols import *
from evaluate_all_sols_check import *
from SB_SupportFunction import *
from SBMA_Fucntion import *

value_heavy = 40


def mayfly(name_path_input, num_gen, pop_size, *parameters):
    # parameters: a1, a2, a3, gmax, gmin, alpha, seed
    a1, a2, a3, gmax, gmin, alpha, seed = parameters
    random.seed(seed)

    # อ่านข้อมูล input เพียงครั้งเดียว
    df_item_pool, df_item_sas_random = read_input(name_path_input)
    num_item = df_item_pool.shape[0]
    half_pop_size = pop_size // 2

    # แทนที่จะสร้าง E_all แบบเต็มและ filter ทีละ item (ซึ่งใช้เวลา O(n^3))
    # เราสามารถคำนวณ sub_E_list โดยตรงด้วย list comprehension:
    sub_E_list = [
        [(i, j) for j in range(num_item) if j != i] + [(j, i) for j in range(num_item) if j != i]
        for i in range(num_item)
    ]

    # สร้าง set ของ item ที่ถือว่าเป็นสินค้าหนัก
    heavy_item_set = set(df_item_pool[df_item_pool['weight'] >= value_heavy].index)

    # Initialize global best (gbest) และ personal best (pbest) สำหรับ male และ female
    gbest_value, gbest_sol, gbest_arc_sol_cut = 100000, [], []
    male_pbest_value = [100000] * half_pop_size
    female_pbest_value = [100000] * half_pop_size
    male_pbest_sol = [[] for _ in range(half_pop_size)]
    female_pbest_sol = [[] for _ in range(half_pop_size)]
    male_pbest_arc_sols = [[] for _ in range(half_pop_size)]
    female_pbest_arc_sols = [[] for _ in range(half_pop_size)]
    male_pbest_arc_sols_cut = [[] for _ in range(half_pop_size)]
    female_pbest_arc_sols_cut = [[] for _ in range(half_pop_size)]
    female_attractor = []

    # ฟังก์ชันช่วยประมวลผล population (male หรือ female)
    def process_mayfly_population(population):
        # ใช้ list comprehensions เพื่อสร้างผลลัพธ์ในแต่ละ mayfly
        evaluations = []
        cur_sols = []
        cur_sols_value = []
        cur_arc_sols = []
        arc_sols_cut = []
        velocity_dict = []
        for mayfly in population:
            random.shuffle(mayfly)
            evaluation = evaluate_all_sols_check(mayfly, df_item_pool, heavy_item_set, name_path_input)
            evaluations.append(evaluation)
            # extract_and_flatten, sol_from_list_to_arc และ cut_arc_sol อยู่ใน SB_SupportFunction
            flat_sol = extract_and_flatten(evaluation)
            cur_sols.append(flat_sol)
            cur_sols_value.append(evaluation[1])
            arc_sol = sol_from_list_to_arc(flat_sol)
            cur_arc_sols.append(arc_sol)
            arc_sol_cut = cut_arc_sol(arc_sol)
            arc_sols_cut.append(arc_sol_cut)
            velocity_dict.append(init_velocity_sol(arc_sol_cut))
        return evaluations, cur_sols, cur_sols_value, cur_arc_sols, arc_sols_cut, velocity_dict

    # Initialize male and female populations (แต่ละ mayfly คือ list [0,1,...,num_item-1])
    male_mayfly_population = [[i for i in range(num_item)] for _ in range(half_pop_size)]
    female_mayfly_population = [[i for i in range(num_item)] for _ in range(half_pop_size)]

    # ใช้ executor เดียวในการประมวลผล male และ female population พร้อมกัน
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_mayfly_population, pop): label
            for pop, label in [(male_mayfly_population, 'male'), (female_mayfly_population, 'female')]
        }
        results = {}
        for fut in concurrent.futures.as_completed(futures):
            results[futures[fut]] = fut.result()
    male_results = results['male']
    female_results = results['female']

    gbest_each_gen = []
    male_evaluations, male_cur_sols, male_cur_sols_value, male_cur_arc_sols, male_arc_sols_cut, male_velocity_dict = male_results
    female_evaluations, female_cur_sols, female_cur_sols_value, female_cur_arc_sols, female_arc_sols_cut, female_velocity_dict = female_results

    # เริ่มต้นวนลูป generation
    for gen in range(num_gen):
        # ---------------------- Male Mayfly Section ----------------------
        for sol in range(half_pop_size):
            m_val = male_cur_sols_value[sol]
            m_sol = male_cur_sols[sol]
            m_arc_sol = sol_from_list_to_arc(m_sol)
            m_arc_sol_cut = cut_arc_sol(m_arc_sol)
            better_than_gbest = False
            if m_val <= male_pbest_value[sol]:
                male_pbest_value[sol] = m_val
                male_pbest_sol[sol] = m_sol[:]
                male_pbest_arc_sols[sol] = [tuple(x) for x in m_arc_sol]
                male_pbest_arc_sols_cut[sol] = [tuple(x) for x in m_arc_sol_cut]
                if m_val <= gbest_value:
                    gbest_value = m_val
                    gbest_sol = m_sol[:]
                    gbest_arc_sol_cut = male_pbest_arc_sols_cut[sol]
                    # Nuptial dance update for male
                    mnew_pos = nuptial_dance(m_sol[:], female_attractor)
                    mevaluation = evaluate_all_sols_check(mnew_pos, df_item_pool, heavy_item_set, name_path_input)
                    male_cur_sols_value[sol] = mevaluation[1]
                    male_cur_sols[sol] = mnew_pos[:]
                    male_cur_arc_sols[sol] = sol_from_list_to_arc(mnew_pos[:])
                    male_arc_sols_cut[sol] = cut_arc_sol([tuple(x) for x in male_cur_arc_sols[sol]])
                    male_velocity_dict[sol] = init_velocity_sol([tuple(x) for x in male_arc_sols_cut[sol]])
                    better_than_gbest = True

            if not better_than_gbest:
                mcoef_velocity = coef_times_velocity(gravity_calculation(gmax, gmin, gen, num_gen),
                                                     male_velocity_dict[sol])
                mpbest_diff = position_minus_position(male_pbest_arc_sols_cut[sol], male_arc_sols_cut[sol])
                mgbest_diff = position_minus_position(gbest_arc_sol_cut, male_arc_sols_cut[sol])
                coef_pbest_diff = coef_times_position(a1, mpbest_diff)
                coef_gbest_diff = coef_times_position(a2, mgbest_diff)
                madded_best_diff = add_velocity(coef_gbest_diff, coef_pbest_diff)
                madded_velocity = add_velocity(mcoef_velocity, madded_best_diff)
                mvelocity_check = check_velocity_inconsistency(madded_velocity)
                mcut_set = creat_cut_set(mvelocity_check, alpha)
                mnew_pos = sol_position_update(mcut_set, male_arc_sols_cut[sol], sub_E_list,
                                               male_cur_sols[sol][0],
                                               male_pbest_sol[sol][0],
                                               gbest_sol[0])[0]
                mevaluation = evaluate_all_sols_check(mnew_pos, df_item_pool, heavy_item_set, name_path_input)
                male_cur_sols_value[sol] = mevaluation[1]
                male_cur_sols[sol] = mnew_pos[:]
                male_velocity_dict[sol] = mvelocity_check

        # ---------------------- Female Mayfly Section ----------------------
        for sol in range(half_pop_size):
            f_val = female_cur_sols_value[sol]
            f_sol = female_cur_sols[sol]
            f_arc_sol = sol_from_list_to_arc(f_sol)
            f_arc_sol_cut = cut_arc_sol(f_arc_sol)
            better_than_gbest = False
            if f_val <= female_pbest_value[sol]:
                female_pbest_value[sol] = f_val
                female_pbest_sol[sol] = f_sol[:]
                female_pbest_arc_sols[sol] = [tuple(x) for x in f_arc_sol]
                female_pbest_arc_sols_cut[sol] = [tuple(x) for x in f_arc_sol_cut]
                female_attractor = female_pbest_sol[sol]
                if f_val <= gbest_value:
                    gbest_value = f_val
                    gbest_sol = f_sol[:]
                    gbest_arc_sol_cut = female_pbest_arc_sols_cut[sol]
                    fnew_pos = random_flight(f_sol[:])
                    fevaluation = evaluate_all_sols_check(fnew_pos, df_item_pool, heavy_item_set, name_path_input)
                    female_cur_sols_value[sol] = fevaluation[1]
                    female_cur_sols[sol] = fnew_pos[:]
                    female_cur_arc_sols[sol] = sol_from_list_to_arc(fnew_pos[:])
                    female_arc_sols_cut[sol] = cut_arc_sol([tuple(x) for x in female_cur_arc_sols[sol]])
                    female_velocity_dict[sol] = init_velocity_sol([tuple(x) for x in female_arc_sols_cut[sol]])
                    better_than_gbest = True

            if not better_than_gbest:
                fcoef_velocity = coef_times_velocity(gravity_calculation(gmax, gmin, gen, num_gen),
                                                     female_velocity_dict[sol])
                female_diff = position_minus_position(male_arc_sols_cut[sol], female_arc_sols_cut[sol])
                coef_female_diff = coef_times_position(a3, female_diff)
                fadded_velocity = add_velocity(fcoef_velocity, coef_female_diff)
                fvelocity_check = check_velocity_inconsistency(fadded_velocity)
                fcut_set = creat_cut_set(fvelocity_check, alpha)
                fnew_pos = sol_position_update(fcut_set, female_arc_sols_cut[sol], sub_E_list,
                                               female_cur_sols[sol][0],
                                               female_pbest_sol[sol][0],
                                               gbest_sol[0])[0]
                fevaluation = evaluate_all_sols_check(fnew_pos, df_item_pool, heavy_item_set, name_path_input)
                female_cur_sols_value[sol] = fevaluation[1]
                female_cur_sols[sol] = fnew_pos[:]
                female_velocity_dict[sol] = fvelocity_check

        gbest_each_gen.append(gbest_value)
        # progress_percent = (gen + 1) / num_gen * 100
        # print(f'Progress: {progress_percent:.2f}%')

    best_solution = evaluate_all_sols_check(gbest_sol, df_item_pool, heavy_item_set, name_path_input)
    # print("----" * 50)
    # print(f"Name of Input File: {name_path_input}")
    # print(f"Number of Generations: {num_gen}")
    # print(f"Population Size: {pop_size}")
    # print(f"Final Best Solution (Tardiness): {min(gbest_each_gen)}")
    # print(f"Best Solution: - Picker : {best_solution[0]} - Tardiness: {best_solution[1]} - Batch: {best_solution[2]}")
    # print(gbest_each_gen)
    # print("----" * 50)
    return gbest_each_gen, best_solution,min(gbest_each_gen)
