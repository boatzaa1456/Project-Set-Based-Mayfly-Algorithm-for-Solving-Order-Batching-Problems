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
# from numba import jit, cuda
value_heavy = 40


# seed_all = 3124
# random.seed(1234)

def read_input(name_path_input):
    # อ่านไฟล์ CSV เพียงครั้งเดียว
    df_item_sas_random = pd.read_csv('df_item_sas_random.csv')

    # อ่านไฟล์อื่นๆ
    duedate_path = f'{name_path_input}\\duedate_{name_path_input}.csv'
    input_location_path = f'{name_path_input}\\input_location_item_{name_path_input}.csv'
    df_duedate = pd.read_csv(duedate_path, header=None)
    df_item_oder = pd.read_csv(input_location_path, header=None)

    # แปลงเป็น list
    list_duedate = df_duedate[0].tolist()
    num_order = df_item_oder.shape[1]

    # ประมวลผลแต่ละ order และสร้าง DataFrame
    order_items = [df_item_oder[order][df_item_oder[order] != 0] for order in range(num_order)]
    df_item_pools = [
        df_item_sas_random[df_item_sas_random['location'].isin(order_item)].assign(duedate=list_duedate[order],
                                                                                   order=order) for order, order_item in
        enumerate(order_items)]

    # รวม DataFrame
    df_item_pool = pd.concat(df_item_pools, ignore_index=True)

    # สร้าง list_order และ list_total_item (ถ้าจำเป็น)
    list_order = [order for order in range(num_order) for _ in range(len(order_items[order]))]
    list_total_item = [item for order_item in order_items for item in order_item.tolist()]

    return df_item_pool, df_item_sas_random


# แปลงจาก list ให้กลายเป็น arc ต้องรู้ว่ามีทั้งหมดกี่ตัว
def sol_from_list_to_arc(sol):
    num_item = len(sol)
    arc_sol = []
    for i in range(num_item - 1):
        arc_sol.append((sol[i], sol[i + 1]))
    return arc_sol


def all_sol_from_list_to_arc(all_sols):
    num_sol = len(all_sols)  # เก็บจำนวนคำตอบของ all_sols
    num_item = len(all_sols[0])
    all_arc_sols = [[(all_sols[i][j], all_sols[i][j + 1]) for j in range(num_item - 1)] for i in range(num_sol)]

    return all_arc_sols


# all_arc_sols = all_sol_from_list_to_arc(cur_sol)
# print(all_arc_sols)

def cut_arc_sol(arc_sol):
    num_item = len(arc_sol) + 1
    arc_sol_list = [[] for _ in range(num_item)]

    for arc in arc_sol:
        arc_sol_list[arc[0]].append(arc)
        arc_sol_list[arc[1]].append(arc)

    arc_sol_cut = [arc_sol_list[item] for item in range(num_item)]

    return arc_sol_cut


def init_velocity_sol(arc_sol_cut):
    import random
    num_item = len(arc_sol_cut)  # นับจำนวน item จากจำนวนสมาชิกของ list arc_sol_cut
    arc_sol_velocity_dict = [{} for _ in range(num_item)]
    for item in range(num_item):
        for arc in arc_sol_cut[item]:
            arc_sol_velocity_dict[item][arc] = round(random.random(), 4)

    return arc_sol_velocity_dict


def coef_times_velocity(coef, arc_sol_velocity_dict):
    num_item = len(arc_sol_velocity_dict)
    coef_times_velocity_dict = [{} for item in range(num_item)]
    # [{}, {}, {}, ...]
    for item in range(num_item):
        for arc in arc_sol_velocity_dict[item].keys():
            if coef * arc_sol_velocity_dict[item][arc] > 1:
                coef_times_velocity_dict[item][arc] = 1
            else:
                coef_times_velocity_dict[item][arc] = round(coef * arc_sol_velocity_dict[item][arc], 4)
    return coef_times_velocity_dict


def position_minus_position(arc_first, arc_second):
    num_item = len(arc_first)
    pos_minus_pos = [[] for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_first[item]:
            if arc not in arc_second[item]:
                pos_minus_pos[item].append(arc)
    return pos_minus_pos


def coef_times_position(c_value, arc_diff):
    import random
    num_item = len(arc_diff)
    coef_times_position_dict = [{} for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_diff[item]:
            coef = c_value * random.random()
            if coef > 1:
                coef = 1
            coef_times_position_dict[item][arc] = round(coef, 3)
    return coef_times_position_dict


def add_velocity(velocity_first, velocity_second):
    num_item = len(velocity_first)
    added_velocity_dict = [{} for item in range(num_item)]
    for item in range(num_item):
        for arc in velocity_first[item]:
            added_velocity_dict[item][arc] = velocity_first[item][arc]
        for arc in velocity_second[item]:
            if arc in added_velocity_dict[item].keys():
                if velocity_second[item][arc] > added_velocity_dict[item][arc]:
                    added_velocity_dict[item][arc] = velocity_second[item][arc]
            else:
                added_velocity_dict[item][arc] = velocity_second[item][arc]
    return added_velocity_dict


def check_velocity_inconsistency(added_velocity_dict):
    num_item = len(added_velocity_dict)
    import copy
    new_added_velocity_dict = [{arc: prob for arc, prob in added_velocity_dict[item].items()} for item in
                               range(num_item)]
    for item in range(num_item):
        for arc_first in added_velocity_dict[item].keys():
            if arc_first in added_velocity_dict[arc_first[0]].keys():
                if added_velocity_dict[item][arc_first] < added_velocity_dict[arc_first[0]][arc_first]:
                    new_added_velocity_dict[item][arc_first] = added_velocity_dict[arc_first[0]][arc_first]
            if arc_first in added_velocity_dict[arc_first[1]].keys():
                if added_velocity_dict[item][arc_first] < added_velocity_dict[arc_first[1]][arc_first]:
                    new_added_velocity_dict[item][arc_first] = added_velocity_dict[arc_first[1]][arc_first]
    return new_added_velocity_dict


def creat_cut_set(added_velocity_dict, alpha):
    num_item = len(added_velocity_dict)
    cut_set = [[] for _ in range(num_item)]
    for item in range(num_item):
        for arc in added_velocity_dict[item].keys():
            if added_velocity_dict[item][arc] >= alpha:
                cut_set[item].append(arc)
    return cut_set


def select_dest_from_source(source, picked_list, *sets):
    # function ทำหน้าที่ในการเลือก item ที่เราจะเดินเก็บถัดไป (dest) จากตำแหน่งปัจจุบันที่เราอยู่ (source) โดยเราต้องการได้ผลลัพธ์เป็น
    # arc ของ (source,dest) และ picked_list เป็น list ที่เก็บ item ที่เราเดินเก็บไปแล้ว
    import random
    for set in sets:
        new_set = []
        if len(set[source]) > 0:
            for arc in set[source]:
                if arc[1] not in picked_list and arc[0] == source:
                    new_set.append(arc)
        if len(new_set) > 0:
            dest = random.choice(new_set)[1]
            break
    arc_source_dest = (source, dest)
    return dest, arc_source_dest


def sol_position_update(cut_set, previous_x, sub_E_list, start_previous_x, start_pbest, start_gbest):
    import random

    num_item = len(cut_set)
    picked_list = []
    picked_list_arc = []

    source = random.choice([start_previous_x, start_pbest, start_gbest, random.choice(range(num_item))])
    picked_list.append(source)

    for item_counter in range(num_item - 1):
        dest, arc_source_dest = select_dest_from_source(source, picked_list, cut_set, previous_x, sub_E_list)
        source = dest
        picked_list.append(dest)
        picked_list_arc.append(arc_source_dest)

    return picked_list, picked_list_arc


def mutShuffleIndexes(individual, indpb):
    """Shuffle the attributes of the input individual and return the mutant.
    The *individual* is expected to be a :term:`sequence`. The *indpb* argument is the
    probability of each attribute to be moved. Usually this mutation is applied on
    vector of indices.

    :param individual: Individual to be mutated.
    :param indpb: Independent probability for each attribute to be exchanged to
                  another position.
    :returns: A tuple of one individual.

    This function uses the :func:`~random.random` and :func:`~random.randint`
    functions from the python base :mod:`random` module.
    """
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            swap_indx = random.randint(0, size - 2)
            if swap_indx >= i:
                swap_indx += 1
            individual[i], individual[swap_indx] = \
                individual[swap_indx], individual[i]

    return individual


def cxPartialyMatched(ind1, ind2):
    """Executes a partially matched crossover (PMX) on the input individuals.
    The two individuals are modified in place. This crossover expects
    :term:`sequence` individuals of indices, the result for any other type of
    individuals is unpredictable.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    Moreover, this crossover generates two children by matching
    pairs of values in a certain range of the two parents and swapping the values
    of those indexes. For more details see [Goldberg1985]_.

    This function uses the :func:`~random.randint` function from the python base
    :mod:`random` module.

    .. [Goldberg1985] Goldberg and Lingel, "Alleles, loci, and the traveling
       salesman problem", 1985.
    """
    size = min(len(ind1), len(ind2))
    p1, p2 = [0] * size, [0] * size

    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Swap the matched value
        ind1[i], ind1[p1[temp2]] = temp2, temp1
        ind2[i], ind2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2


# # Define the simplified nuptial dance function
# def nuptial_dance(solution, attractor):
#     i = random.randint(0, len(solution) - 1)
#     if solution[i] in attractor:
#         j = attractor.index(solution[i])
#         if i != j and solution[j] != attractor[j]:
#             solution[i], solution[j] = solution[j], solution[i]
#     return solution
#
#
# def random_flight(solution):
#     # เพิ่มการตรวจสอบเงื่อนไขเพื่อให้แน่ใจว่า solution มีอย่างน้อย 2 ตัว
#     if len(solution) < 2:
#         return solution
#
#     i, j = random.sample(range(len(solution)), 2)
#     solution[i], solution[j] = solution[j], solution[i]
#     return solution

def nuptial_dance(solution, attractor):
    size = len(solution)
    # กำหนดประสิทธิภาพการสลับตามขนาดของข้อมูล
    num_swaps = 1 if size <= 20 else int(size / 10)  # ขนาดกลางและใหญ่จะมีการสลับมากขึ้น

    for _ in range(num_swaps):
        i = random.randint(0, size - 1)
        if solution[i] in attractor:
            j = attractor.index(solution[i])
            if i != j and solution[j] != attractor[j]:
                solution[i], solution[j] = solution[j], solution[i]
                if size <= 20:  # สำหรับขนาดเล็ก, ทำการสลับเพียงครั้งเดียว
                    break
    return solution


def random_flight(solution):
    size = len(solution)
    if size < 2:
        return solution

    num_swaps = 1 if size <= 20 else int(size / 10)  # ปรับจำนวนการสลับตามขนาด

    for _ in range(num_swaps):
        i, j = random.sample(range(size), 2)
        solution[i], solution[j] = solution[j], solution[i]
        if size <= 20:  # สำหรับขนาดเล็ก, ทำการสลับเพียงครั้งเดียว
            break
    return solution

def replace_with_better_offspring(parents, offspring):
    """
    แทนที่พ่อหรือแม่ด้วยลูกที่ดีกว่าในกลุ่มเพศเดียวกัน

    :param parents: ลิสต์ของพ่อหรือแม่ (ตามเพศ) แต่ละคนเป็น tuple ที่มีค่า tardiness ที่ตำแหน่งที่ 1
    :param offspring: ลิสต์ของลูก ตามเพศเดียวกันกับ parents, แต่ละคนเป็น tuple ที่มีค่า tardiness ที่ตำแหน่งที่ 1
    :return: ลิสต์ใหม่ของประชากรที่อาจรวมถึงลูกที่ดีกว่าแทนที่พ่อหรือแม่
    """
    # สร้างลิสต์ใหม่จาก parents เพื่อไม่เปลี่ยนแปลงข้อมูลเดิม
    new_population = parents.copy()

    # ลูปเพื่อเปรียบเทียบแต่ละลูกกับพ่อหรือแม่
    for i, parent in enumerate(parents):
        # หากมีลูกน้อยกว่าพ่อหรือแม่, อาจไม่ต้องเปรียบเทียบทุกคู่
        if i < len(offspring):
            # เปรียบเทียบค่า tardiness
            if offspring[i][1] < parent[1]:
                # แทนที่พ่อหรือแม่ด้วยลูกที่ดีกว่า
                new_population[i] = offspring[i]

    return new_population


def gravity_calculation(gmax, gmin, gen, num_gen):
    gravity = gmax - ((gmax - gmin) * gen / num_gen)
    return gravity


def extract_and_flatten(solution):
    list_of_lists = solution[2]
    flattened_list = [item for sublist in list_of_lists for item in sublist]

    return flattened_list


def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def alpha_calculation( gen, num_gen):
    alpha = (gen / num_gen)
    return alpha

def rank_and_sort(male_evaluations_new_pos, male_new_pos):
    # สร้างคู่ (male_evaluations_new_pos, male_new_pos) เพื่อเก็บข้อมูลทั้งสอง
    data_pairs = list(zip(male_evaluations_new_pos, male_new_pos))

    # ใช้ฟังก์ชัน sorted เพื่อเรียงลำดับข้อมูลตาม male_evaluations_new_pos
    sorted_data = sorted(data_pairs, key=lambda x: x[0])

    # สร้างรายการของ male_new_pos ใหม่ตามลำดับที่ได้
    ranked_male_new_pos = [data[1] for data in sorted_data]

    # คืนค่าผลลัพธ์ที่เรียงลำดับแล้ว
    return ranked_male_new_pos


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
            evaluation = evaluate_all_sols_check(mayfly, df_item_pool, heavy_item_set, name_path_input)
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
        # Initialize lists before the loop
        # male_coef_velocity = []
        # male_pbest_differant = []
        # male_gbest_differant = []
        # male_added_best_differant = []
        # male_added_velocity = []
        # coef_male_pbest = []
        # coef_male_gbest = []
        # male_velocity_check_incon = []
        # male_cut_set = []
        male_new_pos = []
        male_evaluations_new_pos = []
        female_new_pos = []
        female_evaluations_new_pos = []
        # extrac_new_male_sols = []

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
                    mevaluation = evaluate_all_sols_check(mnew_pos, df_item_pool, heavy_item_set, name_path_input)
                    male_cur_sols_value[sol] = mevaluation[1]
                    male_cur_sols[sol] = mnew_pos[:]
                    male_cur_arc_sols[sol] = sol_from_list_to_arc(mnew_pos[:])
                    male_arc_sols_cut[sol] = cut_arc_sol([tuple(item) for item in male_cur_arc_sols[sol]])
                    male_velocity_dict[sol] = init_velocity_sol([tuple(item) for item in male_arc_sols_cut[sol]])
                    male_new_pos.append(mnew_pos)
                    male_evaluations_new_pos.append(mevaluation[1])
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
                mevaluation = evaluate_all_sols_check(mnew_pos, df_item_pool, heavy_item_set, name_path_input)
                # mextrac_sols = extract_and_flatten(mevaluation)

                # male_coef_velocity.append(mcoef_velocity)
                # male_pbest_differant.append(mpbest_diff)
                # male_gbest_differant.append(mgbest_diff)
                # male_added_best_differant.append(madded_best_diff)
                # male_added_velocity.append(madded_velocity)
                # coef_male_pbest.append(coef_pbest_diff)
                # coef_male_gbest.append(coef_gbest_diff)
                # male_velocity_check_incon.append(mvelocity_check_incon)
                # male_cut_set.append(mcut_set)
                male_new_pos.append(mnew_pos)
                male_evaluations_new_pos.append(mevaluation[1])
                # extrac_new_male_sols.append(mextrac_sols)

                male_cur_sols_value[sol] = mevaluation[1]
                male_cur_sols[sol] = mnew_pos[:]
                male_velocity_dict[sol] = mvelocity_check_incon
                male_each_gen.append(mevaluation[1])

            # Initialize lists before the loop
            # female_coef_velocity = []
            # female_male_diff = []
            # female_added_velocity = []
            # coef_female = []
            # female_cut_set = []
            # female_velocity_check_incon = []

            # extrac_new_female_sols = []

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
                    fevaluation = evaluate_all_sols_check(fnew_pos, df_item_pool, heavy_item_set, name_path_input)
                    female_cur_sols_value[sol] = fevaluation[1]
                    female_cur_sols[sol] = fnew_pos[:]
                    female_cur_arc_sols[sol] = sol_from_list_to_arc(fnew_pos[:])
                    female_arc_sols_cut[sol] = cut_arc_sol([tuple(item) for item in female_cur_arc_sols[sol]])
                    female_velocity_dict[sol] = init_velocity_sol([tuple(item) for item in female_arc_sols_cut[sol]])
                    female_new_pos.append(fnew_pos)
                    female_evaluations_new_pos.append(fevaluation[1])
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
                fevaluation = evaluate_all_sols_check(fnew_pos, df_item_pool, heavy_item_set, name_path_input)
                # fextrac_sols = extract_and_flatten(fevaluation)

                # female_coef_velocity.append(fcoef_velocity)
                # female_male_diff.append(female_diff)
                # coef_female.append(coef_female_diff)
                # female_added_velocity.append(fadded_velocity)
                # female_velocity_check_incon.append(fvelocity_check_incon)
                # female_cut_set.append(fcut_set)
                female_new_pos.append(fnew_pos)
                female_evaluations_new_pos.append(fevaluation[1])
                # extrac_new_female_sols.append(fextrac_sols)

                female_cur_sols_value[sol] = fevaluation[1]
                female_cur_sols[sol] = fnew_pos[:]
                female_velocity_dict[sol] = fvelocity_check_incon
                female_each_gen.append(fevaluation[1])

        rank_male_mayfly = rank_and_sort(male_evaluations_new_pos,male_new_pos)
        rank_female_mayfly = rank_and_sort(female_evaluations_new_pos,female_new_pos)

        for sol in range(half_pop_size):
            offspring_male,offspring_female = cxPartialyMatched(rank_male_mayfly[sol],rank_female_mayfly[sol])
            offspring_male,offspring_female = mutShuffleIndexes(offspring_male,0.2),mutShuffleIndexes(offspring_female,0.2)
            offspring_male_value,offspring_female_value = evaluate_all_sols_check(offspring_male, df_item_pool, heavy_item_set, name_path_input),evaluate_all_sols_check(offspring_female, df_item_pool, heavy_item_set, name_path_input)
            if offspring_male_value[1] <= male_cur_sols_value[sol]:
                male_cur_sols_value[sol] = offspring_male_value[1]
                male_cur_sols[sol] = offspring_male
            if offspring_female_value[1] <= female_cur_sols_value[sol]:
                female_cur_sols_value[sol] = offspring_female_value[1]
                female_cur_sols[sol] = offspring_female


        gbest_each_gen.append(gbest_value)
    #     progress_percent = (gen + 1) / num_gen * 100
    #     print(f'Progress: {progress_percent:.2f}%')
    best_solution = evaluate_all_sols_check(gbest_sol, df_item_pool, heavy_item_set, name_path_input)
    # print("----" * 50)
    # print(f"Name of Input File: {name_path_input}")
    # print(f"Number of Generations: {num_gen}")
    # print(f"Population Size: {pop_size}")
    # print(f"Final Best Solution (Tardiness): {min(gbest_each_gen)}")
    # print(f"Best Solution: - Picker : {best_solution[0]} - Tardiness: {best_solution[1]} - Batch: {best_solution[2]}")
    # print(gbest_each_gen)
    # print("----" * 50)
    return min(gbest_each_gen),best_solution
    # return gbest_each_gen, male_each_gen, female_each_gen , best_solution

# start_time = time.time()
# num_gen = 100
# pop_size = 50
# a1 = 1
# a2 = 2
# a3 = 1
# gmax = 0.9
# gmin = 0.3
# alpha = 0.7
# random_seed = 1111
#
# gbest_each_gen = []
# male_each_gen = []
# female_each_gen = []
#
# name_path_input = '1R-20I-150C-2P'
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
# # #
# # # # คำนวณค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน
# # # average_male = [np.mean(gen) for gen in male_each_gen]
# # # average_female = [np.mean(gen) for gen in female_each_gen]
# # # std_deviation_male = [np.std(gen) for gen in male_each_gen]
# # # std_deviation_female = [np.std(gen) for gen in female_each_gen]
# rounds = np.arange(1, len(gbest_each_gen) + 1)
# # #
# # # # ใช้ Polynomial regression สำหรับเส้นแนวโน้ม
# # # p_male = Polynomial.fit(rounds - 1, average_male, deg=3)
# # # p_female = Polynomial.fit(rounds - 1, average_female, deg=3)
# # #
# # # # สร้างข้อมูลสำหรับเส้นแนวโน้ม
# # # x_new = np.linspace(0, len(male_each_gen) - 1, num=len(male_each_gen))
# # # y_new_male = p_male(x_new)
# # # y_new_female = p_female(x_new)
#
# plt.figure(figsize=(16, 8))
#
# # วาดกราฟข้อมูล
# plt.plot(rounds, gbest_each_gen, '-o', color='red', label='GBest Value', markersize=4, linewidth=1.5)
# # #
# # # # วาดกราฟข้อมูล
# # # plt.plot(rounds, average_male, '-^', color='blue', label='Average Male Value', markersize=4, linewidth=1.5)
# # # plt.plot(rounds, average_female, '-s', color='green', label='Average Female Value', markersize=4, linewidth=1.5)
# # #
# # # # วาดเส้นแนวโน้ม
# # # plt.plot(x_new + 1, y_new_male, 'b--', label='Male Trendline')
# # # plt.plot(x_new + 1, y_new_female, 'g--', label='Female Trendline')
# # #
# # # #แสดงความเคลื่อนไหวด้วยส่วนเบี่ยงเบนมาตรฐาน
# # # plt.fill_between(rounds, np.array(average_male) - np.array(std_deviation_male),
# # #                  np.array(average_male) + np.array(std_deviation_male), color='blue', alpha=0.2,
# # #                  label='Std.Deviation of Male Value')
# # # plt.fill_between(rounds, np.array(average_female) - np.array(std_deviation_female),
# # #                  np.array(average_female) + np.array(std_deviation_female), color='green', alpha=0.2,
# # #                  label='Std.Deviation of Female Value')
# # #
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
