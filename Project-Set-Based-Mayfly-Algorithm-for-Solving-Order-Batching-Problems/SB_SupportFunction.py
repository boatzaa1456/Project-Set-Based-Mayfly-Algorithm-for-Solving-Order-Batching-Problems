import random
import pandas as pd
import csv

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

def extract_and_flatten(solution):
    list_of_lists = solution[2]
    flattened_list = [item for sublist in list_of_lists for item in sublist]

    return flattened_list