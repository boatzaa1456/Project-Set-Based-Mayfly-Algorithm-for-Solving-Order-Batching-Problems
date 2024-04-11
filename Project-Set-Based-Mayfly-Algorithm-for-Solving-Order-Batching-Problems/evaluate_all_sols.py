import csv
import pandas as pd

from calculate_process_time import calculate_completion_time
from batching_item import batching_open
from sequencing_assignment_algorithms import ESDR_algorithms
from routing import precedence_constrained_routing

def evaluate_all_sols_check(new_pop_sol, df_item_poor_batch, heavy_item_set, name_path_input):
    ''' Read File setting parameter : '''
    path_folder = name_path_input + '\\setting_parameter_' + name_path_input + '.csv'
    df_setting_parameter = pd.read_csv(path_folder)


    # Open file
    with open(path_folder) as file_obj:
        # Create reader object by passing the file
        # object to DictReader method
        reader_obj = csv.reader(file_obj)
        # Iterate over each row in the csv file
        # using reader object
        for row in reader_obj:
            if row[0] == 'init_method':
                init_method = int(row[1])
            elif row[0] == 'method_batch':
                method_batch = int(row[1])
            elif row[0] == 'method_routing':
                method_routing = int(row[1])
            elif row[0] == 'num_picker':
                num_picker = int(row[1])
            elif row[0] == 'capacity_picker':
                capacity_picker_1 = int(row[1])
            elif row[0] == 'value_threshold':
                value_threshold_1 = int(row[1])
            elif row[0] == 'rack_x':
                rack_x = int(row[1])
            elif row[0] == 'rack_y':
                rack_y = int(row[1])
            elif row[0] == 'aisle_x':
                aisle_x = int(row[1])
            elif row[0] == 'enter_aisle':
                enter_aisle = int(row[1])
            elif row[0] == 'distance_y':
                distance_y = int(row[1])
            elif row[0] == 'block':
                block = int(row[1])
            elif row[0] == 'aisle':
                aisle = int(row[1])
            elif row[0] == 'depot':
                depot = int(row[1])
            elif row[0] == 'each_aisle_item':
                each_aisle_item = int(row[1])
            elif row[0] == 'setup_time':
                setup_time = int(row[1])
            elif row[0] == 'picking_and_searching_time':
                picking_and_searching_time = int(row[1])
            elif row[0] == 'speed_picker':
                speed_picker = int(row[1])
            elif row[0] == 'sorting_time':
                sorting_time = int(row[1])


    record_list_batch_PSO = []
    record_list_distance = []
    list_batching_item_PSO = []
    ''' Create batching order :'''
    list_index = new_pop_sol

    if method_batch == 1:

        list_batching_item_PSO, df_item_poor_new_PSO = batching_1(df_item_poor_batch, list_index,
                                                                                               capacity_picker_1, value_threshold_1,  name_path_input)

    elif method_batch == 2:

        list_batching_item_PSO, df_item_poor_new_PSO, list_index_item_in_batch = batching_open(df_item_poor_batch, list_index,
                                                                                                  capacity_picker_1, value_threshold_1, heavy_item_set, name_path_input)

    list_batching = []
    num_batch = len(list_batching_item_PSO)
    for i in range(num_batch):
        for j in list_batching_item_PSO[i]:
            list_batching.append(j)

    list_batching = sum(list_batching_item_PSO, [])
    record_list_batch_PSO.append(list_batching)


    ''' Create distance order : '''
    list_distance_batch = []
    for i in range(0, len(list_batching_item_PSO)):
        distance_batch = 0
        item_list = list_batching_item_PSO[i]

        if method_routing == 1:
            ''' method 1 : s_shape + precedence constrained'''
            distance_each_batch = s_shape_routing(item_list, aisle, each_aisle_item, rack_x, rack_y, aisle_x,
                                                  enter_aisle, distance_y)
        elif method_routing == 2:
            ''' method 2 : combined + precedence constrained'''
            distance_each_batch = combined_routing(item_list, aisle, each_aisle_item, rack_x, rack_y, aisle_x,
                                                   enter_aisle, distance_y)
        elif method_routing == 3:
            ''' method 3 : combined + precedence constrained'''
            distance_each_batch = precedence_constrained_routing(item_list, aisle, each_aisle_item, rack_x, rack_y,
                                                                 aisle_x, enter_aisle, distance_y)

        # print(f'total_distance of batch No. {sol} = {distance_each_batch}')
        list_distance_batch.append(distance_each_batch)
        total_distance = sum(list_distance_batch)
    '''record distance each solution '''
    record_list_distance.append(list_distance_batch)


    ''' จำนวนสินค้าแต่ละกลุ่มคำสั่งซื้อ batch '''
    number_item_each_batch = []
    for k in range(0, len(list_index_item_in_batch)):
        number_item_each_batch.append(len(list_index_item_in_batch[k]))

    ''' Calculate process_time each batch :  '''
    process_time_batch = []
    batch = df_item_poor_batch[df_item_poor_batch['location'].isin(list_batching_item_PSO)]

    for i in range(0, len(list_distance_batch)):
        process_time = calculate_completion_time(list_distance_batch[i], number_item_each_batch[i],
                                                     setup_time, picking_and_searching_time, speed_picker, sorting_time)
        process_time_batch.append(float(format(process_time, '.3f')))

    for j in range(len(list_distance_batch)):
        df_item_poor_new_PSO.loc[df_item_poor_new_PSO['batch'] == j + 1, 'process_time'] = process_time_batch[j]

    ''' create sequencing and assignment batch to picker by Random  
            Example : สุ่มเลือก batch ไหน picker ใดหยิบ
                    list_batch = [[7, 1, 5, 6], [8, 2, 0, 3, 4]]
                    Picker No. 1 = b7=(9.567), b1=(18.800), b5=(11.900), b6=(11.367)
                    Picker No. 2 = b8=(9.917), b2=(25.550), b0=(31.067), b3=(31.067), b4=(51.634)
                    Picker No. 1 = [b7 (9.567), b1 (28.367), b5 (40.267), b6 (11.367)]
                    Picker No. 2 = [b8 (9.917), b2=(35.467), b0=(66.534), b3=(97.601), b4=(149.235)] '''

    list_picker, list_tardiness_each_order, total_tardiness, df_item_poor_AS = ESDR_algorithms(df_item_poor_new_PSO, num_picker)

    return list_picker, total_tardiness,list_index_item_in_batch
