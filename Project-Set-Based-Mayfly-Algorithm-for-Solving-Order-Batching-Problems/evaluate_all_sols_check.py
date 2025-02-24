import pandas as pd
from calculate_process_time import calculate_completion_time
from batching_item_check import batching_open  # และ (batching_1) หากมี
from sequencing_assignment_algorithms import ESDR_algorithms
from routing import precedence_constrained_routing, s_shape_routing, combined_routing


def evaluate_all_sols(new_pop_sol, df_item_poor_batch, heavy_item_set, name_path_input):
    # สร้าง path สำหรับไฟล์ตั้งค่า
    path_folder = f"{name_path_input}\\setting_parameter_{name_path_input}.csv"

    # อ่านไฟล์ตั้งค่าเพียงครั้งเดียวแล้วแปลงเป็น dictionary
    df_setting = pd.read_csv(path_folder, header=None, index_col=0, squeeze=True)
    settings = df_setting.to_dict()

    # ดึงค่าต่างๆ จาก settings
    init_method = int(settings.get('init_method', 0))
    method_batch = int(settings.get('method_batch', 0))
    method_routing = int(settings.get('method_routing', 0))
    num_picker = int(settings.get('num_picker', 0))
    capacity_picker_1 = int(settings.get('capacity_picker', 0))
    value_threshold_1 = int(settings.get('value_threshold', 0))
    rack_x = int(settings.get('rack_x', 0))
    rack_y = int(settings.get('rack_y', 0))
    aisle_x = int(settings.get('aisle_x', 0))
    enter_aisle = int(settings.get('enter_aisle', 0))
    distance_y = int(settings.get('distance_y', 0))
    block = int(settings.get('block', 0))
    aisle = int(settings.get('aisle', 0))
    depot = int(settings.get('depot', 0))
    each_aisle_item = int(settings.get('each_aisle_item', 0))
    setup_time = int(settings.get('setup_time', 0))
    picking_and_searching_time = int(settings.get('picking_and_searching_time', 0))
    speed_picker = int(settings.get('speed_picker', 0))
    sorting_time = int(settings.get('sorting_time', 0))

    record_list_batch_PSO = []
    record_list_distance = []

    ''' Create batching order : '''
    list_index = new_pop_sol
    if method_batch == 1:
        # หากมีฟังก์ชัน batching_1 ให้เรียกใช้ (ต้อง import batching_1 ด้วย)
        list_batching_item_PSO, df_item_poor_new_PSO = batching_1(
            df_item_poor_batch, list_index, capacity_picker_1, value_threshold_1, name_path_input
        )
        # สำหรับ method_batch == 1 ให้กำหนด list_index_item_in_batch จาก list_batching_item_PSO
        list_index_item_in_batch = [batch for batch in list_batching_item_PSO]
    elif method_batch == 2:
        list_batching_item_PSO, df_item_poor_new_PSO, list_index_item_in_batch = batching_open(
            df_item_poor_batch, list_index, capacity_picker_1, value_threshold_1, heavy_item_set, name_path_input
        )

    # Flatten รายการ batching ทั้งหมด (ใช้ sum กับ list ว่างเพื่อ flatten)
    list_batching = sum(list_batching_item_PSO, [])
    record_list_batch_PSO.append(list_batching)

    ''' Create distance order : '''
    list_distance_batch = []
    for item_list in list_batching_item_PSO:
        if method_routing == 1:
            distance_each_batch = s_shape_routing(
                item_list, aisle, each_aisle_item, rack_x, rack_y, aisle_x, enter_aisle, distance_y
            )
        elif method_routing == 2:
            distance_each_batch = combined_routing(
                item_list, aisle, each_aisle_item, rack_x, rack_y, aisle_x, enter_aisle, distance_y
            )
        elif method_routing == 3:
            distance_each_batch = precedence_constrained_routing(
                item_list, aisle, each_aisle_item, rack_x, rack_y, aisle_x, enter_aisle, distance_y
            )
        else:
            distance_each_batch = 0  # กรณีไม่มี method_routing ที่ตรง
        list_distance_batch.append(distance_each_batch)

    total_distance = sum(list_distance_batch)
    record_list_distance.append(list_distance_batch)

    ''' คำนวณจำนวนสินค้าของแต่ละ batch (ใช้ list comprehension) '''
    number_item_each_batch = [len(batch) for batch in list_index_item_in_batch]

    ''' Calculate process_time สำหรับแต่ละ batch : '''
    process_time_batch = []
    for distance, num_items in zip(list_distance_batch, number_item_each_batch):
        process_time = calculate_completion_time(
            distance, num_items, setup_time, picking_and_searching_time, speed_picker, sorting_time
        )
        # ปัดเศษให้เหลือ 3 ตำแหน่งทศนิยม
        process_time_batch.append(round(process_time, 3))

    # Optimize DataFrame update: สร้าง mapping จาก batch number ไปยัง process time
    process_time_mapping = {j + 1: pt for j, pt in enumerate(process_time_batch)}
    df_item_poor_new_PSO['process_time'] = df_item_poor_new_PSO['batch'].map(process_time_mapping)

    ''' สร้าง sequencing และ assignment ให้กับ picker โดยใช้ ESDR_algorithms '''
    list_picker, list_tardiness_each_order, total_tardiness, df_item_poor_AS = ESDR_algorithms(
        df_item_poor_new_PSO, num_picker
    )

    return list_picker, total_tardiness, list_index_item_in_batch
