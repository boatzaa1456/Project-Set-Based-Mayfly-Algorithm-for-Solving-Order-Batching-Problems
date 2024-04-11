import pandas as pd
import numpy as np
import math
#
def calculate_completion_time(list_distance_batch, number_item_each_batch, *info):
    
    setup_time, picking_and_searching_time, speed_picker, sorting_time = info
    processing_time = setup_time + ((picking_and_searching_time/60)*number_item_each_batch)+(list_distance_batch/speed_picker) + (3*number_item_each_batch)

    return processing_time

    list_process_time_picker = []
    for i in range(0, num_picker):
        k = 0
        time_picker = 0
        process_time_picker = []
        for j in list_sequencing_batch[i]:
            completion_time = float(completion_time_batch[j])
            completion_time = round(completion_time, 3)
            time_picker += completion_time
            process_time_picker.append(round(time_picker, 3))
            k += 1
        list_process_time_picker.append(process_time_picker)

    return list_process_time_picker





