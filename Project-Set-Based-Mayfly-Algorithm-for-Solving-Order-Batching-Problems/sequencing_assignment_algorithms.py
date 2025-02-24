import pandas as pd
import numpy as np


def calculate_completion_time(df_order, time):
    # คำนวณ completion time โดยใช้ max(process_time) บวกกับเวลา (time)
    max_process_time = df_order['process_time'].max()
    completion_time_batch = max_process_time + time
    return completion_time_batch


def calculate_tardiness_order(df_order, order_no):
    # คำนวณ tardiness ของ order หนึ่ง ๆ
    if df_order.empty:
        return 0
    max_completion_time = df_order['CompletionTime'].max()
    max_duedate = df_order['duedate'].min()
    tardiness_each_order = round(max_completion_time - max_duedate, 3)
    return tardiness_each_order if tardiness_each_order > 0 else 0


def average_dua_date(num_batch, df_item_poor):
    """
    ปรับปรุงการคำนวณค่าเฉลี่ย duedate และ max process time สำหรับแต่ละ batch ด้วย groupby
    ผลลัพธ์ที่ได้เป็นสอง lists:
      - list_average_dua_date_each_batch: ค่าเฉลี่ย duedate ของแต่ละ batch (เรียงตามหมายเลข batchจาก 1 ถึง num_batch)
      - list_process_time: max process_time ของแต่ละ batch
    """
    # Group by 'batch' แล้วคำนวณค่าเฉลี่ย duedate และ max process_time
    grouped = df_item_poor.groupby('batch')
    avg_duedate = grouped['duedate'].mean().round(3)
    max_process = grouped['process_time'].max()

    # สร้าง lists สำหรับ batch 1 ถึง num_batch (สมมติว่า batch ถูกกำหนดเป็นเลข 1,2,...,num_batch)
    list_average_dua_date_each_batch = [avg_duedate.loc[b] for b in range(1, num_batch + 1)]
    list_process_time = [max_process.loc[b] for b in range(1, num_batch + 1)]
    return list_average_dua_date_each_batch, list_process_time


def ESDR_algorithms(df_item_pool_new, num_picker):
    """
    Optimized ESDR_algorithms:
      - กรองรายการที่มี batch != 0 และรีเซ็ต index เพียงครั้งเดียว
      - คำนวณค่าเฉลี่ย duedateและ max process_time ของแต่ละ batch แบบ vectorized โดยใช้ groupby
      - นำข้อมูล batch ทั้งหมดมาจัดเรียง (sorted) ตามค่าเฉลี่ย duedate (ascending)
      - จัดสรร batch ให้กับ pickerในรูปแบบ round-robin พร้อมคำนวณ cumulative completion time
      - อัปเดตค่า CompletionTime ใน DataFrame ทีละ batch (โดยใช้การ assign แบบ vectorized)
      - คำนวณ tardiness ของแต่ละ orderโดย groupby แทนการวนลูปซ้ำ ๆ
    ผลลัพธ์ที่ส่งกลับ: list_picker, list_tardiness_each_order, total_tardiness, df_item_pool
    """
    # กรองรายการที่มี batch != 0 และรีเซ็ต index
    df_item_pool = df_item_pool_new[df_item_pool_new['batch'] != 0].copy()
    df_item_pool.reset_index(drop=True, inplace=True)

    # จำนวน batch และ order
    num_batch = int(df_item_pool['batch'].max())
    num_order = int(df_item_pool['order'].max())

    # คำนวณค่าเฉลี่ย duedateและ max process_time สำหรับแต่ละ batchแบบ vectorized
    list_average_dua_date_each_batch, list_process_time = average_dua_date(num_batch, df_item_pool)

    # สร้างข้อมูล batch ทั้งหมดเป็น list ของ tuple (batch, avg_duedate, process_time)
    batches_info = [(b, list_average_dua_date_each_batch[b - 1], list_process_time[b - 1]) for b in
                    range(1, num_batch + 1)]
    # เรียงลำดับ batch ตามค่าเฉลี่ย duedate (น้อยที่สุดก่อน)
    batches_info_sorted = sorted(batches_info, key=lambda x: x[1])

    # เตรียมผลลัพธ์สำหรับการจัดสรร batch ให้กับ picker
    list_picker = [[] for _ in range(num_picker)]
    # สำหรับเก็บ cumulative completion time ของแต่ละ picker (เริ่มต้นที่ 0)
    cum_completion_time = [0.0 for _ in range(num_picker)]

    # จัดสรร batch (ในลำดับที่เรียงแล้ว) แบบ round-robin
    for i, (batch, avg_duedate, proc_time) in enumerate(batches_info_sorted):
        p = i % num_picker  # เลือก picker ด้วยวิธี round-robin
        # คำนวณ completion time สำหรับ batch นี้
        comp_time = cum_completion_time[p] + proc_time
        cum_completion_time[p] = comp_time
        list_picker[p].append(batch)
        # อัปเดต CompletionTime ใน df_item_pool สำหรับ batch นี้ (ใช้การ assign แบบ vectorized)
        df_item_pool.loc[df_item_pool['batch'] == batch, 'CompletionTime'] = comp_time

    # คำนวณ tardiness สำหรับแต่ละ order โดย groupby
    order_groups = df_item_pool.groupby('order')
    tardiness_dict = {}
    for order, group in order_groups:
        max_completion = group['CompletionTime'].max()
        min_duedate = group['duedate'].min()
        tardiness = round(max_completion - min_duedate, 3)
        tardiness_dict[order] = tardiness if tardiness > 0 else 0
        df_item_pool.loc[df_item_pool['order'] == order, 'TardinessOrder'] = tardiness_dict[order]

    total_tardiness = sum(tardiness_dict.values())
    # สร้าง list ของ tardiness ตาม order จาก 0 ถึง num_order (หากไม่มีให้ถือเป็น 0)
    list_tardiness_each_order = [tardiness_dict.get(i, 0) for i in range(num_order + 1)]

    return list_picker, list_tardiness_each_order, total_tardiness, df_item_pool


def greedy_algorithms(df_item_poor_new, num_picker):



    print(f'--- Start Greedy Algorithms ---')
    pd.set_option('display.max_columns', None)
    ''' Average due date น้อยที่สุด ใน Batch
        input : 1. list_item_in_batch 'เอาไว้นับจำนวน  Batch ทั้งหมด'= [[39, 12, 31, 13, 20, 52, 25], [40, 48, 26, 11, 46, 10, 35, 14], [17, 41, 24, 29], 
                                        [43, 45, 15, 22], [44, 1, 7, 33, 47, 36], [50, 18, 21], [34, 3, 32, 51, 16, 28], 
                                        [2, 5], [8, 38, 49, 9, 0, 23], [19, 30, 42, 4, 27], [37, 6]]  
                2. num_picker = จำนวนพนักงานจากโจทย์
                3. df_item_poor_new  ---> 
        ตัวอย่าง
                    location    duedate       order       batch     process_time  
            0        886        8.54167         14          1       22.483            
            1        870        122.54200       4           1       22.483     
            2        780        105.54200       12          1       22.483
            3         21        63.54170        5           1       22.483
            4        624        200.54200       7           1       22.483
            5        731        10.54170        19          1       22.483 
            6        206        114.54200       9           1       22.483 '''

    ''' Average due date น้อยที่สุด ใน Batch
           Average due date each batch   =  sum[D0 +  D1 + .. + D6]/7 
                                =  89.399
    '''
    ''' ตัด ค่า 0 ของที่แยก bach                           '''
    df_item_poor = df_item_poor_new.drop(df_item_poor_new[df_item_poor_new['batch'] == 0].index)
    df_item_poor.reset_index(inplace=True)
    df_item_poor = df_item_poor.drop(['index'], axis=1)

    # '''เพิ่ม colume --> completion_time '''
    ''' จำนวน Bacth ทั้งหมด  '''
    num_batch = df_item_poor['batch'].iloc[-1]
    ''' จำนวน order ทั้งหมด  '''
    df_order = df_item_poor
    df_order.sort_values(by='order', inplace=True)
    num_order = df_order['order'].iloc[-1]
    ''' จำนวน item ทั้งหมด  '''
    num_item = len(df_item_poor)

    '''Step 1 : calculate average dua date of each Batch [...คำนวณหาค่าเฉลี่ยกำหนดวันเวลาส่งมอบสินค้าของแต่ละกลุ่มคำสั่งซื้อ...] '''
    list_average_dua_date_each_batch, list_process_time = average_dua_date(num_batch, df_item_poor)
    print(f'list_average_dua_date_each_batch = {list_average_dua_date_each_batch}')
    print(f'list_process_time = {list_process_time}')

    '''Step 2 : assignment_batch_to_picker  [...คำนวณหาค่าเฉลี่ยกำหนดวันเวลาส่งมอบสินค้าของแต่ละกลุ่มคำสั่งซื้อ...] '''
    list_picker = [[] for _ in range(num_picker)]
    list_batch_remain = [i for i in range(1, num_batch + 1)]
    list_work_time_picker = [[] for _ in range(num_picker)]
    list_completion_time = [[] for _ in range(num_picker)]
    print(f'list_batch_remain = {list_batch_remain}')

    count = 0
    for i in range(5):

        #print(f'--- Batch run No. {i}')
        if i < num_picker:

            # selection min average dua date of all batch
            min_average_dua_date = min(list_average_dua_date_each_batch)
            # print(f'min_average_dua_date [{i}] = {min_average_dua_date}')
            index_average_dua_date = list_average_dua_date_each_batch.index(min_average_dua_date)
            batch_min = list_batch_remain[index_average_dua_date]

            # คำนวณ completion_time of batch
            time = 0
            df_order = df_item_poor[df_item_poor['batch'] == batch_min]
            completion_time_batch = calculate_completion_time(df_order, time)
            df_item_poor.loc[df_item_poor['batch'] == batch_min, 'CompletionTime'] = completion_time_batch
            max_process_time = df_order['process_time'].max()
            # assignment batch to picker
            list_picker[count].append(batch_min)
            list_work_time_picker[count].append(max_process_time)
            list_completion_time[count].append(completion_time_batch)
            # ลบ batch ที่ถูกมอบหมายออกจาก list
            list_batch_remain.pop(index_average_dua_date)
            list_average_dua_date_each_batch.pop(index_average_dua_date)

        else:


            list_completion_time_check = []
            list_tardiness_each_order_check = []

            for batch in list_batch_remain:
                df_batch_check = df_item_poor.copy()
                '''check work time of each picker [process_time each batch]'''
                list_sum_work_time_picker = []
                for k in range(num_picker):
                    # ผลรวม time working แต่ละ picker
                    sum_work_time_picker = sum(list_work_time_picker[k])
                    # save ผลรวม time working แต่ละ picker to list
                    list_sum_work_time_picker.append(sum_work_time_picker)

                min_work_time_picker = min(list_sum_work_time_picker)
                index_min_time_picker = list_sum_work_time_picker.index(min_work_time_picker)

                # คำนวณ completion_time of all batch
                df_batch = df_batch_check[df_batch_check['batch'] == batch]
                completion_time_batch = calculate_completion_time(df_batch, min_work_time_picker)
                completion_time_batch = float(format(completion_time_batch, '.3f'))
                df_batch_check.loc[df_batch_check['batch'] == batch, 'CompletionTime'] = completion_time_batch
                list_completion_time_check.append(completion_time_batch)

                total_tardiness_check = 0
                for order in range(0, num_order+1):
                    df_order = df_batch_check[df_batch_check['order'] == order]
                    tardiness_each_order = calculate_tardiness_order(df_order, order)
                    if tardiness_each_order < 0:
                        total_tardiness_check += tardiness_each_order

                total_tardiness_check = float(format(total_tardiness_check, '.3f'))
                list_tardiness_each_order_check.append(total_tardiness_check)


            '''ถ้ากรณี tradiness เท่ากัน จะพิจารณา average dua date '''
            max_check = max(list_tardiness_each_order_check)
            list_min_check = []
            list_batch_min_check = []
            list_index_batch_min_check = []
            for i in range(len(list_tardiness_each_order_check)):
                # If the other element is min than first element
                if list_tardiness_each_order_check[i] >= max_check and list_tardiness_each_order_check[i] < 0:
                    min1 = list_tardiness_each_order_check[i]  # It will change
                    list_min_check.append(list_tardiness_each_order_check[i])
                    list_index_batch_min_check.append(i)
                    list_batch_min_check.append(list_batch_remain[i])

            # print(f'list_index_batch_min_check = {list_index_batch_min_check}')
            # print(f'list_batch_min_check = {list_batch_min_check}')
            # print(f'list_min_check = {list_min_check}')

            '''พิจารณา average dua date  '''
            list_index_min_value = []
            list_min_value = []
            for k in list_index_batch_min_check:
                min_value = list_average_dua_date_each_batch[k]
                list_min_value.append(min_value)

            min_tardiness = min(list_min_value)
            index_min_value = list_min_value.index(min_tardiness)

            index_tardiness = list_index_batch_min_check[index_min_value]
            batch_min_tardiness = list_batch_remain[index_tardiness]


            # print(f'min_tardiness = {min_tardiness}')
            # print(f'index_tardiness = {index_tardiness}')
            # print(f'batch_min_tardiness = {batch_min_tardiness}')
            #list_average_dua_date_each_batch

            '''พิจารณา completion_time  '''
            batch_min_completion_time = list_completion_time_check[index_tardiness]

            # print(f'min_tardiness [{index_tardiness}] = {min_tardiness}')
            # print(f'batch_min_tardiness = {batch_min_tardiness}')
            # print(f'batch_min_completion_time = {batch_min_completion_time}')

            '''ลบ Batch ที่ถูกมอบหมายแล้วออก'''
            list_batch_remain.pop(index_tardiness)
            list_average_dua_date_each_batch.pop(index_tardiness)

            '''ลบ Batch ที่ถูกมอบหมายแล้วออก'''
            df_item_poor.loc[df_item_poor['batch'] == batch_min_tardiness, 'CompletionTime'] = batch_min_completion_time
            df_order = df_item_poor[df_item_poor['batch'] == batch_min_tardiness]
            max_process_time = df_order['process_time'].max()

            # assignment batch to picker
            list_picker[index_min_time_picker].append(batch_min_tardiness)
            list_work_time_picker[index_min_time_picker].append(max_process_time)
            list_completion_time[index_min_time_picker].append(batch_min_completion_time)


        count += 1
        if count > num_picker - 1:
            count = 0


    ''' output : 
        examble :   p1 = [B11, B3, B10, B9, B2]
                    p2 = [B6, B7, B1, B4, B8, B5]
                    completion_time
                    p1 = [11.067, 24.8, 37.676, 57.667, 82.434]
                    p2 = [12.9, 32.3, 54.783, 71.916, 80.043, 95.743]
                    pprocess_time
                    p1 = [11.067, 13.733, 12.967, 19.9, 24.767]
                    p2 = [12.9, 19.4, 22.483, 17.133, 8.217, 15.7]'''

    '''calculate tardiness of each order 
                    location    item    category    ...     AverageDuaDate  CompletionTime  TardinessOrder
            44       122        254         1       ...          92.875          57.667         102.709
            38       871        716         0       ...         115.542          80.133         102.709
            24       534        508         0       ...         117.542          95.833         102.709'''
    list_tardiness_each_order = []
    total_tardiness = 0
    for i in range(num_order + 1):
        df_order = df_item_poor[df_item_poor['order'] == i]
        tardiness_each_order = calculate_tardiness_order(df_order, i)
        if tardiness_each_order < 0:
            total_tardiness += tardiness_each_order
        df_item_poor.loc[df_item_poor['order'] == i, 'TardinessOrder'] = tardiness_each_order
        list_tardiness_each_order.append(tardiness_each_order)

    return list_picker, list_tardiness_each_order, total_tardiness, df_item_poor



# list_picker_ESDR, list_tardiness_each_order_ESDR, total_tardiness_ESDR, df_item_poor_ESDR = ESDR_algorithms()
# list_picker, list_tardiness_each_order, total_tardiness, df_item_poor = seed_algorithms()
# list_picker_greedy, list_tardiness_each_order_greedy, total_tardiness_greedy, df_item_poor_greedy = greedy_algorithms()
# print(f'total_tardiness_ESDR = {total_tardiness_ESDR}')
# print(f'total_tardiness = {total_tardiness}')
# print(f'total_tardiness_greedy = {total_tardiness_greedy}')