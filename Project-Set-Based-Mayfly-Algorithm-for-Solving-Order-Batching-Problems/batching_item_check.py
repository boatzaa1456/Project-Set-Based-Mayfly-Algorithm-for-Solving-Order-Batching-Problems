import pandas as pd
import time

value_item_heavy = 100
value_heavy = 40



def batching_A(df_item_poor, list_index_item, *info):
    # จับเวลาการทำงานของกระบวนการ
    start_time = time.time()

    capacity_picker, value_threshold, name_path_input = info

    list_batching_item = []
    list_total_weight_batch = []  # น้ำหนักทั้งหมด พิจารณาสินค้าปัจุบัน
    num_item = len(df_item_poor)  # นับจำนวนสินค้าทั้งหมด

    count_batch = 1  # กำหนดหมายเลข Batch
    df_item_record = pd.DataFrame()  # สร้าง Data Frame ไว้บันทึกข้อมูลทั้งหมด
    list_item_in_batch = []
    dataframe_new = []
    while len(list_index_item) != 0:
        total_weight_batch = 0
        list_item_new = []

        df_item = pd.DataFrame()
        #
        for i in range(0, num_item):
            if i >= len(list_index_item):
                break
            item_current = list_index_item[i]
            data_item_current = df_item_poor.iloc[[list_index_item[i]]]
            weight_item_current = data_item_current['weight'].values
            list_item_new.append(item_current)

            ''' ทำ จนกว่า น้ำหนักรวมสินค้าใน batch >= capacity_picker '''
            total_weight_batch += weight_item_current

            ''' check : capacity condition !!! '''
            if total_weight_batch > capacity_picker:
                list_item_new.remove(item_current)
                break

            ''' Check category condition ของสินค้า 
                ตัวอย่าง :      food = 1
                              nonfood = 0'''
            if weight_item_current >= value_item_heavy:
                ''' *** check : threshold condition !!! 
                        คำถาม : ต้องบวกน้ำหนักตตัวเองไปคิดเลยไหม หรือไม่ ???? '''
                if total_weight_batch > value_threshold:
                    list_item_new.remove(item_current)
                    df_item_poor_drop = df_item[df_item['location'] == item_current].index.values
                    df_item = df_item.drop(df_item_poor_drop)
                    break

            '''สร้าง datafram each batch '''
            df_item = pd.concat([df_item, data_item_current])
            dataframe_new = df_item.sort_values(by=['category', 'self_capacity'], ascending=[True, False])

            list_item_new = dataframe_new.index.values.tolist()
            list_weight_of_self_capacity_current = []
            list_self_capacity = []

            ''' *** check : self capacity condition of each item !!! '''
            for m in range(0, len(dataframe_new)):

                if m == 0:

                    list_self_capacity.append(capacity_picker)
                    weight = dataframe_new['weight'].values[m]
                    total_weight_dataframe_new = weight
                    list_weight_of_self_capacity_current.append(total_weight_dataframe_new)

                else:

                    list_self_capacity.append(dataframe_new['self_capacity'].values[m - 1].sum())
                    total_weight_dataframe_new = dataframe_new['weight'].values[m:len(dataframe_new)].sum()
                    list_weight_of_self_capacity_current.append(total_weight_dataframe_new)

            ''' *** check   : self capacity condition of each item !!! 
                    และ ลบ item ที่ถูกใช้แล้วออกจาก solution space'''
            for m in range(0, len(dataframe_new)):

                if list_weight_of_self_capacity_current[m] > list_self_capacity[m]:
                    list_item_new.remove(item_current)
                    df_item_poor_drop = df_item.index[-1]
                    df_item = df_item.drop(df_item_poor_drop)
                    break

        df_item = df_item.sort_values(by=['category', 'self_capacity'], ascending=[True, False])
        df_item.loc[:, 'batch'] = count_batch
        list_item = []

        for i in range(0, len(df_item)):
            location = df_item['location'].values[i]
            list_item.append(location)

        list_batching_item.append(list_item)
        df_item_record = pd.concat([df_item_record, df_item], ignore_index=True)
        num_columns = len(df_item_record.columns)
        list_zero = [0 for x in range(0, num_columns)]
        df_item_record.loc[len(df_item_record.index)] = list_zero
        total_weight_save = dataframe_new['weight'].values[0:len(list_item_new) + 1].sum()
        list_total_weight_batch.append(total_weight_save)

        list_item_in_batch.append(list_item_new)
        list_index_item = [ele for ele in list_index_item if ele not in list_item_new]

        ''' list item poor = [ ]'''
        if not list_index_item:
            break

        count_batch += 1

    ''' บันทึกข้อมูล Batch '''
    # path_1 = 'output' + '\\Output_' + str(name_path_input) + '_Dataframe_solution.csv'
    # df_item_record.to_csv(path_1, index=False)

    # ''' Check number of item in lists'''
    # count = 0
    # for element in list_batching_item:
    #     count += len(element)
    #
    # count_1 = 0
    # for element in list_item_in_batch:
    #     count_1 += len(element)

    end_time = time.time()
    total_time_run = end_time - start_time
    total_time_run = format(total_time_run, '.3f')
    # print('total time run :', total_time_run, 'seconds')

    return list_batching_item, df_item_record, list_item_in_batch


def batching_B(df_item_poor, list_index_item, *info):
    '''Input : list_index_item = [3, 5, 7, 1, 9, 8, 6, 2, 0, 4]
             :   df_item_poor =
               index    location  item     category  weight    self_capacity     duedate     order
                52       731        314         1       5             19          10.5417     19
            '''

    ''' ข้อมูลภายนอก '''
    # ความสามารถในการหยิบสินค้าของพนักงาน (capacity_picker)
    # ความสามารถในการรับสินค้าหนักของอุปกรณ์หยิบสินค้า (value_threshold)
    # ชื่อไฟล์โจทย์ในการทดลอง (name_path_input)
    capacity_picker, value_threshold, name_path_input = info

    ''' ค่ากำหนดสินค้าว่าป็นสินค้าหนักหรือสินค้าเบา (Item ที่มีน้ำหนักมากกว่า 40 เป็นสินค้าหนัก) 
    (value_heavy = 40)  '''

    df_item_record = pd.DataFrame()  # สร้างตัวแปร Dateframe
    num_item = len(df_item_poor)  # นับจำนวนสินค้าทั้งหมด

    list_batching_item, list_item_in_batch = [], []
    # พิจารณาที่ละ item
    for i in range(num_item):
        '''Index สินค้าที่ถูกพิจารณาปัจจุบัน '''
        item_current = list_index_item[i]
        '''ข้อมูลสินค้าปัจจุบัน
        EX :    index    location  item     category  weight    self_capacity     duedate     order
                52       731        314         1       5             19          10.5417     19'''
        data_item_current = df_item_poor.iloc[[list_index_item[i]]]
        ''' น้ำหนักสินค้าปัจจุบัน '''
        weight_item_current = data_item_current['weight'].values
        data_item_current = data_item_current.copy()
        data_item_current.loc[item_current, 'batch'] = 1
        df_item_record = pd.concat([df_item_record, data_item_current]).astype(int)

        while True:

            ''' ตรวจสอบ 'จำนวน Batch' ที่มีอยู่ปัจจุบันทั้งหมด มีกี่ Batch
                        ถ้า สินค้า (i) มากกว่า 1 ไปเช็ค df_item_record แถวสุดท้าย '''
            count_batch = df_item_record['batch'].values[-1]
            list_weight_each_batch, list_batch_remain = [], []

            for j in range(1, int(count_batch) + 1):

                conditions = 0
                '''ตรวจสอบ น้ำหนัก Item รวมของแต่ละ Batch '''
                weight_each_batch = df_item_record.loc[df_item_record['batch'] == j, 'weight'].sum()
                list_weight_each_batch.append(weight_each_batch)

                ''' ใส่ Item ลงไปใน  Batch น้ำหนัก มากเกิน ความสามารถในการหยิบของพนักงานหรือไหม'''
                if weight_each_batch > capacity_picker:
                    df_item_record.loc[item_current, 'batch'] = j + 1
                    ''' ละเมิดเงื่อนไข '''
                    conditions += 1
                    continue

                ''' เงื่อนไขที่ 1  ด้านน้ำหนักสินค้าถ้าสินค้าปัจจุบันเป็นสินค้าหนัก(Heavy item) ในปัจจุบัน
                    และน้ำหนักทั้งหมดใน Batch >  ค่าความสามารถในหารรับสินค้าของอุปกรณ์ (Threshold) ให้สินค้าปัจจุบันไปอยู่ Batch ใหม่'''
                if weight_item_current >= value_item_heavy:
                    ''' รวมน้ำหนักของสินค้าหนักใน batch '''
                    df_item_batch_current = df_item_record.loc[df_item_record['batch'] == j]
                    weight_item_heavy = df_item_batch_current.loc[
                        df_item_batch_current['weight'] >= value_heavy, 'weight'].sum()
                    if weight_item_heavy > value_threshold:
                        df_item_record.loc[item_current, 'batch'] = j + 1
                        ''' ละเมิดเงื่อนไข '''
                        conditions += 1
                        continue

                '''เงื่อนไขที่ 2 ด้านหมวดหมู่สินค้า --> จัดเรียงสินค้าตามหมวดหมู่สินค้า'''
                '''สร้าง df_batch เพื่อเช็คเงื่อนไข batch โดยการคัดลอกข้อมูล item เฉพาะ Batch ที่ j '''
                df_batch = df_item_record[df_item_record['batch'] == j].copy()
                '''จัดเรียงสินค้าใน data frame โดยแบ่งกลุ่มหวดหมวด (category) 0 = Nonfood, 1 = food
                    และแต่ละกลุ่ม เรียงตามความสามารถในการรับน้ำหนักมากไปหาน้ำหนักน้อย'''
                df_batch = df_batch.sort_values(by=['category', 'self_capacity'], ascending=[True, False])
                ''' เงื่อนไขที่ 3 ด้านความเปราะ -->
                    Ex. list self capacity ของแต่ละ Item เช่น '''
                list_self_capacity, list_total_weight_item_before = [], []
                num_i = len(df_batch)
                for item in range(num_i):
                    if item == 0:
                        self_capacity_item_before = capacity_picker
                        list_self_capacity.append(self_capacity_item_before)
                        '''น้ำหนักรวมที่ item[0] จนถึง item[item-num_i] (น้ำหนักรวมจนถึง item ก่อนหน้าที่พิจารณา) '''
                        total_weight_item = df_batch['weight'].values[:num_i - item].sum()
                        list_total_weight_item_before.append(total_weight_item)
                    else:
                        self_capacity_item_before = df_batch['self_capacity'].values[item - 1]
                        list_self_capacity.append(self_capacity_item_before)
                        '''น้ำหนักรวมที่ item[0] จนถึง item[item-num_i] (น้ำหนักรวมจนถึง item ก่อนหน้าที่พิจารณา) '''
                        total_weight_item = df_batch['weight'].values[item:num_i].sum()
                        list_total_weight_item_before.append(total_weight_item)
                    ''' นำ list ที่ถูกบันทึกไว้ มาเปรียบเทียบ
                            Ex. list self capacity ของแต่ละ Item เช่น
                            list_self_capacity = [100, 39, 21, 18, 15, 46, 29]
                            list_total_weight_item_before = [28, 27, 20, 10, 8, 6, 3] '''
                    if list_total_weight_item_before[item] > list_self_capacity[item]:
                        '''ถ้าละเมิดเงื่อนไขทุก Batch และถ้า batch นั้นเป็น batch สุดท้าย  ให้ Item นั้น อยู่ใน batch เปิดใหม่'''
                        df_item_record.loc[item_current, 'batch'] = j + 1
                        conditions += 1
                        continue

            ''' ถ้าไม่มีละเมิดเงื่อนไข ทำการพิจารณา item ถัดไป '''
            if conditions == 0:
                break

    df_item_record = df_item_record.sort_values(by=['batch', 'category', 'self_capacity'],
                                                ascending=[True, False, True])

    '''แปลงข้อมูลใน Data frame ลง list'''
    num_batch = df_item_record['batch'].iloc[-1]
    for run in range(1, int(num_batch) + 1):
        marks_list = df_item_record[df_item_record['batch'] == run]
        list_batching_item.append(marks_list['location'].values.tolist())
        list_item_in_batch.append(marks_list.index.tolist())

    ''' Output Function
    list_batching_item = [[242, 776, 870, 534, 526, 109, 806, 895], [871, 122]]
    list_item_in_batch = [[8, 3, 9, 1, 6, 7, 4, 5], [2, 0]]'''
    return list_batching_item, df_item_record, list_item_in_batch


def bisect_left_desc(a, x, lo=0, hi=None):
    if hi is None: hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] > x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def bisect_right_desc(a, x, lo=0, hi=None):
    if hi is None: hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] >= x:
            lo = mid + 1
        else:
            hi = mid
    return lo
def join_lists_concatenation(lists):
    new_list = []
    for list in lists:
        new_list+= list
    return new_list

def batching_open(df_item_pool, list_index_item, *info):  # อย่าลืมใส่ heavy_item_set ตอนเรียกใช้

    '''Input : list_index_item = [3, 5, 7, 1, 9, 8, 6, 2, 0, 4]
             :   df_item_poor =
               index    location  item     category  weight    self_capacity     duedate     order
                52       731        314         1       5             19          10.5417     19
            '''
    ''' ข้อมูลภายนอก '''
    # ความสามารถในการหยิบสินค้าของพนักงาน (capacity_picker)
    # ความสามารถในการรับสินค้าหนักของอุปกรณ์หยิบสินค้า (value_threshold)
    # ชื่อไฟล์โจทย์ในการทดลอง (name_path_input)
    capacity_picker, value_threshold, heavy_item_set, name_path_input = info

    ''' ค่ากำหนดสินค้าว่าป็นสินค้าหนักหรือสินค้าเบา (Item ที่มีน้ำหนักมากกว่า 40 เป็นสินค้าหนัก) 
    (value_heavy = 40)  '''

    df_item_record = pd.DataFrame()  # สร้างตัวแปร Dateframe
    num_item = len(df_item_pool)  # นับจำนวนสินค้าทั้งหมด
    count_batch = 1
    weight_each_batch = [0]
    weight_heavy_each_batch = [0]
    list_batch_by_index_category = [[[0], [0]]] # [[[19,32][14]]]]
    list_batch_by_self_cap_index_category = [[[], []]]  # [[[(100,19),(80,32)],[70,14]]]
    list_batch_by_total_weight_above = [[[0], [0]]]   # [[[90, 70], [50]]]
    #list_index_item = [13, 12, 22, 41, 9, 23, 28, 5, 32, 11, 42, 38, 21, 6, 43, 45, 40, 17, 2, 33, 15, 34, 30, 35, 36, 4, 20, 16, 18, 14, 27, 49, 3, 29, 24, 0, 7, 47, 46, 31, 37, 19, 44, 39, 26, 1, 48, 8, 10, 25]
    #list_index_item = [8, 15, 18, 12, 1, 11, 16, 14, 19, 4, 5, 13, 10, 9, 0, 6, 7, 3, 17, 2]
    #print(f'list_index_item = {list_index_item}')
    # พิจารณาที่ละ item, num_item

    for i in range(num_item):
        '''Index สินค้าที่ถูกพิจารณาปัจจุบัน '''
        item_current = list_index_item[i]
        # print(f'------------')
        # print(f'item_current = {item_current}')
        '''ข้อมูลสินค้าปัจจุบัน
        EX :    index    location  item     category  weight    self_capacity     duedate     order
                52       731        314         1       5             19          10.5417     19'''
        data_item_current = df_item_pool.iloc[[item_current]]
        # print(data_item_current)
        ''' น้ำหนักสินค้าปัจจุบัน '''
        weight_item = data_item_current['weight'].values
        weight_item_current = weight_item[0]

        # data_item_current = data_item_current.copy()
        #data_item_current.loc[item_current, 'batch'] = 1
        #df_item_record = pd.concat([df_item_record, data_item_current]).astype(int)

        current_batch = 1
        while True:

            for j in range(0, int(count_batch)):
                # print(f'batch = {j}')
                # print(f'weight_each_batch = {weight_each_batch}')
                # print(f'weight_heavy_each_batch[{j}] = {weight_heavy_each_batch[j]}')
                # print(f'------------------')
                # print(f'current_batch = {current_batch}')
                conditions = 0
                #  1 ตรวจสอบว่าน้ำหนักเกินความสามารถพนักงานหรือไม่

                # if weight_each_batch[j] + weight_item_current > capacity_picker:
                #     # print(f' ++ ความจุเกิน ++++')
                #     conditions = 1
                #     continue
                sum_weight_c = 0
                for i in range(2):
                    for item in list_batch_by_self_cap_index_category[j][i]:
                        sum_weight_c = sum_weight_c + df_item_pool.iloc[item[1]]['weight']

                if sum_weight_c + weight_item_current > capacity_picker:
                    conditions = 1
                    continue

                # 2 ตรวจสอบว่าเป็นสินค้าหนักหรือเปล่า ถ้าเป็นสินค้าหนักน้ำหนักรวมต้องไม่เกิน value threshold
                if df_item_pool.iloc[item_current]['category'] == 0:
                    sum_weight_heavy = 0
                    for item in list_batch_by_self_cap_index_category[j][0]:
                        sum_weight_heavy = sum_weight_heavy + df_item_pool.iloc[item[1]]['weight']

                    if sum_weight_heavy + weight_item_current > value_item_heavy:
                        conditions = 1
                        continue
                # if item_current in heavy_item_set:
                #     if weight_heavy_each_batch[j] + weight_item_current > value_threshold:
                #         # print(f' ++ สินค้าหนักเกิน ++++')
                #         conditions = 1
                #         continue
                # 3 ตรวจสอบความเปราะ
                # แยกชนิดของสินค้าปัจจุบันก่อน

                # ---> สินค้าเป็น Non food
                if data_item_current['category'].values == 0:
                    #print(f'-- Non Food --')
                    # ค่าความสามารถในการรับน้ำหนัก Item ปัจจุบัน
                    values_self_capacity = data_item_current['self_capacity'].values
                    insert_pos = bisect_left_desc(list_batch_by_self_cap_index_category[j][0],
                                                  (data_item_current['self_capacity'].values, item_current))
                    # บันทึกข้อมูลลง list [[(ความสามารถในการรับน้ำหนัก, Index Item)],[]] ; ตัวอย่าง [[[(100,19),(80,32)],[70,14]]]
                    list_batch_by_self_cap_index_category[j][0].insert(insert_pos,
                                                (values_self_capacity[0], item_current))

                    new_list_total_weight_above = [list_batch_by_total_weight_above[j][b][:] for b in range(len(list_batch_by_total_weight_above[j]))]

                    #print(f'NON Food; new_list_total_weight_above = {new_list_total_weight_above}')

                    if (insert_pos < (len(list_batch_by_total_weight_above[j][0]))):
                        weight_above_new_item = new_list_total_weight_above[0][insert_pos] + df_item_pool.iloc[list_batch_by_self_cap_index_category[j][0][insert_pos][1]]['weight']
                    else:
                        weight_above_new_item = new_list_total_weight_above[1][0] + df_item_pool.iloc[list_batch_by_self_cap_index_category[j][0][insert_pos][1]]['weight']

                    new_list_total_weight_above[0].insert(insert_pos,weight_above_new_item)
                    #print(f'NON Food; weight_above_new_item  = {weight_above_new_item}')
                    #print(f'NON Food; new_list_total_weight_above  = {new_list_total_weight_above}')
                    #print(f'NON Food; list_batch_by_self_cap_index_category = {list_batch_by_self_cap_index_category}')

                    # Update น้ำหนัก กลุ่ม Non Food
                    for pos in range(insert_pos):
                        new_list_total_weight_above[0][pos] += weight_item_current

                    #print(f'NON Food;  new_list_total_weight_above  = {new_list_total_weight_above}')

                    for pos in range(insert_pos+1): #แก้ไขตรงนี้ 21-03-2567
                        if pos == insert_pos+1:
                            if (new_list_total_weight_above[0][pos]) > \
                                    list_batch_by_self_cap_index_category[j][1][pos][0]:
                                del list_batch_by_self_cap_index_category[j][0][insert_pos]
                                # print(f' ++ ความเปราะ non food ++++')
                                conditions = 1
                                break
                        else:
                            if (new_list_total_weight_above[0][pos+1]) > \
                                    list_batch_by_self_cap_index_category[j][0][pos][0]:
                                del list_batch_by_self_cap_index_category[j][0][insert_pos]
                                # print(f' ++ ความเปราะ non food ++++')
                                conditions = 1
                                break


                    #print (f'NON Food; new_list_total_weight_above = {new_list_total_weight_above}')
                    #print(f'NON Food; list_batch_by_self_cap_index_category = {list_batch_by_self_cap_index_category}')

                #---> สินค้าเป็น food
                else:
                    #print(f'-- Food --')
                    values_self_capacity = data_item_current['self_capacity'].values
                    # ตำแหน่งที่ต้องการแทรก
                    insert_pos = bisect_left_desc(list_batch_by_self_cap_index_category[j][1],
                                                  (data_item_current['self_capacity'].values, item_current))
                    # [[[(100,19),(80,32)],[70,14]]]
                    list_batch_by_self_cap_index_category[j][1].insert(insert_pos, (values_self_capacity[0], item_current))

                    #[[0], [0]]
                    #print(f'Food; list_batch_by_self_cap_index_category = {list_batch_by_self_cap_index_category}')
                    new_list_total_weight_above = [list_batch_by_total_weight_above[j][b][:] for b in
                                                   range(len(list_batch_by_total_weight_above[j]))]

                    #print(f'Food;  new_list_total_weight_above = {new_list_total_weight_above}')
                    if (insert_pos < (len(list_batch_by_total_weight_above[j][1]))):
                        weight_above_new_item = new_list_total_weight_above[1][insert_pos] + \
                                                df_item_pool.iloc[list_batch_by_self_cap_index_category[j][1][insert_pos][1]]['weight']
                    else:
                        weight_above_new_item = 0

                    new_list_total_weight_above[1].insert(insert_pos, weight_above_new_item)
                    #print(f'Food;  insert_pos = {insert_pos}')
                    #
                    #
                    #print(f'Food;  new_list_total_weight_above = {new_list_total_weight_above}')

                    # น้ำหนักกลุ่ม Non Food
                    for pos in range(len(list_batch_by_self_cap_index_category[j][0])+1):
                        #print(f'pos - {pos}')
                        new_list_total_weight_above[0][pos] += weight_item_current

                    # น้ำหนักกลุ่ม Food
                    for pos in range(0, insert_pos):
                        new_list_total_weight_above[1][pos] += weight_item_current

                    #print(f'Food;  total_weight_above = {new_list_total_weight_above}')
                    #print(f'Food;  list_batch_by_self_cap_index_category = {list_batch_by_self_cap_index_category}')
                    #
                    # print(f'==')
                    # print(f' สมาชิกใน non food = {len(list_batch_by_self_cap_index_category[j][0])}')
                    # print(f'new_list_total_weight_above[0][pos] = {new_list_total_weight_above[0]}')
                    for pos in range(len(list_batch_by_self_cap_index_category[j][0])):
                        if (new_list_total_weight_above[0][pos+1]) > \
                                list_batch_by_self_cap_index_category[j][0][pos][0]:
                            del list_batch_by_self_cap_index_category[j][1][insert_pos]
                            #print(f' ++ ความเปราะ non food ใน food ++++')
                            #print(f'ละเมิด non food')
                            conditions = 1
                            break

                    if conditions:
                        continue

                    for pos in range(insert_pos+1):
                        if (new_list_total_weight_above[1][pos]) > \
                                list_batch_by_self_cap_index_category[j][1][pos][0]:
                            del list_batch_by_self_cap_index_category[j][1][insert_pos]
                            #print(f' ++ ความเปราะ food ใน food ++++')
                            #print(f'ละเมิด food')
                            conditions = 1
                            break


                    #print(f'list_batch_by_self_cap_index_category = {list_batch_by_self_cap_index_category}')
                # ---> สินค้าเป็น food


                if conditions:
                     continue

                if not conditions:  # ถ้า conditons เป็น 0
                    weight_each_batch[j] += weight_item_current
                    if item_current in heavy_item_set:
                        weight_heavy_each_batch[j] += weight_item_current

                    if data_item_current['category'].values == 0:  # non_food
                        list_batch_by_index_category[j][0].insert(insert_pos, item_current)
                        # for pos in range(insert_pos):
                        #     list_batch_by_total_weight_above[j][0][pos] += weight_item_current
                    else:  # food
                        list_batch_by_index_category[j][1].insert(insert_pos, item_current)
                        # for pos in range(len(list_batch_by_index_category[j][0])):
                        #     list_batch_by_total_weight_above[j][0][pos] += weight_item_current
                        # for pos in range(insert_pos):
                        #     list_batch_by_total_weight_above[j][1][pos] += weight_item_current
                    list_batch_by_total_weight_above[j] = new_list_total_weight_above
                    break
            # print(f'new_list_total_weight_above = {list_batch_by_total_weight_above}')
            # print(f'list_batch_by_self_cap_index_category = {list_batch_by_self_cap_index_category}')
            # print(f'condition = {conditions}')
            if conditions:
                count_batch += 1
                if data_item_current['category'].values == 0:
                    values_self_capacity = data_item_current['self_capacity'].values
                    list_batch_by_index_category.append([[item_current],[]])
                    list_batch_by_self_cap_index_category.append([[(values_self_capacity[0],item_current)],[]])
                    list_batch_by_total_weight_above.append([[weight_item_current,0], [0]])

                else:
                    values_self_capacity = data_item_current['self_capacity'].values
                    list_batch_by_index_category.append([[], [item_current]])
                    list_batch_by_self_cap_index_category.append(
                        [[], [(values_self_capacity[0], item_current)]])
                    list_batch_by_total_weight_above.append([[weight_item_current], [weight_item_current,0]])

                weight_each_batch.append(weight_item_current)
                if item_current in heavy_item_set:
                    weight_heavy_each_batch.append(weight_item_current)
                else:
                    weight_heavy_each_batch.append(0)
                break

            if not conditions:
                break


           # ยังไม่เสร็จ App ไปทำต่อให้ return ค่าได้เหมือน batching_open ตัวเดิม 31/1/2024 16:18
    # print(f'list_batch_by_self_cap_index_category = {list_batch_by_self_cap_index_category}')
    # print(f'batch = {len(list_batch_by_index_category)}')
    # print(f'list_batch_by_index_category = {list_batch_by_index_category}')
    '''แปลง ผลลัพท์ที่ได้ ให้สัมพันธ์กับ function อื่น '''
    ''' Output Function : return  list_batching_item, df_item_record, list_item_in_batch
            1) list_batching_location = [[242, 776, 870, 534, 526, 109, 806, 895], [871, 122]]
            2) df_item_record
            3) list_item_in_batch = [[8, 3, 9, 1, 6, 7, 4, 5], [2, 0]]'''

    '''3) Convert output (list_batch_by_index_category) to list_item_in_batch = [[8, 3, 9, 1, 6, 7, 4, 5], [2, 0]] '''
    list_item_in_batch = []
    num_batch = len(list_batch_by_self_cap_index_category)
    for batch in range(num_batch):
        item = []
        for f in range(len(list_batch_by_self_cap_index_category[batch])):
            for index in list_batch_by_self_cap_index_category[batch][f]:
                item.append(index[1])
        list_item_in_batch.append(item)

    # for run_batch in range(len(list_batch_by_index_category)):
    #     # new_list = join_lists_concatenation(list_batch_by_index_category[run_batch])
    #     # list_item_in_batch.append(new_list)


    '''2) Convert list_item_in_batch to Dataframe : (df_item_record)'''
    # list_batching_location = [[] for _ in range(len(list_item_in_batch))]
    # df_item_record = pd.DataFrame()
    # for run_batch in range(len(list_item_in_batch)):
    #     for run_index in range(len(list_item_in_batch[run_batch])):
    #         df_item_record = pd.concat([df_item_record, df_item_pool.iloc[[list_item_in_batch[run_batch][run_index]]]])
    #         '''Add  หมายเลข  batch ของ item ลง dataframe ใน column 'batch'
    #                 location  item  category  weight  self_capacity   duedate  order  batch
    #         14       825   788         0      47            150  208.3960      5        1
    #         1        334   797         0      43            150   27.3958      0        1
    #         19       358   542         0       9             47   12.3958      7        1   '''
    #         df_item_record.loc[list_item_in_batch[run_batch][run_index], 'batch'] = run_batch+1
    #         df_item_record = df_item_record.astype({'batch': int})
    #         '''3) Convert df_item_record to list_batching_location = [[242, 776, 870, 534, 526, 109, 806, 895], [871, 122] '''
    #         list_batching_location[run_batch].append(df_item_pool.loc[list_item_in_batch[run_batch][run_index], 'location'])

    #From Aj Pop
    list_batching_location = [[] for _ in range(len(list_item_in_batch))]
    list_to_create_df_item_record = []
    list_of_indices = []
    for run_batch in range(len(list_item_in_batch)):
        for run_index in range(len(list_item_in_batch[run_batch])):
            list_batching_location[run_batch].append(df_item_pool.loc[list_item_in_batch[run_batch][run_index], 'location'])
            new_index = df_item_pool.iloc[list_item_in_batch[run_batch][run_index]].name
            new_row = df_item_pool.iloc[list_item_in_batch[run_batch][run_index]].tolist() + [int(run_batch+1)]
            #new_row = list_item_in_batch[run_batch][run_index].tolist() + [int(run_batch+1)]
            list_to_create_df_item_record.append(new_row)
            list_of_indices.append(new_index)

    columns = ['location', 'item', 'category', 'weight', 'self_capacity', 'duedate', 'order', 'batch']
    df_item_record = pd.DataFrame(list_to_create_df_item_record, columns=columns, index=list_of_indices)
    df_item_record = df_item_record.astype({'location': int, 'item' : int, 'category' : int, 'order' : int})
    #ฟังก์ชัน ตรวจสอบเงื่อนไข จะ Print รายงานผล เฉพาะ Error เท่านั้น
    check = check_feasibility(list_item_in_batch, df_item_pool,capacity_picker)
    if check == False:
        print(f'list_index_item = {list_index_item}')
        print(f'list_item_in_batch = {list_item_in_batch}')

    return list_batching_location, df_item_record, list_item_in_batch

def check_feasibility(list_batch_check,df_item_check,capacity_picker):
    num_batch=len(list_batch_check)
    for batch in range(num_batch):# เช็ค Capacity_picker แต่ละ Batch
        sum_w = 0
        for item in list_batch_check[batch]:
            sum_w = sum_w + df_item_check.iloc[item]['weight']
        if sum_w > capacity_picker:
            print(f'CHECK --> Batch No. {batch}; Capacity_Error ')
        #else:
            #print(f'CHECK --> Batch No. {batch}; Capacity_Pass ')
    for batch in range(num_batch):# เช็ค category แต่ละ Batch
        list_category = []
        for item in list_batch_check[batch]:
            list_category.append(df_item_check.iloc[item]['category'])
        for c in range(len(list_category)-1):
            if list_category[c] > list_category[c+1]:
                print(f'CHECK --> Batch No. {batch}; Category_Error ')
                return False
            #else:
                #print(f'CHECK --> Batch No. {batch}; Category_Pass ')

    for batch in range(num_batch): # เช็ค value_Heavy ของ Item แต่ละ Batch
        sum_heavy = 0
        for item in list_batch_check[batch]:
            if df_item_check.iloc[item]['weight'] > value_heavy:
                sum_heavy = sum_heavy + df_item_check.iloc[item]['weight']
        if sum_heavy > value_item_heavy:
            print(f'CHECK --> Batch No. {batch}; Value Heavy_Error ')
            return False
        #else:
            #print(f'CHECK --> Batch No. {batch}; Value Heavy_Pass ')

    for batch in range(num_batch): #เช็ค Self Capacity แต่ละ Batch
        for i in range(len(list_batch_check[batch])):
            sum_weight = 0
            for item in list_batch_check[batch][i+1:]:
                sum_weight = sum_weight + df_item_check.iloc[item]['weight']
            self_item_cru = df_item_check.iloc[list_batch_check[batch][i]]['self_capacity']
            if self_item_cru < sum_weight:
                print(f'CHECK --> Batch No. {batch}, Item No. {list_batch_check[batch][i]} ; Self Capacity_Error ')
                return False
            #else:
                #print(f'CHECK --> Batch No. {batch}, Item No. {list_batch_check[batch][i]} ; Self Capacity_Pass ')
    return


# else:
# print(f'CHECK --> Batch No. {batch}; Category_Pass ')
# else:
# print(f'CHECK --> Batch No. {batch}; Value Heavy_Pass ')
# else:
# print(f'CHECK --> Batch No. {batch}, Item No. {list_batch_check[batch][i]} ; Self Capacity_Pass ')
def end():
    return