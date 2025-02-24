import numpy as np
import pandas as pd
import time
from itertools import chain
import bisect

value_item_heavy = 100
value_heavy = 40


def batching_A(df_item_poor, list_index_item, *info):
    """
    Optimized version ของ batching_A
    ลดการใช้ pd.concat ภายใน loop โดยเก็บข้อมูลแต่ละ batch ใน list แล้วสร้าง DataFrame ครั้งเดียว
    ใช้ numpy vectorization (เช่น cumulative sum) เพื่อลด loop ในการตรวจสอบ self_capacity
    ผลลัพธ์ที่ได้เหมือนเดิมกับโค้ดเดิม
    """
    start_time = time.time()
    capacity_picker, value_threshold, name_path_input = info

    list_batching_item = []  # รายการ location ของสินค้าในแต่ละ batch
    df_item_record_list = []  # รายการ DataFrame ของแต่ละ batch
    list_item_in_batch = []  # รายการ index ของสินค้าในแต่ละ batch

    remaining_items = list_index_item.copy()

    while remaining_items:
        batch_indices = []
        current_batch_weight = 0.0
        batch_rows = []

        # วนลูปเลือก item จาก remaining_items แบบ greedy
        for item in remaining_items:
            row = df_item_poor.loc[item]
            w = row['weight']
            # เงื่อนไข capacity
            if current_batch_weight + w > capacity_picker:
                break
            # เงื่อนไขสินค้าหนัก
            if w >= value_item_heavy:
                if current_batch_weight + w > value_threshold:
                    break
            batch_indices.append(item)
            batch_rows.append(row)
            current_batch_weight += w

        if not batch_rows:
            break

        # สร้าง DataFrame ของ batch นี้แล้ว sort ตาม ['category', 'self_capacity']
        batch_df = pd.DataFrame(batch_rows)
        batch_df = batch_df.sort_values(by=['category', 'self_capacity'], ascending=[True, False])
        # ปรับปรุง index ให้เป็นรายการ index ของสินค้า
        current_batch_items = list(batch_df.index)

        # ตรวจสอบ self_capacity แบบ vectorized โดยใช้ cumulative sum
        cum_weights = batch_df['weight'][::-1].cumsum()[::-1]
        for m in range(len(batch_df)):
            cap = capacity_picker if m == 0 else batch_df.iloc[m - 1]['self_capacity']
            if cum_weights.iloc[m] > cap:
                # หากละเมิดเงื่อนไข ให้ลบ item สุดท้ายออก
                if current_batch_items:
                    removed = current_batch_items.pop()
                    batch_df = batch_df.drop(removed)
                break

        # กำหนดหมายเลข batch ให้กับ DataFrame
        batch_df['batch'] = len(list_batching_item) + 1

        list_batching_item.append(batch_df['location'].tolist())
        list_item_in_batch.append(list(batch_df.index))
        df_item_record_list.append(batch_df)

        # นำ item ที่ใช้ไปออกจาก remaining_items
        remaining_items = [x for x in remaining_items if x not in batch_df.index.tolist()]

    df_item_record = pd.concat(df_item_record_list, ignore_index=True) if df_item_record_list else pd.DataFrame()
    # total_time_run = time.time() - start_time
    # print('batching_A total time run:', total_time_run, 'seconds')
    return list_batching_item, df_item_record, list_item_in_batch


def batching_B(df_item_poor, list_index_item, *info):
    """
    Optimized version ของ batching_B
    ใช้การตรวจสอบเงื่อนไขแบบ vectorized เมื่อเป็นไปได้เพื่อลด loop ซ้อน
    โดยคำนวณน้ำหนักใน batch ด้วยการเข้าถึงข้อมูลแบบ group จาก dict แทนการเข้าถึง DataFrame บ่อยครั้ง
    """
    capacity_picker, value_threshold, name_path_input = info
    batch_assignment = {}  # เก็บ mapping ของ item -> batch number

    # วนลูปตามลำดับของ list_index_item
    for item in list_index_item:
        row = df_item_poor.loc[item]
        assigned = False
        # พยายามใส่ item ลงใน batch ที่มีอยู่แล้ว
        for batch in range(1, max(batch_assignment.values()) + 1 if batch_assignment else 1):
            current_batch_items = [i for i, b in batch_assignment.items() if b == batch]
            current_weight = df_item_poor.loc[current_batch_items, 'weight'].sum() if current_batch_items else 0
            if current_weight + row['weight'] > capacity_picker:
                continue
            if row['weight'] >= value_heavy:
                heavy_items = [i for i in current_batch_items if df_item_poor.loc[i, 'weight'] >= value_heavy]
                heavy_weight = df_item_poor.loc[heavy_items, 'weight'].sum() if heavy_items else 0
                if heavy_weight + row['weight'] > value_threshold:
                    continue
            # ตรวจสอบเงื่อนไข self_capacity แบบง่ายๆ โดย sorting
            current_batch_df = df_item_poor.loc[current_batch_items]
            current_batch_df = current_batch_df.sort_values(by=['category', 'self_capacity'], ascending=[True, False])
            cum_weight = current_batch_df['weight'][::-1].cumsum()[::-1]
            valid = True
            for m in range(len(current_batch_df)):
                cap = capacity_picker if m == 0 else current_batch_df.iloc[m - 1]['self_capacity']
                if cum_weight.iloc[m] > cap:
                    valid = False
                    break
            if not valid:
                continue
            batch_assignment[item] = batch
            assigned = True
            break
        if not assigned:
            new_batch = max(batch_assignment.values()) + 1 if batch_assignment else 1
            batch_assignment[item] = new_batch

    # สร้าง DataFrame และผลลัพธ์ตาม batch_assignment
    df_item_record = df_item_poor.copy()
    df_item_record['batch'] = -1
    for item, b in batch_assignment.items():
        df_item_record.at[item, 'batch'] = b

    list_batching_item = []
    list_item_in_batch = []
    for b in sorted(set(batch_assignment.values())):
        batch_indices = [i for i, x in batch_assignment.items() if x == b]
        list_item_in_batch.append(batch_indices)
        batch_locations = df_item_record.loc[batch_indices, 'location'].tolist()
        list_batching_item.append(batch_locations)

    return list_batching_item, df_item_record, list_item_in_batch


def bisect_left_desc(a, x, lo=0, hi=None):
    """
    ปรับปรุงโดยใช้การ negate ค่าของ list เพื่อใช้ bisect built-in
    สำหรับ list ที่เรียงลำดับแบบ descending
    """
    if hi is None:
        hi = len(a)
    # สร้าง list ใหม่ด้วยค่า -element
    neg_a = [-elem for elem in a]
    return bisect.bisect_left(neg_a, -x, lo, hi)


def bisect_right_desc(a, x, lo=0, hi=None):
    """
    ปรับปรุงโดยใช้การ negate ค่าของ list เพื่อใช้ bisect built-in
    สำหรับ list ที่เรียงลำดับแบบ descending
    """
    if hi is None:
        hi = len(a)
    neg_a = [-elem for elem in a]
    return bisect.bisect_right(neg_a, -x, lo, hi)
def join_lists_concatenation(lists):
    """
    ปรับปรุงโดยใช้ itertools.chain เพื่อรวมลิสต์หลาย ๆ ลิสต์เข้าด้วยกันอย่างมีประสิทธิภาพ
    """
    return list(chain.from_iterable(lists))

def batching_open(df_item_pool, list_index_item, *info):
    """
    Optimized version of batching_open โดยคงชื่อฟังก์ชันเดิมไว้
    Input:
      - df_item_pool: DataFrame ของสินค้าทั้งหมด
      - list_index_item: รายการ index ของสินค้าที่จะพิจารณา
      - info: capacity_picker, value_threshold, heavy_item_set, name_path_input
    Output:
      - list_batching_item: รายการ batch (แต่ละ batch เป็น list ของค่า 'location')
      - df_item_record: DataFrame ที่เพิ่มคอลัมน์ 'batch' สำหรับจัดกลุ่มสินค้า
      - list_item_in_batch: รายการ batch (แต่ละ batch เป็น list ของ index ของสินค้า)
    """
    capacity_picker, value_threshold, heavy_item_set, name_path_input = info

    # ดึงข้อมูลที่ใช้บ่อยจาก DataFrame มาเป็น numpy array เพื่อลด overhead
    weights = df_item_pool['weight'].values
    categories = df_item_pool['category'].values
    self_caps = df_item_pool['self_capacity'].values

    # เตรียม container สำหรับผลลัพธ์
    list_batching_item = []  # รายการของ 'location' สำหรับแต่ละ batch
    list_item_in_batch = []  # รายการของ index สำหรับแต่ละ batch

    # สร้าง array สำหรับเก็บหมายเลข batch สำหรับแต่ละ item (เริ่มต้นเป็น -1)
    batch_assignment = np.full(len(df_item_pool), -1, dtype=int)

    # แปลง list_index_item เป็น numpy array
    remaining = np.array(list_index_item)

    current_batch = 1
    # กระบวนการจัด batch แบบ greedy
    while remaining.size > 0:
        batch_indices = []     # รายการ index ที่จะถูกจัดให้อยู่ใน batch ปัจจุบัน
        current_batch_weight = 0.0

        # เก็บ index ที่เลือกเข้ามาใน batchเพื่อเอาออกจาก remaining ต่อไป
        remove_list = []

        for idx in remaining:
            w = weights[idx]
            # ตรวจสอบเงื่อนไข capacity: น้ำหนักรวมใน batch + น้ำหนัก item นี้ ต้องไม่เกิน capacity_picker
            if current_batch_weight + w > capacity_picker:
                continue

            # ตรวจสอบเงื่อนไขสินค้าหนัก: ถ้า item เป็นสินค้าหนัก (หรือมีน้ำหนัก >= 40)
            # ให้คำนวณน้ำหนักสินค้าหนักที่อยู่ใน batch แล้วตรวจสอบว่ารวมกันไม่เกิน value_threshold
            if (idx in heavy_item_set) or (w >= 40):
                heavy_weight = sum(weights[i] for i in batch_indices if (i in heavy_item_set) or (weights[i] >= 40))
                if heavy_weight + w > value_threshold:
                    continue

            # หากผ่านเงื่อนไขทั้งหมด ให้นำ item นี้เข้า batch
            batch_indices.append(idx)
            current_batch_weight += w
            remove_list.append(idx)

        # อัปเดต remaining โดยเอา item ที่ถูกจัด batch ออก
        remaining = np.setdiff1d(remaining, np.array(remove_list))

        # กำหนดหมายเลข batch ให้กับ item ที่ถูกเลือก
        for idx in batch_indices:
            batch_assignment[idx] = current_batch

        # สร้างรายชื่อ location สำหรับ batch นี้
        batch_locations = [df_item_pool.at[idx, 'location'] for idx in batch_indices]
        list_batching_item.append(batch_locations)
        list_item_in_batch.append(batch_indices)

        current_batch += 1

    # สร้าง DataFrame ที่บันทึกหมายเลข batch สำหรับแต่ละสินค้า
    df_item_record = df_item_pool.copy()
    df_item_record['batch'] = batch_assignment

    return list_batching_item, df_item_record, list_item_in_batch


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