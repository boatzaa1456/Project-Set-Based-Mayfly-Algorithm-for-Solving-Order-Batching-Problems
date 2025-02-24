import numpy as np
from collections import defaultdict
import bisect
import textwrap as tr


def slot_aisle(aisle, each_aisle_item):
    """Set location ranges for each aisle using list comprehension."""
    return [(i * each_aisle_item + 1, (i + 1) * each_aisle_item) for i in range(aisle)]


def s_shape_routing(item_list, *info):
    """
    Optimized S-shape routing algorithm.
    Parameters:
      - item_list: list of location numbers (integers)
      - info: (aisle, each_aisle_item, rack_x, rack_y, aisle_x, enter_aisle, distance_y)
    Returns:
      - total_distance: Total routing distance computed by S-shape algorithm.
    """
    aisle, each_aisle_item, rack_x, rack_y, aisle_x, enter_aisle, distance_y = info
    cross_aisle = (rack_x * 2) + aisle_x
    slots = slot_aisle(aisle, each_aisle_item)

    # Sort item_list and determine aisle index for each item using comprehension
    sorted_items = sorted(item_list)
    item_aisles = [next(j for j, (low, high) in enumerate(slots) if low <= item <= high)
                   for item in sorted_items]

    # Group items by aisle
    groups = defaultdict(list)
    for item, a in zip(sorted_items, item_aisles):
        groups[a].append(item)

    # Unique aisles (sorted) where items are present
    travel_aisle = sorted(groups.keys())

    # If only one aisle is involved, follow the single-aisle logic
    if len(travel_aisle) == 1:
        a = travel_aisle[0]
        total_distance = distance_y + cross_aisle * abs(a)
        total_distance += enter_aisle + rack_y / 2
        current_pos = slots[a][1]
        for item in groups[a]:
            # คำนวณระยะทางภายใน aisle ตามสูตรเดิม
            step = ((int(abs(current_pos - item) / 2 + 0.5) * rack_y) if item % 2 == 0
                    else (int(abs(current_pos - item) / 2) * rack_y)) + aisle_x
            total_distance += step
            current_pos = item
        total_distance += int(abs(slots[a][1] - current_pos) / 2) * rack_y
        total_distance += rack_y / 2 + enter_aisle
        total_distance += travel_aisle[-1] * cross_aisle + distance_y
        return total_distance
    else:
        # More than one aisle: process first aisle and then intermediate aisles
        total_distance = distance_y
        prev_aisle = travel_aisle[0]
        total_distance += cross_aisle * abs(prev_aisle)
        # Process first aisle group
        a = prev_aisle
        total_distance += enter_aisle + rack_y / 2
        current_pos = slots[a][1]
        for item in groups[a]:
            step = ((int(abs(current_pos - item) / 2 + 0.5) * rack_y) if item % 2 == 0
                    else (int(abs(current_pos - item) / 2) * rack_y)) + aisle_x
            total_distance += step
            current_pos = item
        total_distance += int(abs(slots[a][1] - current_pos) / 2) * rack_y + rack_y / 2 + enter_aisle

        # Process intermediate aisles
        for a in travel_aisle[1:]:
            # Distance to move between aisles
            total_distance += abs(a - prev_aisle) * cross_aisle
            # Estimate in-aisle travel distance as average range within group
            group_items = groups[a]
            if group_items:
                in_aisle_dist = np.mean([abs(slots[a][1] - x) for x in group_items])
            else:
                in_aisle_dist = 0
            total_distance += enter_aisle + rack_y / 2 + in_aisle_dist + aisle_x + rack_y / 2 + enter_aisle
            prev_aisle = a

        # Return from last aisle to depot
        total_distance += travel_aisle[-1] * cross_aisle + distance_y
        return total_distance


def combined_routing(item_list, *info):
    """
    Optimized Combined Routing algorithm.
    Parameters:
      - item_list: list of location numbers
      - info: (aisle, each_aisle_item, rack_x, rack_y, aisle_x, enter_aisle, distance_y)
    Returns:
      - total_distance: Total routing distance computed by Combined Routing algorithm.
    """
    aisle, each_aisle_item, rack_x, rack_y, aisle_x, enter_aisle, distance_y = info
    cross_aisle = (rack_x * 2) + aisle_x
    slots = slot_aisle(aisle, each_aisle_item)

    sorted_items = sorted(item_list)
    item_aisles = [next(j for j, (low, high) in enumerate(slots) if low <= item <= high)
                   for item in sorted_items]

    groups = defaultdict(list)
    for item, a in zip(sorted_items, item_aisles):
        groups[a].append(item)
    travel_aisle = sorted(groups.keys())

    # Combined routing: sum distance from depot to first aisle, process each aisle, and return to depot.
    total_distance = distance_y
    first_aisle = travel_aisle[0]
    total_distance += first_aisle * cross_aisle  # from depot to first aisle
    for a in travel_aisle:
        total_distance += enter_aisle + rack_y / 2
        if groups[a]:
            # Approximate in-aisle distance as range of item positions
            in_aisle_dist = max(groups[a]) - min(groups[a])
            total_distance += in_aisle_dist + aisle_x
        total_distance += rack_y / 2 + enter_aisle
    last_aisle = travel_aisle[-1]
    total_distance += last_aisle * cross_aisle  # from last aisle back to depot
    total_distance += distance_y
    return total_distance


def precedence_constrained_routing(item_list, *info):
    """
    Placeholder for precedence constrained routing.
    สามารถปรับปรุงในลักษณะเดียวกับฟังก์ชัน combined_routing ได้
    """
    # สำหรับตอนนี้ ให้เรียกใช้ combined_routing เป็นเบื้องต้น
    return combined_routing(item_list, *info)


# ฟังก์ชันช่วยเหลือเพิ่มเติม (เช่น print_tr หรือ section_divider) ไม่ได้มีผลต่อ performance จึงคงไว้ตามเดิม
def section_divider(number):
    print('***' + number * '-' + '***')


def print_tr(text, w=100):
    print(tr.fill(text, width=w))
