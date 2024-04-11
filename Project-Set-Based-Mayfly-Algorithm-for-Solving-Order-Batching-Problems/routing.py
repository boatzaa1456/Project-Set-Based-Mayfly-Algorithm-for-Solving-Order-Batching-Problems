
from collections import Counter
import textwrap as tr
#
def section_divider(number):
    print('***' + number * '-' + '***')

def print_tr(text,w=100):
    lines = text
    print(tr.fill(lines,width=w))


def slot_aisle(aisle, each_aisle_item):
    ''' Set location number for each aisle with the number of aisles and  the number of location in an aisle'''
    slot = []
    for i in range(aisle):
        min = i * each_aisle_item + 1
        max = (i + 1) * each_aisle_item
        slot.append((min, max))
    return slot



def s_shape_routing(item_list,  *info):


    #print("\n S_shape Algorithm \n")
    aisle, each_aisle_item, rack_x, rack_y, aisle_x, enter_aisle, distance_y = info
    #print(f'item_list_test = {item_list}')
    cross_aisle = (rack_x * 2) + aisle_x
    slot = slot_aisle(aisle, each_aisle_item)  # Set location in aisle
    #print(f'slot = {slot}')
    # record which aisle number the item is in using a list item_aisle
    # item_aisle[x] = y means item in xth order are in aisle y
    # item_aisle=[1,2,2,5] means the first item in item_list is in aisle 1, second 2 , third 2 ...

    sorted_item_list = item_list
    item_aisle = []
    for i in range(len(sorted_item_list)):
        for j in range(aisle):
            if sorted_item_list[i] >= slot[j][0] and sorted_item_list[i] <= slot[j][1]:
                item_aisle.append(j)

    item_each_aisle_sorted = []
    for i in range(len(sorted_item_list)):
        item_each_aisle_sorted.append([])

    a = 0
    for i in range(0, len(sorted_item_list)):
        if item_aisle[i-1] == item_aisle[i]:
            item_each_aisle_sorted[a-1].append(sorted_item_list[i])
        else:
            item_each_aisle_sorted[a].append(sorted_item_list[i])
            a += 1

    item_aisle_sorted = []
    for i in range(0, len(item_aisle)):
        if i == 0:
            item_aisle_sorted.append(item_aisle[i])
        elif item_aisle[i-1] == item_aisle[i]:
            print(f'')
        else:
            item_aisle_sorted.append(item_aisle[i])

    item_each_aisle = []
    for i in range(0, len(item_each_aisle_sorted)):
        if item_each_aisle_sorted[i] != []:
            item_each_aisle.append(item_each_aisle_sorted[i])

    total_distance = 0
    total_distance += distance_y
    down = 1
    travel_aisle = item_aisle_sorted
    # Calculate the distance travelled by the picker from the first aisle to the one-before last aisle.
    # If there is only one aisle, we traverse from down and go back to the depot and return the distance.
    if len(travel_aisle) == 1:
        cur_aisle = travel_aisle[0]
        total_distance += cross_aisle * abs(cur_aisle)
        #print(f'In aisle {cur_aisle} and down {down}, from the front cross aisle of aisle 0 '
              #f'to asile {cur_aisle} at the front cross aisle: {total_distance}')
        total_distance += enter_aisle
        total_distance += rack_y / 2
        cur_pos = slot[cur_aisle][1]
        #print(f'In aisle {cur_aisle} and down {down}, from the front cross aisle '
              #f'to the starting bottom location {cur_pos}: {total_distance}')

        for j in item_each_aisle[0]:
            total_distance += ((int(abs(cur_pos - j) / 2 + 0.5) * rack_y) if j % 2 == 0 else (int(abs(
                cur_pos - j) / 2) * rack_y)) + aisle_x
            #print(f'In aisle {cur_aisle} and down {down}, from location {cur_pos} to location {j}: {total_distance}')
            cur_pos = j
        total_distance += int(abs(slot[cur_aisle][1] - cur_pos) / 2) * rack_y
        #print(
            #f'In aisle {cur_aisle} and down {down}, from location {cur_pos} '
            #f'to the bottom location {slot[cur_aisle][1]}: {total_distance}')
        cur_pos = slot[cur_aisle][1]
        total_distance += rack_y / 2
        total_distance += enter_aisle
        #(f'In aisle {cur_aisle} and down {down}, from the bottom location {cur_pos}'
              #f'to the front cross aisle: {total_distance}')
        down = 1
        total_distance += travel_aisle[-1] * cross_aisle
        #print(f'In aisle 0, from the last aisle {travel_aisle[-1]} to aisle 0: {total_distance}')
        total_distance += distance_y
        #print(f'The total distance travelled by S-shape: {total_distance}')
        return total_distance

    else:
        # if there are  more than one aisle
        travel_aisle_no_last = travel_aisle[:-1]
        cur_aisle = 0#travel_aisle[0]
        item = 0
        for i in travel_aisle_no_last:

            total_distance += cross_aisle * abs(i - cur_aisle)
            #print(f'cross_aisle  = {cross_aisle * abs(i - cur_aisle)}')
            #print(f'In aisle {i}, from aisle {cur_aisle} to aisle {i}: {total_distance}')

            cur_aisle = i
            if down == 1:
                total_distance += enter_aisle
                total_distance += rack_y / 2
                cur_pos = slot[i][1]
                #print(f'In aisle {i} and down {down}, from the front cross aisle '
                          #f'to the starting bottom location {cur_pos}: {total_distance}')
                for j in item_each_aisle[item]:
                    total_distance += ((int(abs(cur_pos - j) / 2 + 0.5) * rack_y) if j % 2 == 0 else (int(abs(
                            cur_pos - j) / 2) * rack_y)) + aisle_x
                    #print(f'In aisle {i} and down {down}, from location {cur_pos} to location {j}: {total_distance}')
                    cur_pos = j

                total_distance += int(abs(slot[i][0] - cur_pos) / 2) * rack_y
               # print(
                        #f'In aisle {i} and down {down}, from location {cur_pos} '
                        #f'to the top location {slot[i][0]}: {total_distance}')
                cur_pos = slot[i][0]
                total_distance += rack_y / 2
                total_distance += enter_aisle
                #print(f'In aisle {cur_aisle} and down {down}, from the top location {cur_pos} '
                          #f'to the back cross aisle: {total_distance}')
                down = 0
                item += 1

            else:

                total_distance += enter_aisle
                total_distance += rack_y / 2
                cur_pos = slot[i][0]
                #print(f'In aisle {i} and down {down}, from the back cross aisle '
                          #f'to the starting top location {cur_pos}: {total_distance}')
                for j in item_each_aisle[item]:
                    total_distance += ((int(abs(cur_pos - j) / 2) * rack_y) if j % 2 == 0 else (int(abs(
                            cur_pos - j) / 2 + 0.5) * rack_y)) + aisle_x
                    #print(f'In aisle {i} and down {down}, from location {cur_pos} to location {j}: {total_distance}')
                    cur_pos = j

                total_distance += int(abs(slot[i][1] - cur_pos) / 2) * rack_y
                #print(
                        #f'In aisle {i} and down {down}, from location {cur_pos} to the bottom location {slot[i][1]}: {total_distance}')
                cur_pos = slot[i][1]
                total_distance += rack_y / 2
                total_distance += enter_aisle
                #print(f'In aisle {cur_aisle} and down {down}, from the bottom location {cur_pos} '
                          #f'to the front cross aisle: {total_distance}')
                down = 1
                item += 1

            #section_divider(100)
            #total_distance += travel_aisle[-1] * cross_aisle
            #print(f'In aisle 0, from the last aisle {travel_aisle[-1]} to aisle 0: {total_distance}')
            '''In aisle 0 to Depot'''
            #total_distance += distance_y
            #print(f'The total distance travelled by S-shape: {total_distance}')
            # Compute the distance in the last aisle. This also deals with the situation where there is only
            # one aisle with items because the above for loop would not be activated.

        total_distance += cross_aisle * abs(travel_aisle[-1] - cur_aisle)
        #print(f'cross_aisle  = {cross_aisle * abs(travel_aisle[-1] - cur_aisle)}')
        #print(f'In the last aisle {travel_aisle[-1]}, from aisle {cur_aisle} '
                  #f'to the last aisle {travel_aisle[-1]}: {total_distance}')
        cur_aisle = travel_aisle[-1]
        total_distance += enter_aisle
        total_distance += rack_y / 2

        if down == 1:
            cur_pos = slot[travel_aisle[-1]][1]
            #print(
                    #f'In the last aisle {travel_aisle[-1]} and down {down}, from the front cross aisle '
                    #f'to the starting bottom location {cur_pos}: {total_distance}')
            for j in item_each_aisle[item]:
                total_distance += ((int(abs(cur_pos - j) / 2 + 0.5) * rack_y) if j % 2 == 0 else (int(abs(
                        cur_pos - j) / 2) * rack_y)) + aisle_x
                #print(
                        #f'In the last aisle {travel_aisle[-1]} and down {down}, from location {cur_pos} to location {j}: {total_distance}')
                cur_pos = j

            total_distance += int(abs(slot[travel_aisle[-1]][1] - cur_pos) / 2) * rack_y
            #print(
                    #f'In the last aisle {travel_aisle[-1]} and down {down}, '
                    #f'from location{cur_pos} to the bottom location {slot[travel_aisle[-1]][1]}: {total_distance}')
            cur_pos = slot[travel_aisle[-1]][1]

        else:
            cur_pos = slot[travel_aisle[-1]][0]
            #print(
                    #f'In the last aisle {travel_aisle[-1]} and down {down}, from the back cross aisle '
                    #f'to the starting top location {cur_pos}: {total_distance}')

            for i in item_each_aisle[item]:
                total_distance += ((int(abs(cur_pos - i) / 2) * rack_y) if i % 2 == 0 else (int(abs(
                        cur_pos - i) / 2 + 0.5) * rack_y)) + aisle_x
               #print(
                        #f'In the last aisle {travel_aisle[-1]} and down {down}, from location {cur_pos} to location {i}: {total_distance}')
                cur_pos = i
            total_distance += int(abs(slot[travel_aisle[-1]][1] - cur_pos) / 2) * rack_y

            #print(
                    #f'In the last aisle {travel_aisle[-1]} and down {down}, '
                    #f'from location {cur_pos} to the bottom location {slot[travel_aisle[-1]][1]}: {total_distance}')
            cur_pos = slot[travel_aisle[-1]][1]

        total_distance += rack_y / 2
        total_distance += enter_aisle
        #print(f'In the last aisle {travel_aisle[-1]}, to the front cross aisle location: {total_distance}')
        down = 1
        total_distance += travel_aisle[-1] * cross_aisle
        #print(f'In aisle 0, from the last aisle {travel_aisle[-1]} to aisle 0: {total_distance}')
        total_distance += distance_y
        #print(f'The total distance travelled by S-shape: {total_distance}')
        return total_distance

def combined_routing(item_list ,*info):

    text_width = 60
    #print("\n\n")
    #section_divider(100)
    #print("\n Combined Algorithm \n")

    aisle, each_aisle_item,rack_x, rack_y, aisle_x, enter_aisle, distance_y = info
    cross_aisle = (rack_x * 2) + aisle_x
    slot = slot_aisle(aisle, each_aisle_item)  # Set location in aisle

    # sort item list
    sorted_item_list = item_list

    # record which aisle number the item is in using a list item_aisle
    # item_aisle[x] = y means item in xth order are in aisle y
    # item_aisle=[1,2,2,5] means the first item in item_list is in aisle 1, second 2 , third 2 ...
    item_aisle = []

    for i in range(len(sorted_item_list)):
        for j in range(aisle):
            if sorted_item_list[i] >= slot[j][0] and sorted_item_list[i] <= slot[j][1]:
                item_aisle.append(j)

    #  A two dimensional list item_each_aisle_sorted collect items that need to be collected in each aisle
    # item_each_aisle_sorted=[[2,5,9],[21,25,30],[41,49][ if travel_aisle=[1,2,5], it means that in aisle 1, item 2,5 and 9
    # need to be picked out and for aisle 2 and 5, items 21,25, 30 and items 41,49 are to be collected by the picker.
    item_each_aisle_sorted = []
    for i in range(len(sorted_item_list)):
        item_each_aisle_sorted.append([])

    a = 0
    for i in range(0, len(sorted_item_list)):
        if item_aisle[i - 1] == item_aisle[i]:
            item_each_aisle_sorted[a - 1].append(sorted_item_list[i])
        else:
            item_each_aisle_sorted[a].append(sorted_item_list[i])
            a += 1

    item_aisle_sorted = []
    for i in range(0, len(item_aisle)):
        if i == 0:
            item_aisle_sorted.append(item_aisle[i])
        elif item_aisle[i - 1] == item_aisle[i]:
            pass
        else:
            item_aisle_sorted.append(item_aisle[i])

    item_each_aisle = []
    for i in range(0, len(item_each_aisle_sorted)):
        if item_each_aisle_sorted[i] != []:
            item_each_aisle.append(item_each_aisle_sorted[i])

    #section_divider(100)

    total_distance = 0
    total_distance += distance_y
    total_distance += item_aisle_sorted[0] * cross_aisle
    cur_aisle = item_aisle_sorted[0]
    #print(f'From the depot to the front cross aisle of aisle {cur_aisle} : {total_distance}')
    # First aisle with items
    #item_each_aisle_from_down = sorted(item_each_aisle_sorted[cur_aisle], reverse=True)
    b = 'b' + str(item_aisle_sorted[0])
    a = 'a' + str(item_aisle_sorted[0])
    La_cur = [b]
    #La_cur.extend(item_each_aisle_from_down)
    La_cur.append(a)
    #print(f'Ta in the first aisle {cur_aisle} is {La_cur}')
    Lb_cur = La_cur[:-1]
    Lb_cur.append(b)
    #print(f'Tb in the first aisle {cur_aisle} is {Lb_cur}')
    cLb_cur = total_distance
    # Calculate cta_cur
    cLa_cur = total_distance
    cLa_cur += enter_aisle
    cLa_cur += rack_y / 2
    cur_pos = slot[item_aisle_sorted[0]][1]
    #print(f'La *** In aisle {cur_aisle}, from start to  the bottom location {cur_pos} : {cLa_cur} ')
    item = 0
    for i in item_each_aisle[item]:
        cLa_cur += ((int(abs(cur_pos - i) / 2 + 0.5) * rack_y) if i % 2 == 0 else (int(abs(
            cur_pos - i) / 2) * rack_y)) + aisle_x
        #print(f'La *** In aisle {cur_aisle}, from location {cur_pos} to location {i}: {cLa_cur}')
        cur_pos = i
    cLa_cur += int(abs(item_each_aisle[item][0] - cur_pos) / 2) * rack_y

    cur_pos = slot[item_aisle_sorted[item]][0]
    cLa_cur += int(abs(item_each_aisle[item][0] - cur_pos) / 2) * rack_y
    #print(f'La *** In aisle {cur_aisle}, from location {i} '
          #f'to the top location {cur_pos} : {cLa_cur}')
    cLa_cur += rack_y / 2
    cLa_cur += enter_aisle
    #print(f'La *** In aisle {cur_aisle}, from the top location to the back cross aisle: {cLa_cur}')
    #print(f'La *** In aisle {cur_aisle}, the distance for La_cur : {cLa_cur}')
    #section_divider(100)

    # Calculate ctb_cur
    # cLb_cur = total_distance
    cLb_cur += enter_aisle
    cLb_cur += rack_y / 2
    cur_pos = slot[item_aisle_sorted[0]][1]
    #print(f'Lb *** In aisle {cur_aisle}, from start to the bottom location {cur_pos}: {cLb_cur} ')
    for i in item_each_aisle[item]:
        cLb_cur += ((int(abs(cur_pos - i) / 2 + 0.5) * rack_y) if i % 2 == 0 else (int(abs(
            cur_pos - i) / 2) * rack_y)) + aisle_x
        #print(f'Lb *** In aisle {cur_aisle} and from down, from location {cur_pos} to location {i}: {cLb_cur}')
        cur_pos = i
    cLb_cur += int(abs(slot[item_aisle_sorted[0]][1]- cur_pos) / 2) * rack_y
   # print(
       # f'Lb *** In aisle {cur_aisle} and from down, from location {cur_pos} '
       # f'to the bottom location {slot[item_aisle_sorted[0]][1]} : {cLb_cur}')
    cur_pos = slot[item_aisle_sorted[0]][1]
    cLb_cur += rack_y / 2
    cLb_cur += enter_aisle
    #print(f'Lb *** In aisle {cur_aisle} , for the bottom location to the front cross aisle : {cLb_cur}')
    #print(f'Lb *** In aisle {cur_aisle} the distance for Lb_cur : {cLb_cur}')

    if len(item_aisle_sorted) == 1:
        total_distance = cLb_cur
        total_distance += item_aisle_sorted[0]*cross_aisle #Fixed on 26 Jul
       # print(f'Since there is only one aisle, from the front cross aisle of {cur_aisle} to aisle 0 : {total_distance}')
        total_distance += distance_y
        #print(f'From the front cross aisle of aisle 0 to the depot : {total_distance}')
        picker_route = Lb_cur
        picker_route.append('depot')
        #lines = f"The picker's route is {Lb_cur} with the total distance of {total_distance}"
        #print(tr.fill(lines, width=60))
        return total_distance

    #section_divider(100)
    item += 1
    # Calculate the distance for the in-between aisle
    for i in item_aisle_sorted[1:-1]:
        cur_aisle_prv = cur_aisle
        cur_aisle = i
        distance_between_aisle = abs(cur_aisle - cur_aisle_prv) * cross_aisle
        La_prv = La_cur[:]
        Lb_prv = Lb_cur[:]
        cLa_prv = cLa_cur
        cLb_prv = cLb_cur

        a = 'a' + str(i)
        b = 'b' + str(i)
        # Find La_cur and cLa_cur
        La1_cur = La_prv[:]
        La1_cur.append(a)
        La2_cur = Lb_prv[:]
        La2_cur.append(b)

        #print(f'Aisle {cur_aisle}')
        #print ('***La_cur***')
        # print(f'La1 (from the back cross aisle)*** In aisle {cur_aisle} is {La1_cur}')
        #print_tr(f'La1 (from the back cross aisle)*** In aisle {cur_aisle} is {La1_cur}',text_width)

        cLa1_cur = cLa_prv
        cLa1_cur += distance_between_aisle
        #print(f'La1*** In aisle {cur_aisle} at the back cross aisle : {cLa1_cur} ')
        cLa1_cur += enter_aisle
        cLa1_cur += rack_y / 2
        cur_pos = slot[i][0]
        #print_tr(f'La1*** In aisle {cur_aisle} from start the top location {cur_pos} : {cLa1_cur}')
        for j in item_each_aisle_sorted[item]:
            cLa1_cur += ((int(abs(cur_pos - j) / 2) * rack_y) if j % 2 == 0 else (int(abs(
                cur_pos - j) / 2 + 0.5) * rack_y)) + aisle_x
            #print(f'La1*** In aisle {cur_aisle} from location {cur_pos} to location {j} : {cLa1_cur}')
            cur_pos = j
        cLa1_cur += int(abs(cur_pos - slot[i][0]) / 2) * rack_y
        #print(f'La1*** In aisle {cur_aisle} for location {cur_pos} '
              #f'to the top location {slot[i][0]} :{cLa1_cur}')
        cur_pos = slot[i][0]
        cLa1_cur += rack_y / 2
        cLa1_cur += enter_aisle
        #print(f'La1*** In aisle {cur_aisle} at the back cross-aisle : {cLa1_cur}')
        #print(f'La1*** The total distance for La1_cur : {cLa1_cur}')

        #print_tr(f'La2 (from the front cross aisle)*** In aisle {cur_aisle} is {La2_cur}',text_width)
        cLa2_cur = cLb_prv
        cLa2_cur += distance_between_aisle
        #print(f'La2*** In aisle {cur_aisle} at the front cross aisle : {cLa2_cur} ')
        cLa2_cur += enter_aisle
        cLa2_cur += rack_y / 2
        cur_pos = slot[i][1]
        #print_tr(f'La2*** In aisle {cur_aisle} from start to the bottom location {cur_pos}: {cLa2_cur}')
        for j in item_each_aisle[item]:
            cLa2_cur += ((int(abs(cur_pos - j) / 2 + 0.5) * rack_y) if j % 2 == 0 else (int(abs(
                cur_pos - j) / 2) * rack_y)) + aisle_x
            #print(f'La2*** In aisle {cur_aisle} from location {cur_pos} to location {j}: {cLa2_cur}')
            cur_pos = j
        cLa2_cur += int(abs(slot[i][0] - cur_pos) / 2) * rack_y
        #print(f'La2*** In aisle {cur_aisle} for location {cur_pos} '
              #f'to the top location {slot[i][0]}:{cLa2_cur}')
        cur_pos = slot[i][0]
        cLa2_cur += rack_y / 2
        cLa2_cur += enter_aisle
        #print(f'La2*** In aisle {cur_aisle} at the back cross-aisle : {cLa2_cur}')
        #print(f'La2*** The total distance for La2_cur : {cLa2_cur}')
        cLa_cur, La_cur = (cLa1_cur, La1_cur) if cLa1_cur < cLa2_cur else (cLa2_cur, La2_cur)
        # if cLa1_cur < cLa2_cur:
        #     print (f'La_cur = La1_cur since cLa1_cur < cLa2_cur with distance {cLa_cur}' )
        # else:
        #     print (f'La_cur = La2_cur since cLa1_cur => cLa2_cur with distance {cLa_cur}' )

        # Find Lb_cur and cLb_cur
        Lb1_cur = Lb_prv[:]
        Lb1_cur.append(b)
        Lb2_cur = La_prv[:]
        Lb2_cur.append(a)


        #print ('***Lb_cur***')
        # print(f'Lb1 (from the front cross aisle)*** In aisle {cur_aisle} is {Lb1_cur}')
        #print_tr(f'Lb1 (from the front cross aisle)*** In aisle {cur_aisle} is {Lb1_cur}',text_width)
        cLb1_cur = cLb_prv
        cLb1_cur += distance_between_aisle
        #print(f'Lb1*** In aisle {cur_aisle} at the front cross aisle : {cLb1_cur} ')
        cLb1_cur += enter_aisle
        cLb1_cur += rack_y / 2
        cur_pos = slot[i][1]
        #print_tr(f'Lb1*** In aisle {cur_aisle} from start the bottom location {cur_pos} : {cLb1_cur}')
        for j in item_each_aisle[item]:
            cLb1_cur += ((int(abs(cur_pos - j) / 2 + 0.5) * rack_y) if j % 2 == 0 else (int(abs(
                cur_pos - j) / 2) * rack_y)) + aisle_x
            #print(f'Lb1*** In aisle {cur_aisle} from location {cur_pos} to location {j} : {cLb1_cur}')
            cur_pos = j
        cLb1_cur += int(abs(slot[i][1] - cur_pos) / 2) * rack_y
        #print(f'Lb1*** In aisle {cur_aisle} for location{cur_pos} '
              #f'to the last location{slot[i][1]}:{cLb1_cur}')
        cur_pos = slot[i][1]
        cLb1_cur += rack_y / 2
        cLb1_cur += enter_aisle
        #print(f'Lb1*** In aisle {cur_aisle} at the front cross-aisle : {cLb1_cur}')
        #print(f'Lb1*** The total distance for Lb1_cur : {cLb1_cur}')

        # print(f'Lb2 (from the back cross aisle)*** In aisle {cur_aisle} is {Lb2_cur}')
        #print_tr(f'Lb2 (from the back cross aisle)*** In aisle {cur_aisle} is {Lb2_cur}',text_width)
        cLb2_cur = cLa_prv
        cLb2_cur += distance_between_aisle
        #print(f'Lb2*** In aisle {cur_aisle} at the back cross aisle : {cLb2_cur} ')
        cLb2_cur += enter_aisle
        cLb2_cur += rack_y / 2
        cur_pos = slot[i][0]
        #print_tr(f'Lb2*** In aisle {cur_aisle} from start the top location {cur_pos} : {cLb2_cur}')

        for j in item_each_aisle[item]:
            cLb2_cur += ((int(abs(cur_pos - j) / 2) * rack_y) if j % 2 == 0 else (int(abs(
                cur_pos - j) / 2 + 0.5) * rack_y)) + aisle_x
            #print(f'Lb2*** In aisle {cur_aisle} from location {cur_pos} to location {j} : {cLb2_cur}')
            cur_pos = j
        cLb2_cur += int(abs(slot[cur_aisle][1] - cur_pos) / 2) * rack_y
        #print(f'Lb2*** In aisle {cur_aisle} for location{cur_pos} '
              #f'to the last location {slot[cur_aisle][1]} : {cLb2_cur}')
        cur_pos = slot[i][1]
        cLb2_cur += rack_y/2
        cLb2_cur += enter_aisle
        #print(f'Lb2*** In aisle {cur_aisle} at the front cross-aisle : {cLb2_cur}')
        #print(f'Lb2*** The total distance for Lb2_cur : {cLb2_cur}')

        cLb_cur,Lb_cur = (cLb1_cur,Lb1_cur) if cLb1_cur < cLb2_cur else (cLb2_cur,Lb2_cur)
        # if cLb1_cur < cLb2_cur:
        #     print (f'Lb_cur = Lb1_cur since cLb1_cur < cLb2_cur with distance {cLb_cur}' )
        # else:
        #     print (f'Lb_cur = Lb2_cur since cLb1_cur => cLb2_cur with distance {cLb_cur}' )

        #section_divider(100)
        item +=1

    #Calculate the total distance at the last line
    item -= 1
    cur_aisle_prv = cur_aisle
    cur_aisle = item_aisle_sorted[-1]
    #item_each_aisle_from_down =sorted(item_each_aisle_sorted[cur_aisle],reverse = True)
    distance_between_aisle = int(cur_aisle-cur_aisle_prv)*cross_aisle
    La_prv = La_cur[:]
    Lb_prv = Lb_cur[:]
    cLa_prv = cLa_cur
    cLb_prv = cLb_cur

    a = 'a'+str(cur_aisle)
    b='b'+str(cur_aisle)

    # Find Lb_cur and cLb_cur since
    Lb1_cur=Lb_prv[:]
    Lb1_cur.append(b)
    Lb2_cur = La_prv[:]
    Lb2_cur.append(a)


    #print(f'At the last Aisle {cur_aisle}')
    #print('***Lb_cur***')
    # print(f'Lb1 (from the front cross aisle)*** In aisle {cur_aisle} is {Lb1_cur}')
    #print_tr(f'Lb1 (from the front cross aisle)*** In aisle {cur_aisle} is {Lb1_cur}',text_width)

    cLb1_cur = cLb_prv
    cLb1_cur += distance_between_aisle
    #print(f'Lb1*** In aisle {cur_aisle} at the front cross aisle : {cLb1_cur} ')
    cLb1_cur += enter_aisle
    cLb1_cur += rack_y/2
    cur_pos = slot[cur_aisle][1]
    #print(f'Lb1*** In aisle {cur_aisle} at the bottom location {cur_pos} : {cLb1_cur} ')
    for j in item_each_aisle[item]:
        cLb1_cur += ((int(abs(cur_pos - j) / 2 + 0.5) * rack_y) if j % 2 == 0 else (int(abs(
            cur_pos - j) / 2) * rack_y)) + aisle_x
        #print(f'Lb1*** In aisle {cur_aisle} from location {cur_pos} to location {j} : {cLb1_cur}')
        cur_pos = j
    cLb1_cur +=(int(abs(slot[cur_aisle][1]-cur_pos)/2))*rack_y
    #print(f'Lb1*** In aisle {cur_aisle} for location {cur_pos} '
          #f'to the last location {slot[cur_aisle][1]} : {cLb1_cur}')
    cur_pos = slot[cur_aisle][1]
    cLb1_cur += rack_y/2
    cLb1_cur += enter_aisle
    #print(f'Lb1*** In aisle {cur_aisle} at the front cross-aisle : {cLb1_cur}')
    #print(f'Lb1*** The total distance for Lb1_cur : {cLb1_cur}')

    # print(f'Lb2 (from the back cross aisle)*** In aisle {cur_aisle} is {Lb1_cur}')
    #print_tr(f'Lb2 (from the back cross aisle)*** In aisle {cur_aisle} is {Lb2_cur}',text_width)
    cLb2_cur = cLa_prv
    cLb2_cur += distance_between_aisle
    #print(f'Lb2*** In aisle {cur_aisle} at the back cross aisle : {cLb2_cur} ')
    cLb2_cur += enter_aisle
    cLb2_cur += rack_y/2
    cur_pos = slot[cur_aisle][0]
    #print(f'Lb2*** In aisle {cur_aisle} from the back cross aisle '
          #f'to the top location {cur_pos} : {cLb2_cur} ')
    for j in item_each_aisle[item]:
        cLb2_cur += ((int(abs(cur_pos - j) / 2) * rack_y) if j % 2 == 0 else (int(abs(
            cur_pos - j) / 2 + 0.5) * rack_y)) + aisle_x
        #print(f'Lb2*** In aisle {cur_aisle} from location {cur_pos} to location {j} : {cLb2_cur}')
        cur_pos = j
    cLb2_cur += int(abs(slot[cur_aisle][1]-cur_pos)/2)*rack_y
    #print(f'Lb2*** In aisle {cur_aisle} for location {cur_pos} '
          #f'to the last location {slot[cur_aisle][1]} : {cLb2_cur}')
    cur_pos = slot[cur_aisle][1]
    cLb2_cur += rack_y/2
    cLb2_cur += enter_aisle
    #print(f'Lb2*** In aisle {cur_aisle} at the front cross-aisle : {cLb2_cur}')
    #print(f'Lb2*** The total distance for Lb2_cur : {cLb2_cur}')

    cLb_cur,Lb_cur = (cLb1_cur,Lb1_cur) if cLb1_cur < cLb2_cur else(cLb2_cur,Lb2_cur)
    # if cLb1_cur < cLb2_cur:
    #     print(f'Lb_cur = Lb1_cur since cLb1_cur < cLb2_cur with distance {cLb_cur}')
    # else:
    #     print(f'Lb_cur = Lb2_cur since cLb1_cur => cLb2_cur with distance {cLb_cur}')
    #section_divider(100)
    total_distance = cLb_cur

    total_distance += int(cur_aisle)*cross_aisle
    #print(f'In aisle 0, from the front cross aisle of aisle {cur_aisle} to aisle 0 : {total_distance}')
    total_distance += distance_y
    #print(f'From the front cross aisle of aisle 0 to the depot : {total_distance}')
    #print(f'The total distance travelled by Combined Algortihm : {total_distance}')
    Lb_cur.append('depot')
    picker_route = Lb_cur
    #lines= f"The picker's route is {Lb_cur} with the total distance of {total_distance}"
    #print(tr.fill(lines,width=60))
    #print(f"The picker's route is {Lb_cur} with the total distance of"
    #      f"  {total_distance}")

    return total_distance

def precedence_constrained_routing(item_list ,*info):

    aisle, each_aisle_item, rack_x, rack_y, aisle_x, enter_aisle, distance_y = info
    cross_aisle = (rack_x * 2) + aisle_x
    slot = slot_aisle(aisle, each_aisle_item)  # Set location in aisle

    # sort item list
    sorted_item_list = item_list  # sorted(item_list)
    item_aisle = []

    for i in range(len(sorted_item_list)):
        for j in range(aisle):
            if sorted_item_list[i] >= slot[j][0] and sorted_item_list[i] <= slot[j][1]:
                item_aisle.append(j)

    item_each_aisle_sorted = []
    for i in range(len(sorted_item_list)):
        item_each_aisle_sorted.append([])

    a = 0
    for i in range(0, len(sorted_item_list)):
        if item_aisle[i - 1] == item_aisle[i]:
            item_each_aisle_sorted[a - 1].append(sorted_item_list[i])
        else:
            item_each_aisle_sorted[a].append(sorted_item_list[i])
            a += 1

    travel_aisle = []
    for i in range(0, len(item_aisle)):
        if i == 0:
            travel_aisle.append(item_aisle[i])
        elif item_aisle[i - 1] == item_aisle[i]:
            pass
        else:
            travel_aisle.append(item_aisle[i])

    item_each_aisle_test = []
    for i in range(0, len(item_each_aisle_sorted)):
        if item_each_aisle_sorted[i] != []:
            item_each_aisle_test.append(item_each_aisle_sorted[i])

    total_distance = 0
    total_distance += distance_y
    total_distance += travel_aisle[0] * cross_aisle
    cur_aisle = travel_aisle[0]

    list_routing = []
    La_cur, Lb_cur = [], []
    a = 'a' + str(travel_aisle[0])
    b = 'b' + str(travel_aisle[0])
    list_routing.append(b)
    cLb_cur = total_distance
    cLb_cur += enter_aisle
    cLb_cur += rack_y / 2
    cur_pos = slot[travel_aisle[0]][1]

    item = 0
    for i in item_each_aisle_test[item]:
        cLb_cur += ((int(abs(cur_pos - i) / 2 + 0.5) * rack_y) if i % 2 == 0 else (int(abs(
            cur_pos - i) / 2) * rack_y)) + aisle_x
        cur_pos = i
    cLa_cur = cLb_cur

    '''กลับ Deport'''
    if len(item_each_aisle_test) == 1:
        # Calculate Lb_cur
        cLb_cur += int(abs(slot[travel_aisle[0]][1] - cur_pos) / 2) * rack_y
        cLb_cur += rack_y / 2
        cLb_cur += enter_aisle

        total_distance = cLb_cur
        total_distance += travel_aisle[0] * cross_aisle  # Fixed on 26 Jul
        total_distance += distance_y

        picker_route = Lb_cur
        picker_route.append('depot')
        return total_distance

    item += 1
    for i in travel_aisle[1:]:

        cur_aisle_prv = cur_aisle
        cur_aisle = i
        cLb_cur += int(abs(slot[cur_aisle_prv][1] - cur_pos) / 2) * rack_y
        cLb_cur += rack_y / 2
        cLb_cur += enter_aisle

        '''ช่องทางเดินถัดไป '''
        cLa_cur += int(abs(slot[cur_aisle_prv][0] - cur_pos) / 2) * rack_y
        cLa_cur += enter_aisle
        cLa_cur += rack_y / 2
        a = 'a' + str(i)
        b = 'b' + str(i)

        #print(f'travel_aisle = {i}, {item_each_aisle_test[item]}')
        distance_between_aisle = abs(cur_aisle - cur_aisle_prv) * cross_aisle
        La_prv = La_cur[:]
        Lb_prv = Lb_cur[:]
        cLa_prv = cLa_cur
        cLb_prv = cLb_cur
        cLb_cur = cLb_prv
        cLb_cur += distance_between_aisle
        cLb_cur += enter_aisle
        cLb_cur += rack_y / 2
        cur_pos = slot[i][1]

        for j in item_each_aisle_test[item]:
            cLb_cur += ((int(abs(cur_pos - j) / 2 + 0.5) * rack_y) if j % 2 == 0 else (int(abs(
                cur_pos - j) / 2) * rack_y)) + aisle_x

            cur_pos = j
        cLa_cur = cLa_prv
        cLa_cur += distance_between_aisle
        cLa_cur += enter_aisle
        cLa_cur += rack_y / 2
        cur_pos = slot[i][0]
        for j in item_each_aisle_test[item]:
            cLa_cur += ((int(abs(cur_pos - j) / 2) * rack_y) if j % 2 == 0 else (int(abs(
                cur_pos - j) / 2 + 0.5) * rack_y)) + aisle_x
            cur_pos = j

        if cLa_cur < cLb_cur:
            list_routing.append(a)
            cLa_cur = cLa_cur
            cLb_cur = cLa_cur

        else:
            list_routing.append(b)
            cLa_cur = cLb_cur
            cLb_cur = cLb_cur

        item += 1

    # Calculate the total distance at the last line
    cur_aisle_prv = cur_pos
    cur_pos = slot[cur_aisle][1]
    total_distance = cLb_cur
    total_distance += int(abs(cur_aisle_prv - cur_pos) / 2) * rack_y

    total_distance += enter_aisle
    total_distance += rack_y / 2

    distance = int(cur_aisle) * cross_aisle
    total_distance += int(cur_aisle) * cross_aisle
    total_distance += distance_y

    list_routing.append('depot')
    picker_route = list_routing

    return total_distance


#if __name__ == '__main__':
    #section_divider(100)

    # Set parameters
    #rack_x = 1
    #rack_y = 1
    #aisle_x = 1
    #enter_aisle = 1
    #distance_y = 2

    #block = 1

    #aisle = 10
    #depot = 3
    #each_aisle_item = 90
    #total_item = each_aisle_item * aisle * block

    # item_location = aisle * each_aisle_item
    # order = 15
    # capacity_picker = 60
    # num_item_min = 1
    # num_item_max = item_location

    #item_list = [776, 638, 380]

    #item_list = [74, 744, 341]

    #distance_batch_1 = s_shape_routing(item_list, aisle, each_aisle_item, rack_x, rack_y, aisle_x, enter_aisle, distance_y)
    #distance_batch_2 = combined_routing(item_list, aisle, each_aisle_item, rack_x, rack_y, aisle_x, enter_aisle, distance_y)
    #distance_batch_3 = precedence_constrained_routing(item_list, aisle, each_aisle_item, rack_x, rack_y, aisle_x, enter_aisle, distance_y)

    #print(f' distance s-shape = {distance_batch_1}')
    #print(f' distance combined = {distance_batch_2}')
    #print(f' distance precedence constrained = {distance_batch_3}')