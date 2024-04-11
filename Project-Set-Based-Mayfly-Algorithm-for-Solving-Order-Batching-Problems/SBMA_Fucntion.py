import random

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

def gravity_calculation(gmax, gmin, gen, num_gen):
    gravity = gmax - ((gmax - gmin) * gen / num_gen)
    return gravity


def alpha_calculation( gen, num_gen):
    alpha = (gen / num_gen)
    return alpha