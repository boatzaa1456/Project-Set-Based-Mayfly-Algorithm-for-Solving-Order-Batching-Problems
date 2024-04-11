import timeit
import time
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from SB_SupportFunction import read_input
import numpy as np
from  SBMA_Main import mayfly


start_time = time.time()
num_gen = 100
pop_size = 50
a1 = 1
a2 = 2
a3 = 2
gmax = 0.9
gmin = 0.3
alpha = 0.5
random_seed = 1111

gbest_each_gen = []
male_each_gen = []
female_each_gen = []

name_path_input = '1R-20I-150C-2P'
df_item_pool = read_input(name_path_input)
gbest_each_gen,best_solution = mayfly(name_path_input, num_gen, pop_size, a1, a2, a3, gmax, gmin,
                                alpha, random_seed)
# male_each_gen = chunk_list(male_each_gen, pop_size // 2)
# female_each_gen = chunk_list(female_each_gen, pop_size // 2)
# End the timer
end_time = time.time()
time_taken = end_time - start_time

# Convert time_taken to hours, minutes, and seconds
hours = int(time_taken // 3600)
minutes = int((time_taken % 3600) // 60)
seconds = time_taken % 60
gbest_value = min(gbest_each_gen)
gbest_each_gen = gbest_each_gen[:num_gen]

# Display final results
hours, remainder = divmod(time_taken, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Time Taken: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")
print(f"Time Taken (second) : {time_taken:.2f}")
print("----" * 50)
# #
# # # คำนวณค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน
# # average_male = [np.mean(gen) for gen in male_each_gen]
# # average_female = [np.mean(gen) for gen in female_each_gen]
# # std_deviation_male = [np.std(gen) for gen in male_each_gen]
# # std_deviation_female = [np.std(gen) for gen in female_each_gen]
rounds = np.arange(1, len(gbest_each_gen) + 1)
# #
# # # ใช้ Polynomial regression สำหรับเส้นแนวโน้ม
# # p_male = Polynomial.fit(rounds - 1, average_male, deg=3)
# # p_female = Polynomial.fit(rounds - 1, average_female, deg=3)
# #
# # # สร้างข้อมูลสำหรับเส้นแนวโน้ม
# # x_new = np.linspace(0, len(male_each_gen) - 1, num=len(male_each_gen))
# # y_new_male = p_male(x_new)
# # y_new_female = p_female(x_new)

plt.figure(figsize=(16, 8))

# วาดกราฟข้อมูล
plt.plot(rounds, gbest_each_gen, '-o', color='red', label='GBest Value', markersize=4, linewidth=1.5)
# #
# # # วาดกราฟข้อมูล
# # plt.plot(rounds, average_male, '-^', color='blue', label='Average Male Value', markersize=4, linewidth=1.5)
# # plt.plot(rounds, average_female, '-s', color='green', label='Average Female Value', markersize=4, linewidth=1.5)
# #
# # # วาดเส้นแนวโน้ม
# # plt.plot(x_new + 1, y_new_male, 'b--', label='Male Trendline')
# # plt.plot(x_new + 1, y_new_female, 'g--', label='Female Trendline')
# #
# # #แสดงความเคลื่อนไหวด้วยส่วนเบี่ยงเบนมาตรฐาน
# # plt.fill_between(rounds, np.array(average_male) - np.array(std_deviation_male),
# #                  np.array(average_male) + np.array(std_deviation_male), color='blue', alpha=0.2,
# #                  label='Std.Deviation of Male Value')
# # plt.fill_between(rounds, np.array(average_female) - np.array(std_deviation_female),
# #                  np.array(average_female) + np.array(std_deviation_female), color='green', alpha=0.2,
# #                  label='Std.Deviation of Female Value')
# #
plt.xlabel(f"Time Taken: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")
plt.ylabel('Values')
plt.title(
    f'{name_path_input} - {pop_size} Population Size - {num_gen} Generations - Seed {random_seed} - Tadiness: {gbest_value}')

# Add the legend
plt.legend()
plt.tight_layout()

# Create an inset in the plot for parameter descriptions
param_descriptions = (
    "Parameters:\n"
    f"a1 = {a1}\n"
    f"a2 = {a2}\n"
    f"a3 = {a3}\n"
    f"gmax= {gmax}\n"
    f"gmin= {gmin}\n"
    f"alpha = {alpha}\n"
    # f"sub_size = {sub_size}\n"
    # f"mutation_rate = {mutation_rate}"
)
# Position the text box in figure coords, and set the box style
text_box = plt.text(0.16, 0.04, param_descriptions, transform=plt.gcf().transFigure, fontsize=11,
                    verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9)

# Show the plot with the parameter descriptions
plt.show()
