import pandas as pd

import numpy as np
from numpy import ceil, floor
from haversine import haversine, Unit
from random import randint
import random

init_drone_pos = {}

initial_position = 0
'''
Create a data dictionary with 7 nodes, 
2 are a warehouse the other 5 are delivery points
the location is set in the USA
the demand is set in the same order as the nodes
'''
data = {'node_type': ['warehouse', 'delivery_point', 'delivery_point', 'delivery_point', 'delivery_point', 'warehouse',
                      'delivery_point'],
        'latitude': [42.7173, 40.7128, 41.8781, 39.9526, 38.9072, 39.7545, 40.7128],
        'longitude': [-73.9897, -74.0060, -80.6298, -75.1652, -77.0369, -77.5021, -76.1232],
        'demand': [0, 1, 1, 3, 2, 0, 1]}

proof_case = pd.DataFrame(data)

# Create the distances between the nodes

distances_proof_case = {}
for i in proof_case.index:
    for j in proof_case.index:
        if i != j:
            init_drone_pos[i] = (proof_case['latitude'][i], proof_case['longitude'][i])
            init_drone_pos[j] = (proof_case['latitude'][j], proof_case['longitude'][j])
            distances_proof_case[i, j] = haversine(init_drone_pos[i], init_drone_pos[j], unit=Unit.MILES)

        else:
            distances_proof_case[i, j] = 999
# Calculate the mean distance between the nodes

mean_distance_proof_case = np.round(np.mean(list(distances_proof_case.values())), 0)

mean_battery_range = mean_distance_proof_case * 3

n_drones = 2

battery_range_case_2 = [0.5 * mean_battery_range, 1.5 * mean_battery_range]

# Create the demand dictionary
demand_proof_case = {}
for index, row in proof_case.iterrows():
    demand_proof_case[index] = (f'{row["node_type"]}_{index}', np.round(row['demand'], 0))
# Create the capacity dictionary

# Find the mean demand
mean_demand_proof_case = 0
for index, row in proof_case.iterrows():
    if row['node_type'] == 'delivery_point':
        mean_demand_proof_case += row['demand']

mean_demand_proof_case = np.round(mean_demand_proof_case / len(proof_case[proof_case['node_type'] == 'delivery_point']),
                                  0)

mean_capacity_dron = mean_demand_proof_case * 3

capacity_proof_case = [0.5 * mean_capacity_dron, 1.5 * mean_capacity_dron]

warehouse_index_proof_case = proof_case[proof_case['node_type'] == 'warehouse'].index
delivery_point_index_proof_case = proof_case[proof_case['node_type'] == 'delivery_point'].index
size_delivery = len(delivery_point_index_proof_case)
initial_position_proof_case = {0: 0, 1: 5}


class Chromosome:
    def __init__(self, matrix: []):
        self.matrix = matrix
        self.value = self.calc_values()
        self.valid = 0

    def calc_values(self) -> int:
        suma = 0
        for drone_path in self.matrix:
            for i in range(len(drone_path) - 1):
                suma += distances_proof_case[i, i + 1]
        return suma


class GeneticAlgoritm:
    def __init__(self, inicial_popularion, prob):
        self.inicial_population = inicial_popularion
        inicial_values = []
        for i in range(10):
            inicial_values.append(Chromosome(inicial_value()))
        inicial_values.sort(key=lambda chrome: chrome.value)
        self.prob = prob

    def inicial_value(self):
        can = 0
        while not can:
            random_arr = np.array(list(delivery_point_index_proof_case))
            np.random.shuffle(random_arr)
            if size_delivery - n_drones % 2 == 0:
                random_arr = np.reshape(random_arr, (n_drones, size_delivery-n_drones))
                random_arr = random_arr.tolist()
            else:
                final_ind = size_delivery - 1
                value = random_arr[-1]
                random_arr = random_arr[:final_ind]
                hallo = int(final_ind / n_drones)
                random_arr = np.reshape(random_arr, (n_drones, hallo))
                random_index = random.randint(0, n_drones - 1)
                random_arr = random_arr.tolist()
                random_arr[random_index].append(value)
            for drone_index in range(0, n_drones):
                while len(random_arr[drone_index])>capacity_proof_case[drone_index]:
                    if drone_index != n_drones - 1:
                        value = random_arr[drone_index].pop()
                        random_arr[drone_index + 1].append(value)
                    else:
                        print("the problen can not be resolved")
            random_drone=[random.randint(0,len(warehouse_index_proof_case)-1)for _ in range(n_drones)]
            for drone in range(0, n_drones):
                value = warehouse_index_proof_case[random_drone[drone]]
                random_arr[drone].append(value)
            if self.batteryIsValid(random_arr):
                can = 1
        return random_arr

    def bateryIsValid(matrix)->bool:
        for index in range(len(matrix)):
            if sum(matrix[index]) > battery_range_case_2[index]:
                return False
        return True
    def energyDroneIsValid(self, lista, drone):
        init = initial_position_proof_case[drone]
        sum = distances_proof_case[init, lista[0]]
        for index in range(len(lista) - 1):
            sum += distances_proof_case[lista[index], lista[index + 1]]
        if sum <= battery_range_case_2[drone]:
            return True
        else:
            return False

    def mutation(self, matrix):
        '''
        Mutate the matrix
        The mutation is done by changing the position of two elements in the matrix the matrix 
        cant have the same element twice
        '''
        new_matrix = []

        is_valid = False
        while not is_valid:
            for drone in range(matrix):
                random_val1 = randint(0, len(matrix[drone]) - 1)
                random_val2 = randint(0, len(matrix[drone]) - 1)
                while random_val1 == random_val2:
                    random_val2 = randint(0, len(matrix[drone]) - 1)
                matrix[drone][random_val1], matrix[drone][random_val2] = matrix[drone][random_val2], matrix[drone][
                    random_val1]
                battery = self.batteryIsValid(matrix)
                energy = self.energyDroneIsValid(matrix)
                if battery and energy:
                    is_valid = True
        return matrix

    def two_point_crossover(self, matrix1, matrix2):
        '''
        Use the two point crossover to make the new matrix
        '''
        is_valid = False
        new_matrix = []
        while not is_valid:
            new_matrix = []
            for drone in range(matrix1):
                random_val1 = randint(0, len(matrix1[drone]) - 1)
                random_val2 = randint(0, len(matrix1[drone]) - 1)
                while random_val1 == random_val2:
                    random_val2 = randint(0, len(matrix1[drone]) - 1)
                    if random_val1 > random_val2:
                        random_val1, random_val2 = random_val2, random_val1
                    # make the 2 point crossover
                    new_matrix.append(
                        matrix1[drone][:random_val1] + matrix2[drone][random_val1:random_val2] + matrix1[drone][
                                                                                                 random_val2:])
            battery = self.batteryIsValid(new_matrix)
            energy = self.energyDroneIsValid(new_matrix)
            if battery and energy:
                is_valid = True

        return new_matrix

    def crossover_Middles(self, matrix1, matrix2):
        new_matrix = []
        for drone in range(matrix1):
            can = 0
            while not can:
                arr_combinations = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
                random_val1 = randint(0, 1)
                random_val2 = randint(0, 1)
                combination = randint(0, 1)
                if arr_combinations[random_val1][random_val2][combination] != 1:
                    arr_combinations[random_val1][random_val2][combination] = 1
                    line = self.cross_over_line(matrix1[drone], matrix2[drone], random_val1, random_val2, combination)
                else:
                    find = 0
                    i = 0
                    while i < len(arr_combinations) and not find:
                        j = 0
                        while j < len(arr_combinations[0])and not find:
                            k = 0
                            while k < len(arr_combinations[0][0]) and not find:
                                if arr_combinations[i][j][k] != 1:
                                    arr_combinations[i][j][k] = 1
                                    line = self.cross_over_line(matrix1[drone], matrix2[drone], i, j, k)
                                    if self.energyDroneIsValid(line, drone):
                                        can = 1
                                        find = 1
                                        new_matrix.append(line)
                    if find == 0:
                        return []

        return new_matrix

    def cross_over_line(self, arr1, arr2, random_val1, random_val2, combination)->[]:
        size_m1 = len(arr1)
        size_m2 = len(arr2)
        if size_m1 % 2 != 0:
            if random_val1 == 0:
                size_m1_middle = ceil(size_m1 / 2)
            else:
                size_m1_middle = floor(size_m1 / 2)
        else:
            size_m1_middle = int(size_m1 / 2)

        if size_m2 % 2 != 0:
            if random_val2 == 0:
                size_m2_middle = ceil(size_m2 / 2)
            else:
                size_m2_middle = floor(size_m2 / 2)
        else:
            size_m2_middle = int(size_m2 / 2)

        if combination == 0:
            middlearr1 = arr1[size_m1_middle:]
            middlearr2 = arr2[:size_m2_middle]
            newArr = middlearr2 + middlearr1
        else:
            middlearr1 = arr1[:size_m1_middle]
            middlearr2 = arr2[size_m2_middle:]
            newArr = middlearr1 + middlearr2

        return newArr

