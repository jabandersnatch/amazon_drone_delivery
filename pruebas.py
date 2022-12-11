# Do this (new version)

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


def bateryIsValid(matrix) -> bool:
    for index in range(len(matrix)):
        if sum(matrix[index]) > battery_range_case_2[index]:
            return False
    return True


def energyDroneIsValid(way, drone):
    init = initial_position_proof_case[drone]
    sum = distances_proof_case[init, way[0]]
    for index in range(len(way) - 1):
        sum += distances_proof_case[way[index], way[index + 1]]
    if sum <= battery_range_case_2[drone]:
        return True
    else:
        return False


def drone_capacity_valid(way, drone) -> bool:
    """
    function that verifies if the path of a drone meets the conditions
    :param way:  list of the path that the drone follows
    :param drone: the index of drone that is making the path
    :return isValid: returns true or false of the path follows the rules
    """
    size = len(way) - 1  # because the last node is a warehouse, and it doesn't count
    if size <= capacity_proof_case[drone]:
        return True
    else:
        return False


def cross_over_line(arr1, arr2, random_val1, random_val2, combination) -> []:
    size_m1 = len(arr1)
    size_m2 = len(arr2)
    if size_m1 % 2 != 0:
        if random_val1 == 0:
            size_m1_middle = ceil(size_m1 / 2)
        else:
            size_m1_middle = floor(size_m1 / 2)
    else:
        size_m1_middle = size_m1 / 2
    size_m1_middle = int(size_m1_middle)
    if size_m2 % 2 != 0:
        if random_val2 == 0:
            size_m2_middle = ceil(size_m2 / 2)
        else:
            size_m2_middle = floor(size_m2 / 2)
    else:
        size_m2_middle = size_m2 / 2
    size_m2_middle = int(size_m2_middle)
    if combination == 0:
        middlearr1 = arr1[size_m1_middle:]
        middlearr2 = arr2[:size_m2_middle]
        newArr = middlearr2 + middlearr1
    else:
        middlearr1 = arr1[:size_m1_middle]
        middlearr2 = arr2[size_m2_middle:]
        newArr = middlearr1 + middlearr2

    return newArr


def values_not_in_list(lista, values):
    actual = []
    for i in range(len(lista) - 1):
        if lista[i] in values or lista[i] in actual:
            return False
        actual.append(lista[i])
    return True


def needed_values(matrix):
    '''
    function in production
    :param matrix:
    :return:
    '''
    listnodesdeli = list(delivery_point_index_proof_case)
    for travel in matrix:
        for value in travel:
            listnodesdeli.remove(value)
    if len(listnodesdeli) != 0:
        for value in listnodesdeli:
            can = 0
            for index in range(len(matrix)):
                if len(matrix[index]) < capacity_proof_case[index]:
                    arrchangable = matrix[index].copy()
                    arrchangable.insert(floor(len(arrchangable) / 2), value)
                    if energyDroneIsValid(arrchangable):
                        return 0
    return 1
def is_all_values(matrix):
    listnodesdeli = list(delivery_point_index_proof_case)
    for travel in matrix:
        for value in range(len(travel)-1):
            listnodesdeli.remove(travel[value])
    if len(listnodesdeli) != 0:
        return False
    else:
        return True


def crossover_Middles(matrix1, matrix2) -> []:
    new_matrix = []
    values = []
    for drone in range(len(matrix1)):
        posiblecomb = []
        arr_combinations = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
        i = 0
        while i < len(arr_combinations):
            j = 0
            while j < len(arr_combinations[0]):
                k = 0
                while k < len(arr_combinations[0][0]):
                    if arr_combinations[i][j][k] != 1:
                        arr_combinations[i][j][k] = 1
                        line = cross_over_line(matrix1[drone], matrix2[drone], i, j, k)
                        if energyDroneIsValid(line, drone) and drone_capacity_valid(line, drone) and values_not_in_list(
                                line, values):
                            posiblecomb.append(line)
                    k += 1
                j += 1
            i += 1
        if len(posiblecomb) != 0:
            selected = randint(0, len(posiblecomb) - 1)
            line = posiblecomb[selected]
            new_matrix.append(line)
            cut = len(line) - 1
            cutted = line[:cut]
            values = values + cutted
        else:
            return []
    if is_all_values(new_matrix):
        return new_matrix
    else:
        return []



def inicial_value():
    random_arr = []
    can = 0
    while not can:
        random_arr = np.array(list(delivery_point_index_proof_case))
        np.random.shuffle(random_arr)
        if size_delivery - n_drones % 2 == 0:
            random_arr = np.reshape(random_arr, (n_drones, size_delivery - n_drones))
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
            while len(random_arr[drone_index]) > capacity_proof_case[drone_index]:
                if drone_index != n_drones - 1:
                    value = random_arr[drone_index].pop()
                    random_arr[drone_index + 1].append(value)
                else:
                    print("the problen can not be resolved")
        random_drone = [random.randint(0, len(warehouse_index_proof_case) - 1) for _ in range(n_drones)]
        for drone in range(0, n_drones):
            value = warehouse_index_proof_case[random_drone[drone]]
            random_arr[drone].append(value)
        if bateryIsValid(random_arr):
            can = 1
    return random_arr


i = 0
j=0
for i in range(10000000):
    i+=1
    matrix1 = inicial_value()
    matrix2 = inicial_value()
    matrixmerged = crossover_Middles(matrix1, matrix2)
    if len(matrixmerged) != 0:
        j+=1
        print(j/i)
