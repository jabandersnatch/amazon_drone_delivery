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
            inicial_values.append(Chromosome(self.inicial_value()))
        inicial_values.sort(key=lambda chrome: chrome.value)
        self.prob = prob

    def inicial_value(self):
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
            if self.batteryIsValid(random_arr):
                can = 1
        return random_arr

    def bateryIsValid(matrix) -> bool:
        for index in range(len(matrix)):
            if sum(matrix[index]) > battery_range_case_2[index]:
                return False
        return True

    def energyDroneIsValid(self, way, drone):
        init = initial_position_proof_case[drone]
        sum = distances_proof_case[init, way[0]]
        for index in range(len(way) - 1):
            sum += distances_proof_case[way[index], way[index + 1]]
        if sum <= battery_range_case_2[drone]:
            return True
        else:
            return False

    def drone_capacity_valid(self, way, drone) -> bool:
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
                        while j < len(arr_combinations[0]) and not find:
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

    def cross_over_line(self, arr1, arr2, random_val1, random_val2, combination) -> []:
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

    def comb(self, a, b):
        '''
        This function will combine the two arrays such a
        number will never be repeated.
        a and b have the same length
        '''
        def check_if_repeated(array, element):
            '''
            This function will check if an element is repeated in an array
            '''
            if element in array:
                return True
            else:
                return False

        # First remove the last element of all the arrays and store them in a list
        last_a = [i[-1] for i in a]
        last_b = [i[-1] for i in b]
        # Now remove the last element from all the arrays
        a = [i[:-1] for i in a]
        b = [i[:-1] for i in b]
        print ('End warehouses: ', last_a, last_b)
        # Then we combine last_a and last_b  as a matrix where the rows are the
        # elements of last_a and the columns are the elements of last_b
        comb = np.array([last_a, last_b])
        print ('Combination matrix: ', comb)
        rows_visited = []
        nodes_visited = []

        # we will create a new matrix with empty arrays the same size as the number of rows
        # in the a and b arrays
        new_matrix = [np.array([]) for i in range(len(a))]


        # We will now iterate over the rows of a and b
        # We will start by selecting a random row 
        while len(rows_visited) < len(a):
            # We select a random row
            row = np.random.randint(0, len(a))
            # If the row has already been visited, we select another one
            if row in rows_visited:
                continue
            # We add the row to the list of visited rows
            rows_visited.append(row)
            # We print the current row
            print ('Drone {}' .format(row))
            # Now we merge the two arrays such as
            # merge = [a1, b1, a2, b2, ...]
            # note that a[row] and b[row] have different length
            merge = []
            # First get the length of the shortest array
            min_len = min(len(a[row]), len(b[row]))

            # Add length of the rows and get the mean of the two and floor it

            # This will be the number of elements we will add to merge
            # from a[row] and b[row]
            mean = int(np.floor((len(a[row]) + len(b[row])) / 2))

            # Now we iterate over the shortest array
            for i in range(min_len):
                # Make a choice between a[row] and b[row]
                choice = np.random.randint(0, 2)
                # If choice is 0, we add an element from a[row]
                if choice == 0:
                    if check_if_repeated(merge, a[row][i]):
                        #Check if the element is in nodes_visited
                        if check_if_repeated(nodes_visited, a[row][i]):
                            continue
                        else:
                            merge.append(b[row][i])
                    else:
                        # Check if the element is in nodes_visited
                        if check_if_repeated(nodes_visited, a[row][i]):
                            continue
                        else:
                            merge.append(a[row][i])
                # If choice is 1, we add an element from b[row]
                else:
                    if check_if_repeated(merge, b[row][i]):
                        # Check if the element is in nodes_visited
                        if check_if_repeated(nodes_visited, b[row][i]):
                            continue
                        else:
                            merge.append(a[row][i])
                    else:
                        # Check if the element is in nodes_visited
                        if check_if_repeated(nodes_visited, b[row][i]):
                            continue
                        else:
                            merge.append(b[row][i])

            # Now we add the remaining elements of the longest array
            # to merge ensuring that the elements are not repeated
            if len(a[row]) > len(b[row]):
                for i in range(len(a[row]) - min_len):
                    if check_if_repeated(merge, a[row][i + min_len]):
                        # Check if the element is in nodes_visited
                        if check_if_repeated(nodes_visited, a[row][i + min_len]):
                            continue
                        else:
                            merge.append(a[row][i + min_len])
                    else:
                        # Check if the element is in nodes_visited
                        if check_if_repeated(nodes_visited, a[row][i + min_len]):
                            continue
                        else:
                            merge.append(a[row][i + min_len])

            # Now we make a choice between cuting the last element of
            # merge or the first element of merge in order to make sure
            # that the size of merge is equal to mean
            choice = np.random.randint(0, 2)
            elements_to_remove = len(merge) - mean
            if choice == 0:
                merge = merge[:-elements_to_remove]
            else:
                merge = merge[elements_to_remove:]

            # Now we print the merged array
            print ('Merge: ', merge)
            # Now we add the last element of merge to the list of visited nodes
            for i in merge:
                nodes_visited.append(i)
            # Now we add to the merge row a random choice between the last_a and last_b
            choice = np.random.randint(0, 2)
            merge.append(comb[choice, row])
            # Now we print the final merge
            print ('Final merge: ', merge)

            # Now we add the merge to the new new_matrix in order to return it
            new_matrix[row] = np.array(merge)
            
        # print nodes_visited
        print ('Nodes visited: ', nodes_visited)

        # Now we check that all the nodes in a and b have been visited
            
        # First we get the list of all the nodes in a and b
        all_nodes = []
        for i in a:
            all_nodes.extend(i)
        for i in b:
            all_nodes.extend(i)

        # Now we delete the repeated nodes
        all_nodes = list(set(all_nodes))

        # Now we check if all the nodes have been visited

        if set(all_nodes) == set(nodes_visited):
            print ('All nodes have been visited')

        # Now we make a list of the nodes that have not been visited
        nodes_not_visited = list(set(all_nodes) - set(nodes_visited))
        print ('Nodes not visited: ', nodes_not_visited)

        # Now we add the nodes that have not been visited to the new matrix in a random order
        # First we shuffle the nodes_not_visited
        np.random.shuffle(nodes_not_visited)

        # Now we add the nodes to the new new_matrix
        for i in range(len(nodes_not_visited)):

            # We select a random row
            row = np.random.randint(0, len(a))
            # First we check that size of the row is not one
            if len(new_matrix[row]) == 1:
                # If it is one we add the not visitted nodes at the start of the row
                new_matrix[row] = np.insert(new_matrix[row], 0, nodes_not_visited[i])
            
            else:
                # then we add the node to the row at a random position in the row
                # all the postions are valid except the last one
                pos = np.random.randint(0, len(new_matrix[row]) - 1)
                new_matrix[row] = np.insert(new_matrix[row], pos, nodes_not_visited[i])

        # turn the new_matrix into a list of lists
        new_matrix = new_matrix.tolist()

        # Now we return the new_matrix
        return new_matrix
