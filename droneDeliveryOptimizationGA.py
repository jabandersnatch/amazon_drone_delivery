'''
This script is a genetic algorithm that optimizes the drone delivery
problem.

Created on 2022-12-09
@author: 
    Juan Andres Méndez G.
    Erich Giusseppe Soto Para
'''

# Import libraries
import numpy as np
import pandas as pd

from haversine import haversine, Unit
from random import randint
import random


## Create the parameters of the problem

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

case = pd.DataFrame(data)

# Create the distances between the nodes

distances = {}
for i in case.index:
    for j in case.index:
        if i != j:
            init_drone_pos[i] = (case['latitude'][i], case['longitude'][i])
            init_drone_pos[j] = (case['latitude'][j], case['longitude'][j])
            distances[i, j] = haversine(init_drone_pos[i], init_drone_pos[j], unit=Unit.MILES)

        else:
            distances[i, j] = 999
# Calculate the mean distance between the nodes

# Transform distances to a numpy vector
distances = np.array(list(distances.values()))

# Calculate the mean distance between the nodes

mean_distance = np.round(np.mean(distances), 0)


mean_battery_range = mean_distance * 3

n_drones = 2

battery_range= np.array([mean_battery_range*0.7, mean_battery_range*1.5])

# Create the demand dictionary
demand = {}
for index, row in case.iterrows():
    demand[index] = (f'{row["node_type"]}_{index}', np.round(row['demand'], 0))

# Transform the demand dictionary to a numpy vector
demand = np.array(list(demand.values()))

# Create the capacity dictionary

# Find the mean demand
mean_demand = 0
for index, row in case.iterrows():
    if row['node_type'] == 'delivery_point':
        mean_demand += row['demand']

mean_demand = np.round(mean_demand / len(case[case['node_type'] == 'delivery_point']),0)

mean_capacity_dron = mean_demand * 3

capacity_case = np.array([0.5 * mean_capacity_dron, 1.5 * mean_capacity_dron])

warehouse_index_case = case[case['node_type'] == 'warehouse'].index
delivery_point_index_case = case[case['node_type'] == 'delivery_point'].index
size_delivery = len(delivery_point_index_case)
initial_position_case = {0: 0, 1: 5}

# Create the Genetic Algorithm


def fitness(member):
    return 1

def crossover(a,b):
    return a

def create_new_member():
    # Create an n x n x d matrix that represents the connections between the nodes and the drones
    n = len(case)
    d = n_drones
    member = np.zeros((n, n, d))

    return member

    
def validate_member(member):
    return True

def check_if_battery_constraint_is_met(member):
    if member * distances <= battery_range

    
def create_next_generation(population):
    return population

def main(number_of_iterations):
    return True
