"""
Created on Wen Nov 02 09:52:16 2022

@author: Juan Andrés Méndez G. Erich G
"""
import pandas as pd
from pyomo.environ import *
import matplotlib.pyplot as plt
import numpy as np
from haversine import haversine, Unit

import os


## Variables creation

# Drone ware hosue  initial position
init_drone_pos = {}

initial_position = 0

'''
Create a data dictionary with 7 nodes, 
2 are a warehouse the other 5 are delivery points
the location is set in the USA
the demand is set in the same order as the nodes
'''
data = {'node_type': ['warehouse', 'delivery_point', 'delivery_point', 'delivery_point', 'delivery_point', 'warehouse', 'delivery_point'],
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

mean_battery_range = mean_distance_proof_case * 4

n_drones = 2

battery_range_case_2 = np.round(np.random.uniform(0.5 * mean_battery_range, 1.5 * mean_battery_range, n_drones))

# Create the demand dictionary
demand_proof_case = {}
for index, row in proof_case.iterrows():
    demand_proof_case[index] = (f'{row["node_type"]}_{index}',np.round(row['demand'],0))
# Create the capacity dictionary

# Find the mean demand  
mean_demand_proof_case = 0
for index, row in proof_case.iterrows():
    if row['node_type'] == 'delivery_point':
        mean_demand_proof_case += row['demand']

mean_demand_proof_case = np.round(mean_demand_proof_case / len(proof_case[proof_case['node_type']== 'delivery_point']), 0)

mean_capacity_dron = mean_demand_proof_case * 2

capacity_proof_case = np.round(np.random.uniform(0.5 * mean_capacity_dron, 1.5 * mean_capacity_dron, n_drones))

warehouse_index_proof_case = proof_case[proof_case['node_type'] == 'warehouse'].index
delivery_point_index_proof_case = proof_case[proof_case['node_type'] == 'delivery_point'].index


initial_position_proof_case = {}

for i in range(n_drones):
    initial_position_proof_case[i] = np.random.choice(warehouse_index_proof_case)
# Drone warehouse final battery



os.system("clear")
Model = ConcreteModel()

# %%
drone_set=range(n_drones)
nodes_index=proof_case.index

# Create an x variable that is the size of nodesxnodesxn_dronesxn_travels

Model.x = Var(nodes_index, nodes_index, drone_set, domain=Binary)

# Create an y variable that is the size of nodesxn_dronesxn_travels where the domain is all the rationals that are positive with 0

Model.y = Var(nodes_index, drone_set, domain=Binary)

# Create the objective function

Model.obj = Objective(expr=sum(distances_proof_case[i,j]*Model.x[i, j, d] for i in nodes_index for j in nodes_index for d in drone_set), sense=minimize)

# Create the constraints

'''
Warehouse out: The drone must leave the warehouse
'''
def warehouseOut(Model, i, d):
    return sum(Model.x[i, j, d] for j in nodes_index)<=1
Model.warehouseOut = Constraint(nodes_index, drone_set, rule=warehouseOut)

'''
Drone in: The drone must enter the warehouse
'''
def droneIn(Model, j, d):
    return sum(Model.x[i, j, d] for i in nodes_index)<=1
Model.droneIn = Constraint(nodes_index, drone_set, rule=droneIn)

'''
All that comes in must go out: The drones must leave all the nodes that it goes in 
'''
def allThatComesInMustGoOut(Model, j, d):
    if j not in warehouse_index_proof_case:
        return sum(Model.x[i, j, d] for i in nodes_index) == sum(Model.x[j, k, d] for k in nodes_index)
    else:
        return Constraint.Skip
Model.allThatComesInMustGoOut = Constraint(nodes_index, drone_set, rule=allThatComesInMustGoOut)

'''
The droneOut constraint is the constraint that the drone must leave the warehouse
'''
def droneOut(Model, d):
    return sum(Model.x[initial_position_proof_case[d], i, d] for i in nodes_index)==1
Model.droneOut = Constraint(drone_set, rule=droneOut)

'''
fullfillDemand: The drone must fullfill the demand
'''
def fullfillDemand(Model, d):
    return sum(Model.x[i,j,d]*Model.y[j,d]*demand_proof_case[j][1] for j in nodes_index for i in nodes_index) <= capacity_proof_case[d]
Model.fullfillDemand = Constraint(drone_set, rule=fullfillDemand)

'''
All the delivery points must be visited
'''
def allDeliveryPointsMustBeVisited(Model, i):
    if i in delivery_point_index_proof_case:
        return sum(Model.x[i,j,d] for j in nodes_index for d in drone_set)==1
    else:
        return Constraint.Skip

Model.allDeliveryPointsMustBeVisited = Constraint(nodes_index, rule=allDeliveryPointsMustBeVisited)

'''
All the delivery points must be exited
'''
def allDeliveryPointsMustBeExited(Model, j):
    if j in delivery_point_index_proof_case:
        return sum(Model.x[i,j,d] for i in nodes_index for d in drone_set)==1
    else:
        return Constraint.Skip
    
Model.allDeliveryPointsMustBeExited = Constraint(nodes_index, rule=allDeliveryPointsMustBeExited)

# Solve the model with quadratic constraints

SolverFactory('ipopt').solve(Model)


# Print the results
Model.display()

import matplotlib.pyplot as plt

# Plot the routes of the drones in the proof case use a different color for each drone

for d in drone_set:
    for i in nodes_index:
        for j in nodes_index:
            if Model.x[i,j,d](),0 > 0:
                plt.plot([proof_case['latitude'][i], proof_case['latitude'][j]], [proof_case['longitude'][i], proof_case['longitude'][j]], color='C'+str(d))

# Plot the nodes of the graph with a different color for the warehouses and the delivery points

plt.plot(proof_case['latitude'][warehouse_index_proof_case], proof_case['longitude'][warehouse_index_proof_case], 'ro', label = 'warehouse')
plt.plot(proof_case['latitude'][delivery_point_index_proof_case], proof_case['longitude'][delivery_point_index_proof_case], 'bo', label = 'delivery_point')


# Create the axis

plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.title('Amazon Delivery Drones Proof of Concept')

plt.show()
