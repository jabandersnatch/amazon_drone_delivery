"""
Created on Wen Nov 02 09:52:16 2022

@author: Juan Andrés Méndez G. Erich G
"""
import pandas as pd
from pyomo.environ import *
import matplotlib.pyplot as plt
import numpy as np
import random
from haversine import haversine, Unit

import os


## Variables creation Drone ware hosue  initial position
init_drone_pos = {}

initial_position = 0

data = {'node_type': ['warehouse', 'delivery_point', 'delivery_point', 'delivery_point', 'delivery_point', 'warehouse','delivery_point', 'delivery_point', 'delivery_point', 'delivery_point', 'warehouse','delivery_point', 'delivery_point', 'delivery_point', 'delivery_point', 'warehouse','delivery_point', 'delivery_point', 'delivery_point', 'delivery_point'],
        'latitude': [42.7173, 40.7128, 41.8781, 39.9526, 41.9072, 45.7545, 40.7128, 41.8781, 42.9526, 38.9072, 39.7545, 40.7128, 41.8781, 39.9526, 38.9072, 44.7545, 40.7128, 41.8781, 38.9526, 38.9072],
        'longitude': [-73.9897, -74.0060, -80.6298, -75.1652, -79.0369, -78.5021, -78.1232, -77.0369, -77.5021, -77.1232, -77.5021, -76.1232, -73.0369, -76.5021, -79.1232, -82.5021, -72.1232, -78.0369, -77.5021, -75.1232],
        'demand': [0, 1, 1, 3, 2, 0, 1, 3, 1, 2, 0, 1, 3, 1, 2, 0, 1, 3, 1, 2]}

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

n_drones = 5

rand_percentages = [0.4, 0.6, 0.8, 1.0]
battery_range_case_2 = []
for i in range(n_drones):
    battery_range_case_2.append(mean_battery_range * random.choice(rand_percentages))


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

mean_capacity_dron = mean_demand_proof_case * 4

capacity_proof_case = {}

for i in range(n_drones):
    capacity_proof_case[i] = mean_capacity_dron * random.choice(rand_percentages)


warehouse_index_proof_case = proof_case[proof_case['node_type'] == 'warehouse'].index
delivery_point_index_proof_case = proof_case[proof_case['node_type'] == 'delivery_point'].index
size_delivery = len(delivery_point_index_proof_case)

initial_position_proof_case = {}
# Create the initial position dictionary for the drones
for i in range(n_drones):
    initial_position_proof_case[i] = random.choice(warehouse_index_proof_case)



# Make warehouse index a list
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
For each delivery point the y variable must be 1
'''
def yDeliveryPoints(Model, j):
    return sum(Model.y[j,d] for d in drone_set) == 1
Model.yDeliveryPoints = Constraint(delivery_point_index_proof_case, rule=yDeliveryPoints)

'''
All delivery points must be visited by a drone
'''
def visitDeliveryPoints(Model, j):
    return sum(Model.x[i,j,d] for i in nodes_index for d in drone_set) == 1
Model.visitDeliveryPoints = Constraint(delivery_point_index_proof_case, rule=visitDeliveryPoints)


'''
For each drone they must visit the same number of delivery points as they exit
'''
def visitExitDeliveryPoints(Model, d, j):
    return sum(Model.x[i,j,d] for i in nodes_index) == sum(Model.x[j,i,d] for i in nodes_index)
Model.visitExitDeliveryPoints = Constraint(drone_set, delivery_point_index_proof_case, rule=visitExitDeliveryPoints)

'''
The battery range constraint
'''
def battery(Model, d):
    return sum(Model.x[i,j,d] * distances_proof_case[i,j] for i in nodes_index for j in nodes_index )<=battery_range_case_2[d]
Model.batery = Constraint(drone_set, rule=battery)

'''
Delete sub-tours
'''
def subtour_elimination(Model, i, j):
    if i not in warehouse_index_proof_case and j not in warehouse_index_proof_case:
        return sum(Model.x[i,j,d] for d in drone_set) + sum(Model.x[j,i,d] for d in drone_set) <= 1
    else:
        return Constraint.Skip
Model.subtour_elimination = Constraint(nodes_index, nodes_index, rule=subtour_elimination)


'''
A node can't be visited by himself
'''
def noSelfVisit(Model, i):
    return sum(Model.x[i,i,d] for d in drone_set) == 0
Model.noSelfVisit = Constraint(nodes_index, rule=noSelfVisit)

# Solve the model with quadratic constraints using couenne
SolverFactory('couenne').solve(Model, tee=True)



# Print the results
Model.display()

# Plot the routes of the drones in the proof case use a different color for each drone

for d in drone_set:
    for i in nodes_index:
        for j in nodes_index:
            if np.round(Model.x[i,j,d]()) == 1:
                plt.plot([proof_case['latitude'][i], proof_case['latitude'][j]], [proof_case['longitude'][i], proof_case['longitude'][j]], color= 'C'+str(d))
                # Plot the index of the node
                plt.text(proof_case['latitude'][i], proof_case['longitude'][i], str(i))

# Plot the nodes of the graph with a different color for the warehouses and the delivery points

plt.plot(proof_case['latitude'][warehouse_index_proof_case], proof_case['longitude'][warehouse_index_proof_case], 'ro', label = 'warehouse')
plt.plot(proof_case['latitude'][delivery_point_index_proof_case], proof_case['longitude'][delivery_point_index_proof_case], 'bo', label = 'delivery_point')


# Create the axis

plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.title('Amazon Delivery Drones Proof of Concept')

plt.show()
