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
Create a data dictionary with 4 nodes, 
one node is the warehouse the other two are delivery_point 
the location is set in the USA
'''
data = {'node_type': ['warehouse', 'delivery_point', 'delivery_point'],
        'latitude': [42.7173, 40.7128, 41.8781],
        'longitude': [-73.9897, -74.0060, -87.6298]}

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

n_drones = 1

battery_range_case_2 = 


# Drone warehouse final battery



os.system("clear")
Model = ConcreteModel()

print(initial_position)

# %%
drone_set=range(n_drones)
nodes_index=proof_case.index

# Create the Sets
n_travels = 10
# Create an x variable that is the size of nodesxnodesxn_dronesxn_travels

Model.x = Var(nodes_index, nodes_index, drone_set, range(n_travels), domain=Binary)

# Create an y variable that is the size of nodesxn_dronesxn_travels where the domain is all the rationals that are positive with 0

Model.y = Var(nodes_index, drone_set, range(n_travels), domain=NonNegativeReals)

# Create the objective function

Model.obj = Objective(expr=sum(distances_proof_case[i,j]*Model.x[i, j, k, t] for i in nodes_index for j in nodes_index for k in drone_set for t in range(n_travels)), sense=maximize)

# Restriction 1: The dron cant travel more than the battery range
def battrest(Model, d, v):
    return sum (Model.x[i, j, d , v] * distances_proof_case[i, j] for i in nodes_index for j in nodes_index) <= battery_range_case_2[v-1]

Model.battrest = Constraint(drone_set, range(n_travels), rule=battrest)

# Restriction 2: Delivery points must be supplied 
def delivrest(Model, j):
    return demand_case_2[j][1] == sum(Model.x[i, j, d, v] * capacity_case_2[d] * Model.y[j,d,v] for i in nodes_index for d in drone_set for v in range(n_travels))


Model.delivrest = Constraint(proof_case.index, rule=delivrest)


# Restriction 3: Ensure demand satisfaction
def demandrest(Model, d, v):
    return sum(Model.x[i, j, d, v] * Model.y[j,d,v] for i in nodes_index for j in nodes_index) <= 1

Model.demandrest = Constraint(drone_set, range(n_travels), rule=demandrest)

# Restriction 4: Ensure that the drone outs from the warehouse
def warehouseoutrest(Model, i, d, v):
    if i in warehouse_index_case_2:
        return sum(Model.x[i, j, d, v] for j in nodes_index ) <= 1
    else:
        return Constraint.Skip

Model.warehouseoutrest = Constraint(nodes_index, drone_set, range(n_travels), rule=warehouseoutrest)

# Restriction 5: Ensure that the drone in from the warehouse
def warehouseinrest(Model, j, d, v):
    if j in warehouse_index_case_2:
        return sum(Model.x[i, j, d, v] for i in nodes_index) <= 1
    else:
        return Constraint.Skip

Model.warehouseinrest = Constraint(nodes_index, drone_set, range(n_travels), rule=warehouseinrest)

# Restriction 6: The drones must enter and exit all the delivery points
def deliverypointrest(Model, j, d, v):
    if i not in warehouse_index_case_2:
        return sum(Model.x[i, j, d, v] for i in nodes_index) == sum(Model.x[j, i, d, v] for i in nodes_index if i in proof_case)
    else:
        return Constraint.Skip

Model.deliverypointrest = Constraint(nodes_index, drone_set, range(n_travels), rule=deliverypointrest)

SolverFactory('mindtpy').solve(Model, mip_solver='glpk',nlp_solver='ipopt')

# %%
# Display x and y variables
Model.x.display()

# %%

import matplotlib.pyplot as plt

# Plot the routes of the drones

for d in range(n_drones):
    for v in range(n_travels):
        for i in proof_case.index:
            for j in proof_case.index:
                if Model.x[i, j, d, v]() == 1:
                    plt.plot([proof_case['latitude'][i], proof_case['latitude'][j]], [proof_case['longitude'][i], proof_case['longitude'][j]], 'k-')

# Plot the nodes of the graph with a different color for the warehouses and the delivery points

plt.plot(proof_case['longitude'][warehouse_index_case_2], proof_case['latitude'][warehouse_index_case_2], 'ro')
plt.plot(proof_case['longitude'][delivery_point_index_case_2], proof_case['latitude'][delivery_point_index_case_2], 'bo')

# Create the legend
plt.legend(['Warehouse', 'Delivery Point'])

# Create the axis

plt.xlabel('Longitude')
plt.ylabel('Latitude')

# plt.title('Amazon Delivery Drones Case 2')

plt.show()
