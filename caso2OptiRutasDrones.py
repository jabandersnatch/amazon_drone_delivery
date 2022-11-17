# %% [markdown]
# ## Implementación de clustering para la selección de sucursales.
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.express as px

# %%
amazon_delivery_drones_case_2 = pd.read_csv('AmazonDroneDelivery.csv', sep='\t', index_col=0)

# %% [markdown]
# # Caso 2 plot global

# %%
fig = px.scatter_mapbox(amazon_delivery_drones_case_2,
                        lat="latitude",
                        lon="longitude",
                        color="NODE_TYPE",
                        title="Amazon Delivery Drones Case 2",
                        zoom=5,

                        )
fig.update_layout(
                    mapbox_center_lat = 4.570868,
                    mapbox_center_lon = -74.297333,
                    mapbox_style="open-street-map",
                )
fig.show()
print('plot complete')

# %% [markdown]
# ## Hallar la matriz de distances entre las sucursales

# %%
# Create a dictionary to store the distances
from haversine import haversine, Unit
distances_case_2 = {}
# Iterate over the rows of the dataframe
for index, row in amazon_delivery_drones_case_2.iterrows():
    for index2, row2 in amazon_delivery_drones_case_2.iterrows():
        # Calculate the distance between the two rows using the haversine formula
        if index != index2:

           distance = haversine((row['latitude'], row['longitude']), (row2['latitude'], row2['longitude']))
        else:
            distance=99

        # Add the distance to the dictionary
        distances_case_2[(index, index2)] = distance


# %%
# Create a csv file with the distances

last_index_case_2 = amazon_delivery_drones_case_2.index.max()

print(last_index_case_2)
with open('distances_case_2.csv', 'w') as f:
    f.write(' \t')
    for index, row in amazon_delivery_drones_case_2.iterrows():
        node = row['NODE_TYPE']
        if  index == last_index_case_2:
            f.write( f'{node}_{index}')
        else:
            f.write( f'{node}_{index}\t')

    f.write('\n')

    for index, row in amazon_delivery_drones_case_2.iterrows():
        node = row['NODE_TYPE']
        f.write(f'{node}_{index}\t')

        for index2, row2 in amazon_delivery_drones_case_2.iterrows():
            if index2 == last_index_case_2:
                f.write(f'{np.round(distances_case_2[(index, index2)],3)}')
            else:
                f.write(f'{np.round(distances_case_2[(index, index2)],3)}\t')
        count_col = 0
        f.write('\n')

# %%

# Create the DEMAND dictionary
demand_case_2 = {}
for index, row in amazon_delivery_drones_case_2.iterrows():
    demand_case_2[index] = (f'{row["NODE_TYPE"]}_{index}',np.round(row['DEMAND'],0))

# Create the DEMAND csv file
with open('demand_case_2.csv', 'w') as f:
    for key in demand_case_2.keys():
        f.write(f'{demand_case_2[key][0]}\t{demand_case_2[key][1]}\n')

# %%
# Get the mean of distances
mean_distance_case_2 = np.mean(list(distances_case_2.values()))
mean_distance_case_2 = np.round(mean_distance_case_2, 0)
# Get the mean of demands
mean_demand_case_2 = 0
for index, row in amazon_delivery_drones_case_2.iterrows():
    if row['NODE_TYPE'] == 'delivery_point':
        mean_demand_case_2 += row['DEMAND']

mean_demand_case_2 = mean_demand_case_2/len(amazon_delivery_drones_case_2[amazon_delivery_drones_case_2['NODE_TYPE'] == 'delivery_point'])


mean_demand_case_2 = np.round(mean_demand_case_2,0)

print(f'Distancia media: {mean_distance_case_2}[km]')
print(f'Demanda media: {mean_demand_case_2}[kg]')

# %%
# Create a random uniform capacity for the drones
n_drones_case_2 = len(amazon_delivery_drones_case_2[amazon_delivery_drones_case_2['NODE_TYPE'] == 'warehouse']) * 2
capacity_case_2 = np.random.uniform(0.5*mean_demand_case_2, 1.5*mean_demand_case_2, n_drones_case_2)
capacity_case_2 = np.round(capacity_case_2,0)
# Create a random uniform battery range for the drones
mean_battery_range_case_2 = 10*mean_distance_case_2
battery_range_case_2 = np.random.uniform(0.5*mean_battery_range_case_2, 1.5*mean_battery_range_case_2, n_drones_case_2)

# %%
# Create the info_drones dictionary
info_drone_dict_case_2 = {}
for i in range(n_drones_case_2):
    info_drone_dict_case_2[i] = (f'drone_{i}', capacity_case_2[i], battery_range_case_2[i])

# Create the info_drones.csv file
with open('info_drones_case_2.csv', 'w') as f:
    for key in info_drone_dict_case_2.keys():
        f.write(f'{info_drone_dict_case_2[key][0]}\t{np.round(info_drone_dict_case_2[key][1],0)}\t{np.round(info_drone_dict_case_2[key][2],0)}\n')

# %% [markdown]
# # One hot encoding NODE_TYPE

# %%
# Make one hot encoding for the NODE_TYPE column
amazon_delivery_drones_case_2 = pd.get_dummies(amazon_delivery_drones_case_2, columns=['NODE_TYPE'])

# Create an array that contains wich index is a warehouse

warehouse_index_case_2 = amazon_delivery_drones_case_2[amazon_delivery_drones_case_2['NODE_TYPE_warehouse'] == 1].index

# Create an array that constains wich index is a delivery point

delivery_point_index_case_2 = amazon_delivery_drones_case_2[amazon_delivery_drones_case_2['NODE_TYPE_delivery_point'] == 1].index

# Create an array that contains wich index is a charging station

charging_station_index_case_2 = amazon_delivery_drones_case_2[amazon_delivery_drones_case_2['NODE_TYPE_charging_station'] == 1].index

# Create an dictionary with size of warehouse x n_drones_case_2 this is a binary table that tells the initial position of the drones
initial_position_case_2 = {}
# randomize the initial position of the drones

for i in range(n_drones_case_2):
    initial_position_case_2[i] = np.random.choice(warehouse_index_case_2)

# Print the initial position of the drones

# %%


"""
Created on Wen Nov 02 09:52:16 2022

@author: Juan Andrés Méndez G. Erich G
"""
from pyomo.environ import *
import numpy as np

import os

os.system("clear")
Model = ConcreteModel()

# %%
drone_set=range(n_drones_case_2)
nodes_index=amazon_delivery_drones_case_2.index

# Create an x variable that is the size of nodesxnodesxn_dronesxn_travels

Model.x = Var(nodes_index, nodes_index, drone_set, domain=Binary)

# Create an y variable that is the size of nodesxn_dronesxn_travels where the domain is all the rationals that are positive with 0

Model.y = Var(nodes_index, drone_set, domain=Binary)

# Create the objective function


Model.obj = Objective(expr=sum(distances_case_2[i,j]*Model.x[i, j, d] for i in nodes_index for j in nodes_index for d in drone_set), sense=minimize)

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
    if j not in warehouse_index_case_2:
        return sum(Model.x[i, j, d] for i in nodes_index) == sum(Model.x[j, k, d] for k in nodes_index)
    else:
        return Constraint.Skip
Model.allThatComesInMustGoOut = Constraint(nodes_index, drone_set, rule=allThatComesInMustGoOut)

'''
The droneOut constraint is the constraint that the drone must leave the warehouse
'''
def droneOut(Model, d):
    return sum(Model.x[initial_position_case_2[d], i, d] for i in nodes_index)==1
Model.droneOut = Constraint(drone_set, rule=droneOut)

'''
fullfillDemand: The drone must fullfill the demand
'''
def fullfillDemand(Model, d):
    return sum(Model.x[i,j,d]*Model.y[j,d]*demand_case_2[j][1] for j in nodes_index for i in nodes_index) <= capacity_case_2[d]
Model.fullfillDemand = Constraint(drone_set, rule=fullfillDemand)

'''
All the delivery points must be visited
'''
def allDeliveryPointsMustBeVisited(Model, i):
    if i in delivery_point_index_case_2:
        return sum(Model.x[i,j,d] for j in nodes_index for d in drone_set)==1
    else:
        return Constraint.Skip

Model.allDeliveryPointsMustBeVisited = Constraint(nodes_index, rule=allDeliveryPointsMustBeVisited)

'''
All the delivery points must be exited
'''
def allDeliveryPointsMustBeExited(Model, j):
    if j in delivery_point_index_case_2:
        return sum(Model.x[i,j,d] for i in nodes_index for d in drone_set)==1
    else:
        return Constraint.Skip
    
Model.allDeliveryPointsMustBeExited = Constraint(nodes_index, rule=allDeliveryPointsMustBeExited)

# Solve the model with quadratic constraints

SolverFactory('ipopt').solve(Model)


# Print the results
Model.display()

import matplotlib.pyplot as plt

# Plot the routes of the drones in the case 2 use a different color for each drone

for d in drone_set:
    for i in nodes_index:
        for j in nodes_index:
            if Model.x[i,j,d]()> 0:
                plt.plot([amazon_delivery_drones_case_2['latitude'][i], amazon_delivery_drones_case_2['latitude'][j]], [amazon_delivery_drones_case_2['longitude'][i], amazon_delivery_drones_case_2['longitude'][j]], color='C'+str(d))

# Plot the nodes of the graph with a different color for the warehouses and the delivery points

plt.plot(amazon_delivery_drones_case_2['latitude'][warehouse_index_case_2], amazon_delivery_drones_case_2['longitude'][warehouse_index_case_2], 'ro', label = 'warehouse')
plt.plot(amazon_delivery_drones_case_2['latitude'][delivery_point_index_case_2], amazon_delivery_drones_case_2['longitude'][delivery_point_index_case_2], 'bo', label = 'delivery_point')

plt.legend()

# Create the axis

plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.title('Amazon Delivery Drones Case 2 of Concept')

plt.show()
