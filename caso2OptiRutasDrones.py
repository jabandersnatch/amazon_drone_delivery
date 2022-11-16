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
amazon_delivery_drones_case_2 = pd.read_csv('AmazonDroneDeliveryCase2.csv', sep='\t', index_col=0)

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
            distance=999

        # Add the distance to the dictionary
        distances_case_2[(index, index2)] = distance


# %%
# Create a csv file with the distances
import csv

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
n_drones_case_2 = len(amazon_delivery_drones_case_2[amazon_delivery_drones_case_2['NODE_TYPE'] == 'warehouse']) * 3
capacity_case_2 = np.random.uniform(0.5*mean_demand_case_2, 1.5*mean_demand_case_2, n_drones_case_2)
capacity_case_2 = np.round(capacity_case_2,0)
# Create a random uniform battery range for the drones
mean_battery_range_case_2 = 4*mean_distance_case_2
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

# %%

"""
Created on Wen Nov 02 09:52:16 2022

@author: Juan Andrés Méndez G. Erich G
"""
from pyomo.environ import *
import matplotlib.pyplot as plt
import numpy as np

import os

os.system("clear")
Model = ConcreteModel()

for i in demand_case_2.keys():
    print(demand_case_2[i][0], demand_case_2[i][1])

# %%

# Create the Sets
n_travels = 10
# Create an x variable that is the size of nodesxnodesxn_dronesxn_travels

Model.x = Var(amazon_delivery_drones_case_2.index, amazon_delivery_drones_case_2.index, range(n_drones_case_2), range(n_travels), domain=Binary)

# Create an y variable that is the size of nodesxn_dronesxn_travels where the domain is all the rationals that are positive with 0

Model.y = Var(amazon_delivery_drones_case_2.index, range(n_drones_case_2), range(n_travels), domain=NonNegativeReals)

# Create the objective function

Model.obj = Objective(expr=sum(distances_case_2[i,j]*Model.x[i, j, k, t] for i in amazon_delivery_drones_case_2.index for j in amazon_delivery_drones_case_2.index for k in range(n_drones_case_2) for t in range(n_travels)), sense=minimize)

# Restriction 1: The dron cant travel more than the battery range
def battrest(Model, d, v):
    return sum (Model.x[i, j, d , v] * distances_case_2[i, j] for i in amazon_delivery_drones_case_2.index for j in amazon_delivery_drones_case_2.index) <= battery_range_case_2[v-1]

Model.battrest = Constraint(range(n_drones_case_2), range(n_travels), rule=battrest)

# Restriction 2: Delivery points must be supplied 
def delivrest(Model, j):
    return demand_case_2[j][1] == sum(Model.x[i, j, d, v] * capacity_case_2[d] * Model.y[j,d,v] for i in amazon_delivery_drones_case_2.index for d in range(n_drones_case_2) for v in range(n_travels))

Model.delivrest = Constraint(delivery_point_index_case_2, rule=delivrest)

# Restriction 3: Ensure demand satisfaction
def demandrest(Model, d, v):
    return sum(Model.x[i, j, d, v] * Model.y[j,d,v] for i in amazon_delivery_drones_case_2.index for j in amazon_delivery_drones_case_2.index) <= 1

Model.demandrest = Constraint(range(n_drones_case_2), range(n_travels), rule=demandrest)

# Restriction 4: Ensure that the drone outs from the warehouse
def warehouseoutrest(Model, j, d, v):
    return sum(Model.x[i, j, d, v] for i in amazon_delivery_drones_case_2.index if i in warehouse_index_case_2) <= 1

Model.warehouseoutrest = Constraint(amazon_delivery_drones_case_2.index, range(n_drones_case_2), range(n_travels), rule=warehouseoutrest)

# Restriction 5: Ensure that the drone in from the warehouse
def warehouseinrest(Model, j, d, v):
    return sum(Model.x[i, j, d, v] for i in amazon_delivery_drones_case_2.index if i in warehouse_index_case_2) <= 1

Model.warehouseinrest = Constraint(amazon_delivery_drones_case_2.index, range(n_drones_case_2), range(n_travels), rule=warehouseinrest)

# Restriction 6: The drones must enter and exit all the delivery points
def deliverypointrest(Model, j, d, v):
    return sum(Model.x[i, j, d, v] for i in amazon_delivery_drones_case_2.index if i in delivery_point_index_case_2) == sum(Model.x[j, i, d, v] for i in amazon_delivery_drones_case_2.index if i in delivery_point_index_case_2)

Model.deliverypointrest = Constraint(amazon_delivery_drones_case_2.index, range(n_drones_case_2), range(n_travels), rule=deliverypointrest)

SolverFactory('mindtpy').solve(Model, mip_solver='glpk',nlp_solver='ipopt')

# %%
# Display x and y variables
Model.x.display()

# %%

import matplotlib.pyplot as plt

# Plot the routes of the drones

for d in range(n_drones_case_2):
    for v in range(n_travels):
        for i in amazon_delivery_drones_case_2.index:
            for j in amazon_delivery_drones_case_2.index:
                if Model.x[i, j, d, v]() == 1:
                    plt.plot([amazon_delivery_drones_case_2['longitude'][i], amazon_delivery_drones_case_2['longitude'][j]], [amazon_delivery_drones_case_2['latitude'][i], amazon_delivery_drones_case_2['latitude'][j]], 'k-')

# Plot the nodes of the graph with a different color for the warehouses and the delivery points

plt.plot(amazon_delivery_drones_case_2['longitude'][warehouse_index_case_2], amazon_delivery_drones_case_2['latitude'][warehouse_index_case_2], 'ro')
plt.plot(amazon_delivery_drones_case_2['longitude'][delivery_point_index_case_2], amazon_delivery_drones_case_2['latitude'][delivery_point_index_case_2], 'bo')

# Create the legend
plt.legend(['Warehouse', 'Delivery Point'])

# Create the axis

plt.xlabel('Longitude')
plt.ylabel('Latitude')

# plt.title('Amazon Delivery Drones Case 2')

plt.show()
