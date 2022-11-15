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
amazon_delivery_drones = pd.read_csv('AmazonDroneDelivery.csv', sep='\t', index_col=0)

# %% [markdown]
# # Caso 1 plot global

# %%
fig = px.scatter_mapbox(amazon_delivery_drones,
                        lat="latitude",
                        lon="longitude",
                        color="NODE_TYPE",
                        title="Amazon Delivery Drones",
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
distances = {}
# Iterate over the rows of the dataframe
for index, row in amazon_delivery_drones.iterrows():
    for index2, row2 in amazon_delivery_drones.iterrows():
        # Calculate the distance between the two rows using the haversine formula
        if index != index2:

           distance = haversine((row['latitude'], row['longitude']), (row2['latitude'], row2['longitude']))
        else:
            distance=999

        # Add the distance to the dictionary
        distances[(index, index2)] = distance


# %%
# Create a csv file with the distances
import csv
print(amazon_delivery_drones.shape)

last_index = amazon_delivery_drones.index.max()

print(last_index)
with open('distances.csv', 'w') as f:
    f.write(' \t')
    for index, row in amazon_delivery_drones.iterrows():
        node = row['NODE_TYPE']
        if  index == last_index:
            f.write( f'{node}_{index}')
        else:
            f.write( f'{node}_{index}\t')

    f.write('\n')

    for index, row in amazon_delivery_drones.iterrows():
        node = row['NODE_TYPE']
        f.write(f'{node}_{index}\t')

        for index2, row2 in amazon_delivery_drones.iterrows():
            if index2 == last_index:
                f.write(f'{np.round(distances[(index, index2)],3)}')
            else:
                f.write(f'{np.round(distances[(index, index2)],3)}\t')
        count_col = 0
        f.write('\n')
# %%
# Create the DEMAND dictionary
demand = {}
for index, row in amazon_delivery_drones.iterrows():
    demand[index] = (f'{row["NODE_TYPE"]}_{index}',row['DEMAND'])

# Create the DEMAND csv file
with open('demand.csv', 'w') as f:
    for key in demand.keys():
        f.write(f'{demand[key][0]}\t{np.round(demand[key][1],0)}\n')
# %%
# Get the mean of distances
mean_distance = np.mean(list(distances.values()))
mean_distance = np.round(mean_distance, 0)
# Get the mean of demands
mean_demand = 0
for index, row in amazon_delivery_drones.iterrows():
    if row['NODE_TYPE'] == 'delivery_point':
        mean_demand += row['DEMAND']

mean_demand = mean_demand/len(amazon_delivery_drones[amazon_delivery_drones['NODE_TYPE'] == 'delivery_point'])


mean_demand = np.round(mean_demand,0)

print(f'Distancia media: {mean_distance}[km]')
print(f'Demanda media: {mean_demand}[kg]')

# %%
# Create a random uniform capacity for the drones
n_drones = len(amazon_delivery_drones[amazon_delivery_drones['NODE_TYPE'] == 'warehouse']) * 3
capacity = np.random.uniform(0.5*mean_demand, 1.5*mean_demand, n_drones)
capacity = np.round(capacity,0)
# Create a random uniform battery range for the drones
mean_battery_range = 4*mean_distance
battery_range = np.random.uniform(0.5*mean_battery_range, 1.5*mean_battery_range, n_drones)
# %%
# Create the info_drones dictionary
info_drone_dict = {}
for i in range(n_drones):
    info_drone_dict[i] = (f'drone_{i}', capacity[i], battery_range[i])

# Create the info_drones.csv file
with open('info_drones.csv', 'w') as f:
    for key in info_drone_dict.keys():
        f.write(f'{info_drone_dict[key][0]}\t{np.round(info_drone_dict[key][1],0)}\t{np.round(info_drone_dict[key][2],0)}\n')

# %% [markdown]
# # One hot encoding NODE_TYPE

# %%
# Make one hot encoding for the NODE_TYPE column
amazon_delivery_drones = pd.get_dummies(amazon_delivery_drones, columns=['NODE_TYPE'])

# Create an array that contains wich index is a warehouse

warehouse_index = amazon_delivery_drones[amazon_delivery_drones['NODE_TYPE_warehouse'] == 1].index

# Create an array that constains wich index is a delivery point

delivery_point_index = amazon_delivery_drones[amazon_delivery_drones['NODE_TYPE_delivery_point'] == 1].index

# Create an array that contains wich index is a charging station

charging_station_index = amazon_delivery_drones[amazon_delivery_drones['NODE_TYPE_charging_station'] == 1].index


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

# %%

# Create the Sets


