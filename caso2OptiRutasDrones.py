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
    demand_case_2[index] = (f'{row["NODE_TYPE"]}_{index}',row['DEMAND'])

# Create the DEMAND csv file
with open('demand_case_2.csv', 'w') as f:
    for key in demand_case_2.keys():
        f.write(f'{demand_case_2[key][0]}\t{np.round(demand_case_2[key][1],0)}\n')

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


