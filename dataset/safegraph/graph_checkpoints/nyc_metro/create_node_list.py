import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_regions_to_retry():
    l = []
    counter=0
    for root, dirs, files in os.walk("../earth_engine/download_landsat_images/"):
        if '.ipynb_checkpoints' in root:
            continue
        if root == 'download_landsat_images/':
            continue
        for f in files:
            if f == 'least_cloudy_clipped_highres.tif':
                counter+=1
        if 'least_cloudy_clipped_highres.tif' not in files:
            l.append(root)
#             print(root)
            
#     if 'download_landsat_images/' in 
# #     l.remove('download_landsat_images/')
#     al = len(os.listdir("download_landsat_images/"))
#     print(f'Num folders total: {al}')
    print(f'Num target files total: {counter}')
    print()
    l = [item[-11:] for item in l]
    return l

def get_regions_to_retry_mapillary():
    l = []
    for root, dirs, files in os.walk("../mapillary/data/nyc_metro/"):
        if '.ipynb_checkpoints' in root:
            continue
        if root == 'data/':
            continue
        if root == 'nyc_metro/':
            continue
        if len(os.listdir(root)) == 0:
#             print(root)
            l.append(root)
   
    l = [item[-11:] for item in l]
    return l


tract_data_dir = r'../Tracts/nyc_metro_boundaries/nyc_metro_boundaries.shp'

df = gpd.read_file(tract_data_dir)
df.rename(columns={'GEOID_TRAC': 'GEOID'}, inplace=True)  
df = df[['GEOID', 'STATEFP', 'COUNTYFP', 'TRACTCE', 'geometry']] 

# no data here
geoids = get_regions_to_retry()
df.drop(index=df.loc[df.GEOID.isin(geoids)].index, inplace=True)

geoids = get_regions_to_retry_mapillary()
df.drop(index=df.loc[df.GEOID.isin(geoids)].index, inplace=True)

df.reset_index(inplace=True)

# for now just get 2 census tracts per county
select_idx = []
unique_codes = df[['STATEFP', 'COUNTYFP']].drop_duplicates()
for st, cty in zip(unique_codes['STATEFP'], unique_codes['COUNTYFP']):
    idx = df.loc[(df.STATEFP == st) & (df.COUNTYFP == cty)].index.tolist()
    try:
        select_idx.extend(idx[0:20])
    except Exception as e:
        print(e)
        continue


# print(df.index.values)
# print(select_idx)
# print(len(unique_codes))
select = df.iloc[select_idx]
print(f'Shape: {select.shape}')
select['GEOID'].to_csv('graph_checkpoints/nyc_metro/node_list.csv', index=False)


t = df.plot()
plt.savefig('graph_checkpoints/nyc_metro/map')
t = df.boundary.plot()
plt.savefig('graph_checkpoints/nyc_metro/boundarymap')

    

