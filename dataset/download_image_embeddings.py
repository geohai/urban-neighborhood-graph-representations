import mapillary.interface as mly
import json
import random
from helper_funcs import *
import requests
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import copy
import pickle
import random
from shapely.geometry import Point
import tqdm
import sys
sys.path.append('safegraph/utils/')
sys.path.append('mapillary/')
import os
from shapely.geometry import box

#--------------- SETUP --------------#
# Load checkpoint or start new
with open('data_files/final_node_data/sv.json', 'r') as fp:
        region_bow_map = json.load(fp)
# region_bow_map = {}


graph_obj_path = 'safegraph/compute_graph_checkpoints/grandjunction_denver/checkpoint_0.pkl'
with open(graph_obj_path, 'rb') as f:
    g = pickle.load(f) # CensusTractMobility object
idx_node_map = g.get_idx_node() # dictionary = (idx: region_id)

start_time = '2019-11-01' # string in ISO format
end_time = '2023-01-01' # string in ISO format
username = 'julialromero'
res=1024

with open('mapillary/mapillary_key.txt', 'r') as f:
    key = f.readline()
client_id = key
BASE_DIR = 'mapillary/data/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# POI and SV do not require an actual graph structure
node_embedding_length = 200
num_sample = 50
mly.set_access_token(client_id)

def query_mapillary(geom, imgfilter='all'):
    minx, miny, maxx, maxy = geom.bounds
    bbox = {'west': minx, 'south': miny, 'north': maxy, 'east':maxx}
    # print(bbox)
    # print(f"{bbox['west']}, {bbox['south']},{bbox['east']},{bbox['north']}")

    bb_data = json.loads(
        mly.images_in_bbox(bbox,
                            # max_captured_at="*",
                            # min_captured_at="2012-01-01",
                            image_type=imgfilter,) # compass_angle=(0, 360),
    )
    return bb_data
def Count_files_in_subd():
    l = []
    
    for root, dirs, files in os.walk("mapillary/data"):
        # print("{} in {}".format(len(files), root))
        l.append(len(files))

    print(f'Num folders total: {len(os.listdir("mapillary/data"))}')
    print(f'Num folders with 0 files: {l.count(0)}')
    print()
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def point_in_poly(coords, ply):
    point = Point(coords[0], coords[1])
    
    if ply.contains(point):
        return True
    print(point)
    print(ply)
    return False
def remove_images_not_in_bbox(full_image_list, geom):
    bbox = box(*geom.bounds)
    print(bbox)
    i = 0
    while True:
        if i == len(full_image_list):
            break
        in_poly = point_in_poly(full_image_list[i]['geometry']['coordinates'], bbox)
        if in_poly == False:
            full_image_list.remove(full_image_list[i])
        else: 
            i+=1
    return full_image_list
# createCleanDir(BASE_DIR)
# createDir(BASE_DIR)

model = EncoderInception3()
Count_files_in_subd()
with open('data_files/final_node_data/log_image_embedding.txt', 'w') as fp:
        fp.write(' ')

print(f'Num of regions: {len(g.tract_data)}')
not_found_counter = 0
for region_counter, (geoid, geom) in tqdm.tqdm(enumerate(zip(g.tract_data['GEOID'], g.tract_data['geometry']))):
    if region_counter < 115:
        continue
    
    print(f'\n\n\n-----REGION COUNTER {region_counter}------')
    print(f'Region id: {geoid}')
    bb_data = query_mapillary(geom, 'all')

    # now randomly sample num_sample images, query, and save
    createDir(BASE_DIR + str(geoid))
    imgList = []
    full_image_list = bb_data['features'].copy()
    
    full_image_list = remove_images_not_in_bbox(full_image_list, geom)
    if len(full_image_list) == 0:
        region_bow_map[str(geoid)] = None
        print('No images found in this area.')
        not_found_counter+=1
        continue

    for i in range(0, num_sample):
        filename = BASE_DIR + str(geoid) + "/" + str(i) + ".jpg"

        sample = random.choice(full_image_list)  # TODO randomly choose 50 all at once  
        url = mly.image_thumbnail(image_id=sample['properties']['id'], resolution=res)

        with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
            data = response.read() # a `bytes` object
            out_file.write(data)

            # Log filename and geo coordinates
            coords = ",".join(map(str, sample['geometry']['coordinates']))
        coords = ' '
        imgList.append([filename, coords])

        writeGeo(BASE_DIR + str(geoid), imgList)
        print('Log: Saved image')

    with open('data_files/final_node_data/log_image.txt', 'a') as fp:
        fp.write(f'{region_counter}: {geoid} \n')


    bow_list = []
    input_batch = None
    batch = None
    for file, coords in imgList:
        if not os.path.isfile(file):
            continue
        img = Image.open(file).convert('RGB')
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
        if batch == None:
            batch = input_batch
        else:
            batch = torch.concat((batch, input_batch), 0)

    # create bow embeddings; store in dict 
    img_embedding_tensor = model(batch.float())
    region_bow_map[str(geoid)] = img_embedding_tensor.tolist().copy()

    with open('data_files/final_node_data/sv.json', 'w') as fp:
        json.dump(region_bow_map, fp)

    with open('data_files/final_node_data/log_image_embedding.txt', 'a') as fp:
        fp.write(f'{region_counter}: {geoid} \n')
    
    if region_counter % 50 == 0:
        print(f'-- SAVED {region_counter}')

    print(f'Num regions with no images: {not_found_counter}')
