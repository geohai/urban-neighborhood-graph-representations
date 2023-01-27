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
import sys
import tqdm
sys.path.append('safegraph/utils/')

preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_random(number, polygon):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    print(minx)
    print(miny)
    print(maxx)
    print(maxy)
    while len(points) < number:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points

def point_in_poly(coords, ply):
    point = Point(coords[0], coords[1])
    if ply.contains(point):
        return True
    return False

#--------------- SETUP --------------#
# read node list
# read obj
graph_obj_path = 'safegraph/compute_graph_checkpoints/checkpoint_0.pkl'
with open(graph_obj_path, 'rb') as f:
    g = pickle.load(f) # CensusTractMobility object
idx_node_map = g.get_idx_node() # dictionary = (idx: region_id)

start_time = '2019-11-01' # string in ISO format
end_time = '2023-01-01' # string in ISO format
username = 'julialromero'
res=1024
client_id = 'MLY|5990009514397174|ec6d0114c86e5a941092e3d2731c87a0'
BASE_DIR = 'mapillary/'
device = 'cpu'


# POI and SV do not require an actual graph structure
# num_nodes = len(node_list)
node_embedding_length = 200
num_sample = 50
mly.set_access_token(client_id)


def query_mapillary(geom):
    minx, miny, maxx, maxy = geom.bounds
    bbox = {'west': minx, 'south': miny, 'north': maxy, 'east':maxx}

    bb_data = json.loads(
        mly.images_in_bbox(bbox,
                            max_captured_at="*",
                            min_captured_at="2012-01-01",
                            image_type="pano",
                            compass_angle=(0, 360),) 
    )

    return bb_data


createCleanDir(BASE_DIR)
createDir(BASE_DIR)

model = EncoderInception3()

# for each region, collect images and feed thru inception; store in dict
region_bow_map = {}
for geoid, geom in zip(g.tract_data['GEOID'], g.tract_data['geometry']):
    print('-----REGION INCREMENT------')
    bb_data = query_mapillary(geom)


    # now randomly sample num_sample images, query, and save
    createDir(BASE_DIR + str(geoid))
    imgList = []
    for i in tqdm(range(0, num_sample)):
        filename = BASE_DIR + str(geoid) + "/" + str(i) + ".jpg"
        in_poly = False
        while in_poly == False:
            sample = random.choice(bb_data['features'])  # TODO randomly choose 59 all at once
            in_poly = point_in_poly(bb_data['features'][i]['geometry']['coordinates'], geom)

            
        url = mly.image_thumbnail(image_id=sample['properties']['id'], resolution=res)

        # get_thumbnail(image_id, app_access_token)
        with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
            data = response.read() # a `bytes` object
            out_file.write(data)

            # Log filename and geo coordinates
            coords = ",".join(map(str, sample['geometry']['coordinates']))
            imgList.append([filename, coords])

        writeGeo(BASE_DIR + str(geoid), imgList)

    bow_list = []
    batch = None
    for file, coords in imgList:
        img = Image.open(file).convert('RGB')
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
        if batch == None:
            batch = input_batch
        else:
            batch = torch.concat((batch, input_batch), 0)

    img_embedding_tensor = model(batch.float())
        
    # create bow embeddings; store in dict 
    region_bow_map[str(geoid)] = img_embedding_tensor.tolist().copy()

    with open('data_files/sv.json', 'w') as fp:
        json.dump(region_bow_map, fp)
