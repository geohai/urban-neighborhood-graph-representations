import mapillary.interface as mly
import json
import random
from helper_funcs import *
import requests
import pandas as pd
import torch
from torch import nn
import torchvision
from torchvision import transforms
from PIL import Image
import copy


preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#--------------- SETUP --------------#
# read node list
nodes = pd.read_csv('data_files/node_list.csv')
node_list = np.array(nodes['RegionID'])
# TODO: add bbox: bboxes = node_list = np.array(nodes['bbox'])
bbox = {'west': 8.944175, 'south': 46.001107, 'east': 8.954180, 'north': 46.003565 } 

start_time = '2019-11-01' # string in ISO format
end_time = '2023-01-01' # string in ISO format
username = 'julialromero'
res=1024
client_id = 'MLY|5990009514397174|ec6d0114c86e5a941092e3d2731c87a0'
BASE_DIR = 'mapillary/'
device = 'cpu'


# POI and SV do not require an actual graph structure
num_nodes = len(node_list)
node_embedding_length = 200
num_sample = 50

mly.set_access_token(client_id)

createCleanDir(BASE_DIR)
createDir(BASE_DIR)

bbox = {'west': -105.289433, 'south': 40.001297, 'east': -105.244103, 'north': 40.033032} # random dummy bounding box

model = EncoderInception3()

# for each region, collect images and feed thru inception; store in dict
region_bow_map = {}
for region in range(0, num_nodes):
    print('-----REGION INCREMENT------')
    print(region)
    bb_data = json.loads(
        mly.images_in_bbox(bbox,
                            max_captured_at="*",
                            min_captured_at="2012-01-01",
                            image_type="pano",
                            compass_angle=(0, 360),) 
    )
    bbox = dummy_bbox(bbox) # shifting bbox over for dummy data

    # now randomly sample num_sample images, query, and save
    createDir(BASE_DIR + str(region))
    imgList = []
    for i in range(0, num_sample):
        filename = BASE_DIR + str(region) + "/" + str(i) + ".jpg"
        sample = random.choice(bb_data['features'])
        query = sample['properties']['id']
        url = mly.image_thumbnail(image_id=query, resolution=res)

        # get_thumbnail(image_id, app_access_token)
        with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
            data = response.read() # a `bytes` object
            out_file.write(data)

            # Log filename and geo coordinates
            coords = ",".join(map(str, sample['geometry']['coordinates']))
            imgList.append([filename, coords])

        writeGeo(BASE_DIR + str(region), imgList)

    bow_list = []
    batch = None
    for file, coord in imgList:
        img = Image.open(file).convert('RGB')
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
        if batch == None:
            batch = input_batch
        else:
            batch = torch.concat((batch, input_batch), 0)

    img_embedding_tensor = model(batch.float())
        
    # create bow embeddings; store in dict 
    region_bow_map[str(region)] = img_embedding_tensor.tolist().copy()

    with open('data_files/sv.json', 'w') as fp:
        json.dump(region_bow_map, fp)
