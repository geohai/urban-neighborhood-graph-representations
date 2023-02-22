import mapillary.interface as mly
import json
import random
from helper_funcs import *
from torchvision import transforms
from PIL import Image
import pickle
import random
from shapely.geometry import Point
import tqdm
import sys
sys.path.append('../safegraph/')
import os
from shapely.geometry import box
import urllib
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
logging.getLogger("urllib3").setLevel(logging.WARNING)

#--------------- SETUP --------------#
# Load checkpoint or start new
# with open('data_files/final_node_data/sv.json', 'r') as fp:
#         region_bow_map = json.load(fp)
# region_bow_map = {}

graph_obj_path = '../safegraph/graph_checkpoints/nyc_metro/checkpoint_1.pkl'
with open(graph_obj_path, 'rb') as f:
    g_ = pickle.load(f) # CensusTractMobility object




# POI and SV do not require an actual graph structure
node_embedding_length = 200
num_sample = 50


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
    return False
def remove_images_not_in_bbox(full_image_list, geom):
    bbox = box(*geom.bounds)
    # print(bbox)
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



# ---------------- DOWNLOAD IMAGES ----------------- #

def thread_routine(geoid, geom, region_counter, BASE_DIR, res=256):
    try:
        if os.path.exists(BASE_DIR + str(geoid)):
            if len(os.listdir(BASE_DIR + str(geoid))) >= 50:
                # print(geoid)
                return 'already downloaded'
            else:
                createCleanDir(BASE_DIR + str(geoid))

        # print(f'\n\n\n-----REGION COUNTER {region_counter}------')
        print(f'Region id: {geoid}')
        bb_data = query_mapillary(geom, 'all')

        # now randomly sample num_sample images, query, and save
        createDir(BASE_DIR + str(geoid))
        imgList = []
        full_image_list = bb_data['features'].copy()
        
#         full_image_list = remove_images_not_in_bbox(full_image_list, geom)
        
        if len(full_image_list) == 0:
            # region_bow_map[str(geoid)] = None
            # not_found_counter+=1
            return f'No images found in {geoid}.'
            

        for i in range(0, num_sample):
            # print('in loop')
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

        with open('data/nyc_metro/log_image.txt', 'a') as fp:
            fp.write(f'{region_counter}: {geoid} \n')

        return f'Finished Region {region_counter} -- {geoid}'

    except Exception as e:
        print(f'-------\nEXCEPT:  {e}\n-------')
        return e

def runner(g, BASE_DIR='data/nyc_metro/', res=256):
    
    
    threads = []
    iterate_over = enumerate(zip(g['GEOID'], g['geometry']))
    # 10 = 73 per hour
    with ThreadPoolExecutor(max_workers=10) as executor:
        for region_counter_, (geoid_, geom_) in iterate_over:
            threads.append(executor.submit(thread_routine, geoid_, geom_, region_counter_, BASE_DIR))
        
        for task in as_completed(threads):
            print(task.result())

if __name__ == "__main__":
    start_time = '2019-11-01' # string in ISO format
    end_time = '2023-01-01' # string in ISO format
    res=256

    with open('mapillary_key.txt', 'r') as f:
        key = f.readline()
    client_id = key
    mly.set_access_token(client_id)
    BASE_DIR = 'data/nyc_metro/'
    
    print(f'Num of regions: {len(g_.tract_data)}')
    runner(g_.tract_data)
