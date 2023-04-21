# Median Age, Years of Education, and
# Percentage of White Population as demographic attributes, and
# Poverty Rate, Average Household Income and Employment Rate
# as economic attributes

# linear regression and RF regression
# train set - 85%, test - 15%
# evaluate on R^2

import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
import pickle
from census import Census
import sys, os
import numpy as np
from sklearn.decomposition import PCA
sys.path.append('dataset/safegraph/utils')

for root, dirs, files in os.walk('dataset/'):
    for d in dirs:
        sys.path.append(os.path.join(root, d))

sys.path.append('train')
from models import *

root = '/mnt/e/julia/regional-representations-graph-model/'

with open(root + 'dataset/api_keys//census_key.txt', 'r') as f:
    key = f.readline()
c = Census(key)

# read in embeddings
with open(root + 'dataset/safegraph/graph_checkpoints/nyc_metro/checkpoint_norm.pkl', 'rb') as f:
    g = pickle.load(f)
num_nodes = g.num_nodes
idx_node_map = g.get_idx_node()

save_dir = root + 'train/outputs/distance_1000/' #'../train/outputs/1_poi/'
doc=save_dir + 'distance_30_last.tar'
checkpoint=torch.load(doc)
model = NodeEmbeddings(num_nodes, embedding_dim=200)
model.load_state_dict(checkpoint,strict=False)

for param in model.parameters():
      embeddings = param
    
df = pd.DataFrame(embeddings.detach().numpy()).reset_index()
df_idx = pd.DataFrame.from_dict(idx_node_map, orient='index').reset_index().rename(columns={0:'GEOID'})
df_emb = df_idx.merge(df, on='index')
# https://api.census.gov/data/2019/acs/acs5/variables.html

# 'B01001_002E' - men
# 'B01001_026E' - women
# B01001_001E - sex by age total
# B02001_002E - total white (including hispanic)
# B02001_003E - total black
# B02001_004E - american indian or alaskan native
# B02001_005E - asian
# B02001_006E - hawaiian/pacific islander
# B02001_007E - other
# B01001H_001E - white, not hispanic/latino
# B01001I_001E - hispanic / latino
# B01002_001E - total median age
# B19013_001E - median household income in past 12 months
# B19083_001E gini index of income inequality
# B27001_001E - health insurance coverage status

# B08101_009E - num who drive to work alone (car, truck van)
# B08101_017E - num who carpool in car, truck or van to work
# B08101_025E - num who take public transit
# B08101_033E - num who walked
# B08101_049E - num who work from home
var = ['NAME', 'B01001_001E', 'B01001_026E', 'B01001_002E', 'B02001_003E', 'B02001_004E', 'B02001_005E', 'B02001_006E', 'B02001_007E', 
        'B01001H_001E', 'B01001I_001E', 'B01002_001E', 'B19013_001E', 'B19083_001E', 'B27001_001E', 
        'B08101_009E', 'B08101_017E', 'B08101_025E', 'B08101_033E', 'B08101_049E']
name = ['NAME', 'total_pop',  'num_women', 'num_men', 'total_black', 'total_na', 'total_asian', 'total_paisl', 'total_other', 'total_white', 'total_latino', 'total_median_age', 
        'median_household_income', 'income_gini_index', 'health_insurance_coverage', 'num_drive_alone', 'num_carpool', 'num_public_transit', 'num_walk', 'num_wfh']
name_compute_ratio = ['num_women', 'num_men', 'total_black', 'total_na', 'total_asian', 'total_paisl', 'total_other', 'total_white', 'total_latino', 'health_insurance_coverage', 
                      'num_drive_alone', 'num_carpool', 'num_public_transit', 'num_walk', 'num_wfh']

census = c.acs5.state_county_tract(fields = var,
                                    state_fips = '08',
                                    county_fips = '*',
                                    tract = "*",
                                    year = 2020)

df = pd.DataFrame(census)
df.rename(columns=dict(zip(var, name)), inplace=True)
for col in name_compute_ratio:
    df[col] = df[col]/df['total_pop']

# combine embeddings with labels
df['GEOID'] = df['state'].astype(str) + df['county'].astype(str) + df['tract'].astype(str)
df_emb = df_emb.merge(df, on='GEOID', how='left')
df_emb.drop(columns=['state', 'county', 'tract', 'index'], inplace=True)
df_emb.drop(index=df_emb[df_emb.isna().any(axis=1)].index, inplace=True)

predict_labels = ['total_white', 'total_black', 'total_median_age', 'median_household_income', 'income_gini_index', 'health_insurance_coverage']

for lab in predict_labels:
    X = np.array(df_emb[[i for i in range(0, 200)]])
    y = np.array(df_emb[lab])

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=42)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # binarize labels
    pre = preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal')
    # pre = preprocessing.Binarizer(threshold=np.median(y_train))
    pre.fit(y_train)

    y_train = np.ravel(pre.transform(y_train))
    y_test =  np.ravel(pre.transform(y_test))

    print(f'\n------Feature {lab}------')
    print('- Logistic Regression -')
    pipe = make_pipeline(PCA(n_components=20), preprocessing.StandardScaler(), LogisticRegression())
    pipe.fit(X_train, y_train)
    print(pipe.score(X_train, y_train))
    print(pipe.score(X_test, y_test))

    print('- RF Classification -')
    pipe = make_pipeline(PCA(n_components=20), preprocessing.StandardScaler(), RandomForestClassifier(min_samples_split=4, max_depth=5))
    pipe.fit(X_train, y_train)
    print(pipe.score(X_train, y_train))
    print(pipe.score(X_test, y_test))
