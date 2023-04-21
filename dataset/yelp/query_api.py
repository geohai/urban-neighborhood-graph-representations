from __future__ import print_function

import argparse
import json
import pprint
import requests
import sys
import urllib
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.parse import urlencode
import random
from shapely.geometry import Point
root = '/mnt/e/julia/regional-representations-graph-model/dataset/'
sys.path.append(root + 'safegraph/')

import pickle
# https://www.yelp.com/developers/v3/manage_app
with open(root + 'api_keys/yelp_key2.txt') as f:
    lines = f.readline()
API_KEY = lines

# API constants, you shouldn't have to change these.
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
BUSINESS_PATH = '/v3/businesses/'  # Business ID will come after slash.


# Defaults for our simple example.
SEARCH_LIMIT = 50
# num_reviews = 5

def request(host, path, api_key, url_params=None):
    """Given your API_KEY, send a GET request to the API.
    Args:
        host (str): The domain host of the API.
        path (str): The path of the API after the domain.
        API_KEY (str): Your API Key.
        url_params (dict): An optional set of query parameters in the request.
    Returns:
        dict: The JSON response from the request.
    Raises:
        HTTPError: An error occurs from the HTTP request.
    """
    url_params = url_params or {}
    url = '{0}{1}'.format(host, quote(path.encode('utf8')))
    headers = {
        'Authorization': 'Bearer %s' % api_key,
    }

    print(u'Querying {0} ...'.format(url))

    response = requests.request('GET', url, headers=headers, params=url_params)

    return response.json()

def search(api_key, longitude, latitude):
    """Query the Search API by a search term and location.
    Returns:
        dict: The JSON response from the request.
    """

    url_params = {
        'longitude': longitude.replace(' ', '+'),
        'latitude': latitude.replace(' ', '+'),
        'radius': '800'.replace(' ', '+'), # 0.5 miles
        'limit': SEARCH_LIMIT
    }
    return request(API_HOST, SEARCH_PATH, api_key, url_params=url_params)

# def get_business_reviews(api_key, business_id):
#     """Query the Business API by a business ID.
#     Args:
#         business_id (str): The ID of the business to query.
#     Returns:
#         dict: The JSON response from the request.
#     """
#     business_path = BUSINESS_PATH + business_id + '/reviews'

#     return request(API_HOST, business_path, api_key)

def query_api(longitude, latitude):
    response = search(API_KEY,  longitude, latitude)

    businesses = response.get('businesses')

    if not businesses:
        print(u'No businesses for {0},{1} found.'.format(longitude, latitude))
        return ''

    # append all words to a sentence
    bow_list = ''
    for business in businesses:
        for cat in business['categories']: # get category, prices, customer reviews
            try:
                bow_list += cat['alias'] + ' '
            except:
                pass
#         try:
#             bow_list += business['price'] + ' '
#         except:
#             pass
        
        business_id = business['id']
#         print(f'Business ID: {business_id}')
#         reviews = get_business_reviews(API_KEY, business_id).get('reviews')
#         if (reviews != None):
#             temp_num_reviews = min(len(reviews), num_reviews)
#             for review in reviews[0:temp_num_reviews]:
#                 bow_list += review['text'] + ' '
#     print(bow_list)

    print('----')
    return bow_list
