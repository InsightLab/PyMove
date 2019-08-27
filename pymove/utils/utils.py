# TODO: Andreza
import numpy as np
import pandas as pd
import time
from scipy.interpolate import interp1d

from pymove import utils as ut
from pymove import gridutils

"""main labels """
dic_labels = {"id" : 'id', 'lat' : 'lat', 'lon' : 'lon', 'datetime' : 'datetime'}

dic_features_label = {'tid' : 'tid', 'dist_to_prev' : 'dist_to_prev', "dist_to_next" : 'dist_to_next', 'dist_prev_to_next' : 'dist_prev_to_next', 
                    'time_to_prev' : 'time_to_prev', 'time_to_next' : 'time_to_next', 'speed_to_prev': 'speed_to_prev', 'speed_to_next': 'speed_to_next',
                    'period': 'period', 'day': 'day', 'index_grid_lat': 'index_grid_lat', 'index_grid_lon' : 'index_grid_lon',
                    'situation':'situation'}

def format_labels(df_, current_id, current_lat, current_lon, current_datetime):
    """ 
    Format the labels for the PyRoad lib pattern 
        labels output = lat, lon and datatime
    """ 
    dic_labels['id'] = current_id
    dic_labels['lon'] = current_lon
    dic_labels['lat'] = current_lat
    dic_labels['datetime'] = current_datetime
    return dic_labels
    
def show_trajectories_info(df_, dic_labels=dic_labels):
    """
        show dataset information from dataframe, this is number of rows, datetime interval, and bounding box 
    """
    try:
        print('\n======================= INFORMATION ABOUT DATASET =======================\n')
        print('Number of Points: {}\n'.format(df_.shape[0]))
        if dic_labels['id'] in df_:
            print('Number of IDs objects: {}\n'.format(df_[dic_labels['id']].nunique()))
        if dic_features_label['tid'] in df_:
            print('Number of TIDs trajectory: {}\n'.format(df_[dic_features_label['tid']].nunique()))
        if dic_labels['datetime'] in df_:
            print('Start Date:{}     End Date:{}\n'.format(df_[dic_labels['datetime']].min(), df_[dic_labels['datetime']].max()))
        if dic_labels['lat'] and dic_labels['lon'] in df_:
            print('Bounding Box:{}\n'.format(get_bbox(df_, dic_labels))) # bbox return =  Lat_min , Long_min, Lat_max, Long_max) 
        if dic_features_label['time_to_prev'] in df_:            
            print('Gap time MAX:{}     Gap time MIN:{}\n'.format(round(df_[dic_features_label['time_to_prev']].max(),3), round(df_[dic_features_label['time_to_prev']].min(), 3)))
        if dic_features_label['speed_to_prev'] in df_:            
            print('Speed MAX:{}    Speed MIN:{}\n'.format(round(df_[dic_features_label['speed_to_prev']].max(), 3), round(df_[dic_features_label['speed_to_prev']].min(), 3))) 
        if dic_features_label['dist_to_prev'] in df_:            
            print('Distance MAX:{}    Distance MIN:{}\n'.format(round(df_[dic_features_label['dist_to_prev']].max(),3), round(df_[dic_features_label['dist_to_prev']].min(), 3))) 
            
        print('\n=========================================================================\n')
    except Exception as e:
        raise e    

def get_bbox(df_, dic_labels=dic_labels):
    """
    A bounding box (usually shortened to bbox) is an area defined by two longitudes and two latitudes, where:
    Latitude is a decimal number between -90.0 and 90.0. Longitude is a decimal number between -180.0 and 180.0.
    They usually follow the standard format of: 
    bbox = left,bottom,right,top 
    bbox = min Longitude , min Latitude , max Longitude , max Latitude 
    """
    try:
        return (df_[dic_labels['lat']].min(), df_[dic_labels['lon']].min(), df_[dic_labels['lat']].max(), df_[dic_labels['lon']].max())
    except Exception as e:
        raise e

def save_bbox(bbox_tuple, file, tiles='OpenStreetMap', color='red'):
    m = folium.Map(tiles=tiles)
    m.fit_bounds([ [bbox_tuple[0], bbox_tuple[1]], [bbox_tuple[2], bbox_tuple[3]] ])
    points_ = [ (bbox_tuple[0], bbox_tuple[1]), (bbox_tuple[0], bbox_tuple[3]), 
                (bbox_tuple[2], bbox_tuple[3]), (bbox_tuple[2], bbox_tuple[1]),
                (bbox_tuple[0], bbox_tuple[1]) ]
    folium.PolyLine(points_, weight=3, color=color).add_to(m)
    m.save(file) 