import matplotlib
import matplotlib.pyplot as plt
import colorsys
import numpy as np
import folium
from folium.plugins import HeatMap, HeatMapWithTime, MarkerCluster, FastMarkerCluster
from matplotlib.colors import LinearSegmentedColormap
import itertools
import pandas as pd
from collections import namedtuple
import seaborn as sns
#from pymove import trajutils
from pymove.utils import constants
from tqdm import tqdm_notebook as tqdm
from folium import plugins

dict_labels = {'tnz_id':'id',
             'tnz_lat':'lat',
             'tnz_lon':'lon',
             'tnz_datetime':'datetime',
             
             'event_id':'id',
             'event_lat':'lat',
             'event_lon':'lon',
             'event_datetime':'datetime',

             'poi_id':'Nome do Local',
             'poi_lat':'Latitude',
             'poi_lon':'Longitude',
             
             'lat': 'lat',
             'lon': 'lon',
             'datetime': 'datetime',
             'id': 'id'}

dict_plot = {'radius': 150, 
            'event_point':'purple',
            'tnz_point':'orange', 
            'poi_point':'black',
            'line':'blue', 
            'start':'green', 
            'end':'red'}

trajutils_dic_labels = {"id":constants.TRAJ_ID, 'lat':constants.LATITUDE, 'lon':constants.LONGITUDE, 'datetime':constants.DATETIME}

def formate_tags(line, slice_):
    
    map_formated_tags = map(lambda tag: '{}: {}'.format(tag, line[tag]), slice_)

    return  '<br/>'.join(map_formated_tags)

# http://www.color-hex.com/color/

# Already on pymove.visualization 
def rgb(RGBcolors):
    """
    Return a tuple of integers, as used in AWT/Java plots. 
    
    Parameters
    ----------
    RGBcolors: tuple, list, numpy array
        A three position sequence with the percent of blue, red, and green.
    """
    blue  = RGBcolors[0]
    red   = RGBcolors[1]
    green = RGBcolors[2]
    return int(red*255), int(green*255), int(blue*255)

# Already on pymove.visualization
def hexRgb(RGBcolors):
    """ 
    Return a hex string, as used in Tk plots. 
    
    Parameters
    ----------
    RGBcolors: tuple, list, numpy array
        A three position sequence with the percent of blue, red, and green.
    """
    return "#%02X%02X%02X" % rgb(RGBcolors)
# Already on pymove.visualization
def cmap_hex_color(cmap, i):
    return matplotlib.colors.rgb2hex(cmap(i))
# Already on pymove.visualization                                              #label_id=trajutils.dic_labels['id']  dic_labels = trajutils.dic_labels
def save_map(df, file, tiles='OpenStreetMap', label_id=constants.TRAJ_ID, dic_labels = trajutils_dic_labels, cmap='tab20'):
    #df['lat'] = df['lat'].astype('float64')
    #df['lon'] = df['lon'].astype('float64')
    m = folium.Map(tiles=tiles)
    #m.fit_bounds([ [df[trajutils.dic_labels['lat']].min(), df[trajutils.dic_labels['lon']].min()], [df[trajutils.dic_labels['lat']].max(), df[trajutils.dic_labels['lon']].max()] ])
    m.fit_bounds([ [df[constants.LATITUDE].min(), df[constants.LONGITUDE].min()], [df[constants.LATITUDE].max(), df[constants.LONGITUDE].max()] ])
    
    ids = df[label_id].unique()
    
    #vals = np.linspace(0,1,256)
    #np.random.seed(color_seed)
    #np.random.shuffle(vals)
    #cmap_ = plt.cm.colors.ListedColormap(plt.cm.hsv(vals))
    # cmap_ = plt.cm.get_cmap( base_cmap, lut=ids.shape[0] )
    cmap_ = plt.cm.get_cmap( cmap )
    N = cmap_.N
    
    for id_ in ids:
        id_index = np.where(ids==id_)[0][0]
        df_ = df[ df[label_id] == id_ ]
        #print('id:{}, shape:{}'.format(id_, df_.shape))
        #tnz_points_ = [ (tnz_point[0], tnz_point[1]) for tnz_point in df_[[trajutils.dic_labels['lat'], trajutils.dic_labels['lon']]].values]
        tnz_points_ = [ (tnz_point[0], tnz_point[1]) for tnz_point in df_[[constants.LATITUDE, constants.LONGITUDE]].values]
        color_ = cmap_hex_color(cmap_, (id_index % N))
        #print( 'id_index:{}, color:{}'.format(id_index, color_) )
        folium.PolyLine(tnz_points_, weight=3, color=color_).add_to(m)
    m.save(file) 
# Already on pymove.visualization                           #trajutils.dic_labels['id']  #trajutils.dic_labels     
def save_wkt(df, file_str, label_id=constants.TRAJ_ID, dic_labels=trajutils_dic_labels):
    str_ = '{};linestring\n'.format(label_id)
    ids = df[label_id].unique()
    for id_ in ids:
        df_ = df[ df[label_id] == id_ ]
        str_ += '{};LINESTRING('.format(id_)
        str_ += ','.join('{} {}'.format(x[0],x[1]) for x in df_[dic_labels['lon'], dic_labels['lat']].values)
        str_ += ')\n'
    open(file_str, "w").write(str_)    
    
def invert_map(map_):
    inv_map = {}
    items_ = map_.items()
    for k, v in items_:
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map
    
""" https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html """
# Already on pymove.visualization
def show_object_id_by_date(df_, label_id = 'id', create_features=True,  kind=['bar', 'bar', 'line', 'line'], figsize=(21,9), save_fig=True, name='shot_tnz_points_by_date.png', low_memory=True):
    
    if low_memory:
         fig, ax = plt.subplots(2,2,figsize=figsize)
    
    else:
        fig, ax = plt.subplots(2,2,figsize=figsize)
        trajutils.create_update_date_features(df_)
        trajutils.create_update_day_of_the_week_features(df_)
        trajutils.create_update_hour_features(df_)
        trajutils.create_update_time_of_day_features(df_)  
        
        df_.groupby(['period'])['id'].nunique().plot(subplots=True, kind = kind[0], rot=0, ax=ax[0][0], fontsize=12)
        df_.groupby(['day'])['id'].nunique().plot(subplots=True,  kind = kind[1], ax=ax[0][1], rot=0, fontsize=12)
        df_.groupby(['date'])['id'].nunique().plot(subplots=True, kind = kind[2], grid=True, ax=ax[1][0], rot=90, fontsize=12)
        df_.groupby(['hour'])['id'].nunique().plot(subplots=True, kind = kind[3], grid=True, ax=ax[1][1], fontsize=12)

    if save_fig:
        plt.savefig(fname=name, fig=fig)
# Already on pymove.visualization                          #trajutils.dic_labels
def show_lat_lon_GPS(df_, dic_labels=trajutils_dic_labels, kind='scatter', figsize=(21,9), save_fig=False, plot_start_and_end = True, name='show_gps_tnz_points.png'):
    try:
        if dic_labels['lat'] in df_ and dic_labels['lon'] in df_:
            df_.drop_duplicates([dic_labels['lat'], dic_labels['lon']]).plot(kind=kind, x=dic_labels['lon'], y=dic_labels['lat'], figsize=figsize)          
        
            if plot_start_and_end:     
                plt.plot(df_.iloc[0][dic_labels['lon']], df_.iloc[0][dic_labels['lat']], 'yo', markersize=10)             # start tnz_point
                plt.plot(df_.iloc[-1][dic_labels['lon']], df_.iloc[-1][dic_labels['lat']], 'yX', markersize=10)           # end tnz_point
            if save_fig == True:
                plt.savefig(name)  
    except Exception as e:
        raise e
# Already on pymove.core.dataframe as plot_all_features
def show_all_features(df_, figsize=(21,15), dtype=np.float64, save_fig=True, name='features.png'):
    try:
        col_float = df_.select_dtypes(include=[dtype]).columns
        tam = col_float.size
        if(tam > 0):
            fig, ax = plt.subplots(tam,1, figsize=figsize)
            ax_count = 0
            for col in col_float:
                ax[ax_count].set_title(col)
                df_[col].plot(subplots=True, ax=ax[ax_count])
                ax_count+=1
            
            if save_fig:
                plt.savefig(fname=name, fig=fig)
    except Exception as e:
        raise e
# Already on pymove.core.dataframe as plot_trajs_id                     #trajutils.dic_labels['id']  trajutils.dic_label
def show_traj_mpl(df, label_tid=constants.TRAJ_ID, dic_labels=trajutils_dic_labels, figsize=(21,9), return_fig=True, markers='o', markersize=10):
    fig = plt.figure(figsize=figsize)
    ids = df[label_tid].unique()
    #colors = itertools.cycle(["r", "b", "g", "y", "black"])
    for id_ in ids:
        df_id = df[ df[label_tid] == id_ ]
        plt.plot(df_id[dic_labels['lon']], df_id[dic_labels['lat']], markers,  markersize=markersize)
    if return_fig:
        return fig
# Already on pymove.core.dataframe as plot_trajs_                              #trajutils.dic_features_label['tid'] trajutils.dic_labels
def show_traj_mpl_id(df, tid, label_tid=constants.TID, dic_labels=trajutils_dic_labels, figsize=(10,10), markersize=10):
    fig = plt.figure(figsize=figsize)
    df_ = df[ df[label_tid] == tid ]
    plt.plot(df_.iloc[0][dic_labels['lon']], df_.iloc[0][dic_labels['lat']], 'yo', markersize=markersize)             # start tnz_point
    plt.plot(df_.iloc[-1][dic_labels['lon']], df_.iloc[-1][dic_labels['lat']], 'yX', markersize=markersize)           # end tnz_point
    
    if 'isNode'not in df_:
        plt.plot(df_[dic_labels['lon']], df_[dic_labels['lat']])
        plt.plot(df_.loc[:, dic_labels['lon']], df_.loc[:, dic_labels['lat']], 'r.', markersize=markersize)  # tnz_points
    else:
        filter_ = df_['isNode'] == 1
        df_nodes = df_.loc[filter_]
        df_tnz_points = df_.loc[~filter_]
        plt.plot(df_nodes[dic_labels['lon']], df_nodes[dic_labels['lat']], linewidth=3)
        plt.plot(df_tnz_points[dic_labels['lon']], df_tnz_points[dic_labels['lat']])
        plt.plot(df_nodes[dic_labels['lon']], df_nodes[dic_labels['lat']], 'go', markersize=markersize)   # nodes
        plt.plot(df_tnz_points[dic_labels['lon']], df_tnz_points[dic_labels['lat']], 'r.', markersize=markersize)  # tnz_points  
    return df_, fig
# Already on pymove.visualization
def create_base_map(default_location, default_zoom_start=12):
    """
    Generate a folium map.
    Parameters
    ----------
    default_location : tuple.
        Represents coordinates lat, lon which will be the center of the map.
    default_zoom_start : int, optional, default 12.
        Represents the zoom which will be the center of the map.
    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map.
    """
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map
# Already on pymove.visualization as faster_cluster
def show_faster_cluster(
    move_data,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    save_as_html=False,
    filename='faster_cluster.html'):
    """
    Generate visualization of Faster Cluster using folium plugin.
    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.
    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map. If not entered, the first data from the dataset is
        used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map. If not entered, the first data from the dataset is
        used.
    zoom_start : int, optional, default 12.
        Represents the trajectory id.
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated using the function create_base_map(), with
        the lat_origin, lon_origin and zoom_start.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    filename : String, optional, default 'faster_cluster.html'.
        Represents the file name of new file .html.
    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map with visualization.
    """
    LATITUDE = 'lat'
    LONGITUDE = 'lon'

    move_data.reset_index(inplace=True)

    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_data.loc[0][LATITUDE]
            lon_origin = move_data.loc[0][LONGITUDE]
        base_map = create_base_map(default_location=[lat_origin, lon_origin], default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    callback = """\
    function (row) {
        var marker;
        marker = L.circle(new L.LatLng(row[0], row[1]), {color:'red'});
        return marker;
    };
    """
    FastMarkerCluster(move_data.loc[:n_rows, [LATITUDE, LONGITUDE]].values.tolist(), callback=callback).add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map
# Already on pymove.visualization as cluster
def show_clusters(
    move_data,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    save_as_html=False,
    filename='cluster.html'):
    """
    Generate visualization of Marker Cluster using folium plugin.
    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.
    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map. If not entered, the first data from the dataset is
        used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map. If not entered, the first data from the dataset is
        used.
    zoom_start : int, optional, default 12.
        Represents the trajectory id.
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated using the function create_base_map(), with
        the lat_origin, lon_origin and zoom_start.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    filename : String, optional, default 'cluster.html'.
        Represents the file name of new file .html.
    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map with visualization.
    """
    LATITUDE = 'lat'
    LONGITUDE = 'lon'

    move_data.reset_index(inplace=True)

    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_data.loc[0][LATITUDE]
            lon_origin = move_data.loc[0][LONGITUDE]
        base_map = create_base_map(default_location=[lat_origin, lon_origin], default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    mc = MarkerCluster()
    for row in move_data[:n_rows].iterrows():
        pop = "<b>Latitude:</b> " + str(row[1].lat) \
              + "\n<b>Longitude:</b> " + str(row[1].lon) \
              + "\n<b>Datetime:</b> " + str(row[1].datetime)
        mc.add_child(folium.Marker(location=[row[1].lat, row[1].lon], popup=pop))
    base_map.add_child(mc)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map

def add_traj_folium(df,
              dict_plot = dict_plot,
              dict_labels = dict_labels,
              sort=False,
              folium_map = None,
              slice_tags = None,
              tiles='OpenStreetMap'):
              
        def circle_maker(iter_tuple, map_):
            
            _,line = iter_tuple


            x = line[dict_labels['tnz_lat']]
            y = line[dict_labels['tnz_lon']]
            
            tags_formated = formate_tags(line,slice_tags)
            
            folium.Circle(
            radius=1,
            location=[x, y],
            popup=tags_formated,
            color=dict_plot['tnz_point'],
            fill=False
            ).add_to(map_)
        
        def plot_incial_end_points(list_rowns, map_):
            
            #plot the start tnz_point
            line = list_rowns[0][1]
            #tags_formated = str(line[slice_tags]).replace('\n','<br/>').replace('\s', ':\s')
            tags_formated = formate_tags(line,slice_tags)
            
            x = line[dict_labels['tnz_lat']]
            y = line[dict_labels['tnz_lon']]
            

            folium.Marker(
                location=[x, y],
                popup='START<br/>'+tags_formated,
                icon=folium.Icon(color='green')
            ).add_to(folium_map)

            # folium.Circle(
            #             radius=1,
            #             location=[x, y],
            #             tooltip='START<br/>'+tags_formated,
            #             color=dict_plot['start'],
            #             fill=False,
            #         ).add_to(folium_map)

            #plot the last tnz_point
            line = list_rowns[-1][1]
            #tags_formated = str(line[slice_tags]).replace('\n','<br/>').replace('\s', ':\s')
            tags_formated = formate_tags(line,slice_tags)

            x = line[dict_labels['tnz_lat']]
            y = line[dict_labels['tnz_lon']]
            
            # folium.Circle(
            #             radius=1,
            #             location=[x, y],
            #             tooltip='END<br/>'+tags_formated,
            #             color=dict_plot['end'],
            #             fill=False,
            #         ).add_to(folium_map)
            folium.Marker(
                location=[x, y],
                popup='END<br/>'+tags_formated,
                icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(folium_map)
        

        if not slice_tags:
            slice_tags = df.columns
    
        # If not have a map a map is create with mean to lat and lon
        if not folium_map:
            initial_lat = df[ dict_labels['tnz_lat'] ].mean()
            initial_lon = df[ dict_labels['tnz_lon'] ].mean()
            folium_map = create_folium_map([initial_lat, initial_lon], tiles=tiles)
        
        #if needs sort the data
        if sort:       
            df.sort_values(dict_labels['tnz_datetime'], inplace=True)

        #plot the lines
        tnz_points = list(zip(df[dict_labels['tnz_lat']],df[dict_labels['tnz_lon']]))
        
        folium.PolyLine(tnz_points, 
                        color= dict_plot['line'],weight=2).add_to(folium_map)
        
            
        list(map(lambda x: circle_maker(x,folium_map), df.iterrows()))   
        
        plot_incial_end_points(list(df.iterrows()), folium_map)

        return folium_map

def add_point_folium(df,
              dict_plot = dict_plot,
              dict_labels = dict_labels,
              folium_map = None,
              slice_tags = None,
              tiles='cartodbpositron'):

        if not slice_tags:
            slice_tags = df.columns
    
        # If not have a map a map is create with mean to lat and lon
        if not folium_map:
            initial_lat = df[ dict_labels['lat'] ].mean()
            initial_lon = df[ dict_labels['lon'] ].mean()
            folium_map = create_folium_map([initial_lat, initial_lon], tiles=tiles)
        
        def circle_maker(iter_tuple, map_):
            
            _,line = iter_tuple


            x = line[dict_labels['lat']]
            y = line[dict_labels['lon']]
            
            #tags_formated = str(line[slice_tags]).replace('\n','<br/>').replace('\s', ':\s')
            tags_formated = formate_tags(line,slice_tags)
            
            folium.Circle(
            radius=1,
            location=[x, y],
            popup=tags_formated,
            tooltip=tags_formated,
            color=dict_plot['poi_point'],
            fill=False
            ).add_to(map_)
        

        list(map(lambda x: circle_maker(x,folium_map), df.iterrows()))   

        return folium_map

def add_poi_folium(df,
              dict_plot = dict_plot,
              dict_labels = dict_labels,
              folium_map = None,
              slice_tags = None):

        if not slice_tags:
            slice_tags = df.columns
    
        # If not have a map a map is create with mean to lat and lon
        if not folium_map:
            initial_lat = df[ dict_labels['poi_lat'] ].mean()
            initial_lon = df[ dict_labels['poi_lon'] ].mean()
            folium_map = create_folium_map([initial_lat, initial_lon])
        
        def circle_maker(iter_tuple, map_):
            
            _,line = iter_tuple


            x = line[dict_labels['poi_lat']]
            y = line[dict_labels['poi_lon']]
            
            #tags_formated = str(line[slice_tags]).replace('\n','<br/>').replace('\s', ':\s')
            tags_formated = formate_tags(line,slice_tags)
            
            folium.Circle(
            radius=1,
            location=[x, y],
            popup=tags_formated,
            tooltip=tags_formated,
            color=dict_plot['poi_point'],
            fill=False
            ).add_to(map_)
        

        list(map(lambda x: circle_maker(x,folium_map), df.iterrows()))   

        return folium_map
        
def add_event_folium(df,
              dict_plot = dict_plot,
              dict_labels = dict_labels,
              folium_map = None,
              slice_tags = None,
              tiles='OpenStreetMap'):

        if not slice_tags:
            slice_tags = df.columns
    
        # If not have a map a map is create with mean to lat and lon
        if not folium_map:
            initial_lat = df[ dict_labels['event_lat'] ].mean()
            initial_lon = df[ dict_labels['event_lon'] ].mean()
            folium_map = create_folium_map([initial_lat, initial_lon], tiles=tiles)
        
        def circle_maker(iter_tuple, map_):
            
            _,line = iter_tuple


            x = line[dict_labels['event_lat']]
            y = line[dict_labels['event_lon']]
            
            #tags_formated = str(line[slice_tags]).replace('\n','<br/>').replace('\s', ':\s')
            tags_formated = formate_tags(line,slice_tags)
            
            folium.Circle(
            radius=1,
            location=[x, y],
            tooltip=tags_formated,
            color=dict_plot['event_point'],
            fill=False
            ).add_to(map_)

            folium.Circle(
            radius=dict_plot['radius'],
            location=[x, y],
            tooltip=tags_formated,
            color=dict_plot['event_point'],
            fill=False,
            fill_color=dict_plot['event_point']
            ).add_to(map_)


        

        list(map(lambda x: circle_maker(x,folium_map), df.iterrows()))   

        return folium_map
        
def show_trajs_with_event(df_tnz, 
                        window_time_tnz,
                        df_event,
                       window_time_event,
                       radius,
                      dict_plot = dict_plot,
                      dict_labels = dict_labels,
                    slice_event_show = None, 
                    slice_tnz_show = None):
    """
        Plot a trajectory, incluiding your tnz_points lat lon and your tags, and CVP
    """
        
    #building structure for deltas
    delta_event = pd.to_timedelta(window_time_event, unit='s')
    delta_tnz = pd.to_timedelta(window_time_tnz, unit='s')
    
    #length of df_tnz
    len_df_tnz = df_tnz.shape[0]
        
    #building structure for lat and lon array
    lat_arr = np.zeros(len_df_tnz)
    lon_arr = np.zeros(len_df_tnz)
    
    #folium map list
    folium_maps = []
    
    #for each cvp in df_cvp
    for _, line in df_event.iterrows():
        
        event_lat = line[dict_labels['event_lat']]
        
        event_lon = line[dict_labels['event_lon']]
        
        event_datetime = line[dict_labels['event_datetime']]
        
        event_id = line[dict_labels['event_id']]
                
        #building time window for cvp search
        start_time = pd.to_datetime(event_datetime - delta_event)
        end_time = pd.to_datetime(event_datetime + delta_event)
        
        #filtering df_tnz for time window
        df_filted=pd.DataFrame(trajutils.filter_by_datetime(df_tnz,
                                     dic_labels = dict_labels,
                                     startDatetime = start_time,
                                     endDatetime=end_time))
        
        #length of df_temp
        len_df_temp = df_filted.shape[0]
        
        #using the util part of the array for haversine function
        lat_arr[:len_df_temp] = event_lat
        lon_arr[:len_df_temp] = event_lon
        
        #building distances to cvp column
        df_filted['distances'] = trajutils.haversine(lat_arr[:len_df_temp],
                                         lon_arr[:len_df_temp], 
                                         df_filted[dict_labels['tnz_lat']],
                                         df_filted[dict_labels['tnz_lon']])
        
        #building nearby column
        df_filted['nearby'] = df_filted['distances'].map(lambda x: (x<=radius))
        
        #if any data for df_tnz in cvp time window is True
        if df_filted['nearby'].any():
            
            #buildng the df for the first tnz_points of tnz in nearby cvp
            df_begin = df_filted[df_filted['nearby'] == True].sort_values(dict_labels['tnz_datetime'])
            
            #making the folium map
            #folium_map = create_folium_map([df_begin[dict_labels['tnz_lat']].mean(),
            #                                   df_begin[dict_labels['tnz_lon']].mean()])
            
            #plot cvp in map

            df_ = df_event[df_event[dict_labels['event_id']] == event_id]
            
            folium_map = add_event_folium(df_, 
            dict_plot=dict_plot, 
            dict_labels=dict_labels, 
            slice_tags=slice_event_show)
        
            #keep only the first tnz_point nearby to cvp for each tnz
            df_begin.drop_duplicates(subset=[dict_labels['tnz_id'],'nearby'], inplace=True)
            
            #for each tnz nearby to cvp
            tnzs = []
            for time_tnz, id_tnz in zip(df_begin[dict_labels['tnz_datetime']], 
                                            df_begin[dict_labels['tnz_id']]):
                
                #making the time window for tnz
                start_time = pd.to_datetime(time_tnz - delta_tnz)
                end_time = pd.to_datetime(time_tnz + delta_tnz)

                #building the df for one id
                df_id = df_tnz[df_tnz[dict_labels['tnz_id']] == id_tnz]

                #filtering df_id for time window
                df_temp = pd.DataFrame(trajutils.filter_by_datetime(df_id,
                                         dic_labels = dict_labels,
                                         startDatetime = start_time,
                                         endDatetime=end_time))
                
                tnzs.append(df_temp)
                #add to folium map created
                add_traj_folium(df_temp, 
                          dict_labels=dict_labels, 
                          dict_plot=dict_plot,
                        folium_map=folium_map,
                        slice_tags=slice_tnz_show, 
                        sort=True)
            
            #add to folium maps list: (id cvp, folium map, quantity of tnz in map, df)  
            folium_maps.append( (folium_map, pd.concat(tnzs)) )

    return folium_maps

def show_traj_id_with_event(df_tnz, 
                        window_time_tnz,
                        df_event,
                       window_time_event,
                       radius,
                       tnz_id,
                      dict_plot = dict_plot,
                      dict_labels = dict_labels,
                    slice_event_show = None, 
                    slice_tnz_show = None):
    
    df_id = df_tnz[df_tnz[dict_labels['tnz_id']] == tnz_id]

    return show_trajs_with_event(df_id, 
                        window_time_tnz,
                        df_event,
                       window_time_event,
                       radius,
                      dict_plot = dict_plot,
                      dict_labels = dict_labels,
                    slice_event_show = slice_event_show, 
                    slice_tnz_show = slice_tnz_show)
"""  
def show_traj_with_poi_by_radius(df_tnz, 
                    df_poi,
                    radius,
                    folium_map,   
                    dict_plot = dict_plot,
                    dict_labels = dict_labels,
                    slice_poi_show = None):

    df_filted = df_tnz.drop_duplicates(subset=[dict_labels['tnz_lat'],dict_labels['tnz_lon']])

    lat_arr = np.zeros(df_poi.shape[0])
    
    lon_arr = np.zeros(df_poi.shape[0])
    
    for lat, lon in zip(df_filted[dict_labels['tnz_lat']], df_filted[dict_labels['tnz_lon']]):
        
        #length of df_temp
        #len_df_temp = df_filted.shape[0]
        
        #using the util part of the array for haversine function
        lat_arr[:] = lat
        lon_arr[:] = lon
        
        #building distances to cvp column
        df_poi['distances'] = trajutils.haversine(lat_arr,
                                         lon_arr, 
                                         df_poi[dict_labels['poi_lat']],
                                         df_poi[dict_labels['poi_lon']])
        
        df_poi['nerby'] = df_poi['distances'].map(lambda x: x<= radius)
        
        add_poi_folium(df_poi[df_poi['nerby'] == True],  
        dict_plot=dict_plot, 
        dict_labels=dict_labels, 
        folium_map=folium_map, 
        slice_tags=slice_poi_show)
   
    return folium_map

def show_traj_with_poi_and_event_by_radius(df_event,
                        window_time_event,
                        radius_event,
                        df_tnz,
                        window_time_tnz,
                        df_poi,
                        radius_poi,
                        dict_plot = dict_plot,
                        dict_labels = dict_labels,
                        slice_event_show = None,
                        slice_poi_show = None, 
                        slice_tnz_show=None):
    
    list_maps = show_trajs_with_event(df_tnz, window_time_tnz,
                                        df_event, window_time_event, radius_event,
                                        dict_plot=dict_plot, 
                                        dict_labels=dict_labels, 
                                        slice_event_show=slice_event_show, 
                                        slice_tnz_show=slice_tnz_show)
    
    for i in range(len(list_maps)):

        map_, df_traj = list_maps[i]

        add_poi_folium(df_traj, 
        df_poi, 
        radius_poi, 
        map_,
        dict_plot = dict_plot,
        dict_labels = dict_labels,
        slice_poi_show = slice_poi_show)

    return list_maps
"""
def create_folium_map(default_location, default_zoom_start=12, tiles='OpenStreetMap'):
    """
    Generate a folium map.
    Parameters
    ----------
    default_location : tuple.
        Represents coordinates lat, lon which will be the center of the map.
    default_zoom_start : int, optional, default 12.
        Represents the zoom which will be the center of the map.
    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map.
    """
        
    base_map = folium.Map(location=default_location,
                          control_scale=True, 
                          zoom_start=default_zoom_start,
                          tiles=tiles)
    return base_map
                                 #trajutils.dic_labels['id']
def show_grid_polygons(df_, id_, label_id =constants.TRAJ_ID, label_polygon='polygon', figsize=(10,10)):   
    fig = plt.figure(figsize=figsize)
    
    #filter dataframe by id
    df_ = df_[ df_[label_id] == id_]
    
    xs_start, ys_start = df_.iloc[0][label_polygon].exterior.xy
    #xs_end, ys_end = df_.iloc[1][label_polygon].exterior.xy
    
    plt.plot(ys_start,xs_start, 'bo', markersize=20)             # start tnz_point
    #plt.plot(ys_end, xs_end, 'rX', markersize=20)           # end tnz_point
   

    for idx in range(df_.shape[0]):
        xs, ys = df_[label_polygon].iloc[idx].exterior.xy
        plt.plot(ys,xs, 'g', linewidth=2, markersize=5) 

    return df_, fig
    
#https://deparkes.co.uk/2016/06/10/folium-map-tiles/
def folium_plot_bbox(bbox_tuple, tiles='OpenStreetMap', color='red', save=False, filename='bbox_.html'):
    m = folium.Map(tiles=tiles) # tiles = OpenStreetMap, mapquestopen, MapQuest Open Aerial, Mapbox Bright, Mapbox Control Room, stamenterrain, stamentoner, stamenwatercolor, rra, cartodbdark_matter
    m.fit_bounds([ [bbox_tuple[0], bbox_tuple[1]], [bbox_tuple[2], bbox_tuple[3]] ])
    tnz_points_ = [ (bbox_tuple[0], bbox_tuple[1]), (bbox_tuple[0], bbox_tuple[3]), 
                (bbox_tuple[2], bbox_tuple[3]), (bbox_tuple[2], bbox_tuple[1]),
                (bbox_tuple[0], bbox_tuple[1]) ]
    polygon = folium.PolyLine(tnz_points_, weight=3, color=color)
    polygon.add_to(m)
    
    if save == True:
        m.save(filename)
    
    return m, polygon

def generateBaseMap(default_location, default_zoom_start=12):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map
# Already on pymove.visualization
def heatmap(df_, n_rows, lat_origin=None, lon_origin=None, zoom_start=12, radius = 8, max_zoom = 13, base_map=None, save_as_html=False, filename='heatmap.html'):
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = df_.loc[0]['lat']
            lon_origin = df_.loc[0]['lon']
        base_map = generateBaseMap(default_location=[lat_origin, lon_origin], default_zoom_start=zoom_start)

    COUNT = 'count'
    df_[COUNT] = 1
    HeatMap(data=df_.loc[:n_rows, ['lat', 'lon', COUNT]].groupby(['lat', 'lon']).sum().reset_index().values.tolist(),
        radius=radius, max_zoom=max_zoom).add_to(base_map)
    df_.drop(columns=[COUNT], inplace=True)
    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map

def heatmap_with_time(df_, n_rows, lat_origin=None, lon_origin=None, zoom_start=12, radius = 5, min_opacity = 0.5, max_opacity = 0.8, base_map=None, save_as_html=False,
                      filename='heatmap_with_time.html'):
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = df_.loc[0]['lat']
            lon_origin = df_.loc[0]['lon']
        base_map = generateBaseMap(default_location=[lat_origin, lon_origin], default_zoom_start=zoom_start)
    COUNT = 'count'
    df_['hour'] = df_.datetime.apply(lambda x: x.hour)
    df_[COUNT] = 1
    df_hour_list = []
    for hour in df_.hour.sort_values().unique():
        df_hour_list.append(df_.loc[df_.hour == hour, ['lat', 'lon', COUNT]].groupby(
            ['lat', 'lon']).sum().reset_index().values.tolist())

    HeatMapWithTime(df_hour_list[:n_rows], radius=radius, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'},
                    min_opacity=min_opacity, max_opacity=max_opacity, use_local_extrema=True).add_to(base_map)

    df_.drop(columns=[COUNT, 'hour'], inplace=True)
    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map
    
def create_geojson_features_line(df_, label_lat='lat',label_lon='lon',label_datetime='datetime'):
    print('> Creating GeoJSON features...')
    features = []
    
    row_iterator = df_.iterrows()
    _, last = next(row_iterator)
    colums = df_.columns

    for i, row in tqdm(row_iterator, total=df_.shape[0]-1):
        last_time = last[label_datetime].strftime("%Y-%m-%dT%H:%M:%S")
        next_time = row[label_datetime].strftime("%Y-%m-%dT%H:%M:%S")
        
        popup_list = [i+': '+str(last[i]) for i in colums]
        popup1 = '<br>'.join(popup_list)
        #= 'id: '+last['id']+'<br>timestamp: '+last_time+'<br>lat: '+str(row['lat'])+'<br>lon: '+str(row['lon'])# +'<br>activity: '+ row['y_unique']
        #popup2 = 'id: '+row['id']+'<br>timestamp: '+next_time+'<br>lat: '+str(row['lat'])+'<br>lon: '+str(row['lon'])
        
        
        feature = {
            'type': 'Feature',
            'geometry': {
                'type':'LineString', 
                'coordinates':[
                    [last['lon'], last['lat']],
                    [row['lon'], row['lat']]
                ]

            },
            'properties': {
                'times': [last_time, next_time],
                'popup':popup1,
                'style': {'color' : 'red',
                'icon': 'circle',
                'iconstyle':{
                    'color': 'red',
                    'weight':4
                    #'radius': row['Numerical_data']*40
                    }
                }
            }
        }
        _, last = i, row
            
        features.append(feature)

    return features

def plot_traj_timestamp_geo_json(df_, label_datetime='datetime', label_lat='lat',label_lon='lon', tiles='cartodbpositron'):
    features = create_geojson_features_line(df_, label_datetime)
    print('creating folium map')
    map_ = create_folium_map(default_location=[df_[label_lat].mean(), df_[label_lon].mean()], tiles=tiles)
    print('Genering timestamp map')
    plugins.TimestampedGeoJson({
   'type': 'FeatureCollection',
   'features': features,
    }, period='PT1M', add_last_point=True).add_to(map_)
    return map_

def plot_train_val_test_datasets(y_train, y_val, y_test, labels_y=None, save_fig=True, path_fig='distribuition_datasets.png', fontsize_title='12'):
    fig, ax = plt.subplots(1,3,figsize=(20,5))
    seq_plot = [y_train,y_val,y_test]
    titles=['Train set', 'Validation set', 'Test Set']
    sns.set(style='white', palette='deep', font='sans-serif',  color_codes=True, font_scale=1.2, rc={"lines.linewidth": 1.0}) 

    for i_ax, dataset in enumerate(seq_plot):
        size_vetor = len(dataset)
        sns.countplot(y=dataset, ax=ax[i_ax])
        #plt.xlabel('records')
        ax[i_ax].set_xlabel('{} records'.format(size_vetor), fontsize=14)
        ax[i_ax].set_xlim(0, len(dataset))
        ax[i_ax].set_title(titles[i_ax], weight='bold').set_fontsize(fontsize_title)
        for i, p in enumerate(ax[i_ax].patches):
            if labels_y == None:
                percentage = ' {:.1f}% '.format(100 * p.get_width()/size_vetor)
            else:
                percentage = ' {:.1f}%  - {}'.format(100 * p.get_width()/size_vetor, labels_y[i])
            x = p.get_x() + p.get_width() + 0.02
            y = p.get_y() + p.get_height()/2
            ax[i_ax].annotate(percentage, (x, y))
    if save_fig == True:
        fig.savefig(path_fig)
