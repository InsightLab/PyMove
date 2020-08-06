from pymove import gridutils
from pymove import trajutils
import numpy as np
import pandas as pd

def apply_grid_Area_in_df(df_, bbox, cell_size):

    '''
    Aply a grid in dataframe and discard locations outside
    the bounding_box specified by parameter

    Parameters
    ----------
    df_: pandas.core.DataFrame
        Input trajectory data.

    bbox: list
        bounding box which specifies the bounding area to be applied in spacial area.  

    cell_size: int
        number that represents the size of the grid cells in meters.

    '''
    print("\n-------------------------------------------")
    print("The data will be distributed in a grid of")
    print("...limits {} and \n ...cells of size {} meters".format(bbox, cell_size))
    print("-------------------------------------------\n")

    # create a virtual grid to apply in dataframe
    dic_grid = gridutils.create_virtual_grid(cell_size=cell_size, bbox=bbox)
    # applying grid in dataframe
    trajutils.create_update_index_grid_feature(df_, dic_grid=dic_grid)

    # verify if there are locals out of bounding box
    # if there is not, do nothing and return the dataframe
    # else, remove such locals
    out_bbox = df_[(df_['index_grid_lat'] < 0) | (df_['index_grid_lon'] < 0)]['id'].unique()

    if not out_bbox:
        print("... There is no one cell outside the Grid Boundaries")

    else:
        print("-----------------------------------------")
        print("The Following Cells Were Outside The Grid Boundaries")
        print(out_bbox.tolist())
        print("-----------------------------------------")
        df_.drop(df_[df_['id'].isin(out_bbox)].index, inplace=True)

    # df_.reset_index(drop=True, inplace=True)
    return df_

# ------------------------------------------------------------------------------------------------

def create_distincts_time_slot_in_minute_from_datetime(
    df,                           
    slot_interval, 
    label_time_slot='time_slot',
    datetime='datetime'
                                                      ):

    '''
    Creates a column that represents the unique time_slot referent to datetime

    Parameters
    ----------
    df: pandas.core.DataFrame
        Input trajectory data.

    slot_interval: int
        Number that represents the time_slot minutes 

    label_time_slot: str, optional, default 'time_slot'
        name to be given to the time_slot column

    datetime: str, optional, default 'datetime'
        represents the column name datetime

    '''
    try:
        if df['datetime'].dtype == np.dtype('<M8[ns]'):

            day_in_minutes = df.datetime.dt.day * 24 * 60
            hour_in_minutes = df.datetime.dt.hour * 60

            min_day = day_in_minutes + hour_in_minutes + df.datetime.dt.minute

            df[label_time_slot] = min_day / slot_interval
            df[label_time_slot] = df[label_time_slot].astype(int)
            print('time_slot :{}/n', format(min_day / slot_interval))

        else:
            print('column {} has not a datetime in our dtype')

    except Exception as e:
        raise e

# ------------------------------------------------------------------------------------------------

def filter_min_max_people(df, min_people, max_people):

    print(df.columns)
    g = df.groupby(['index_grid', 'time_slot']).agg({'id': 'nunique'}).query(
        'id >=' + str(min_people) + ' and id <=' + str(max_people))
    # -----
    # print("Get N pessoas por time_slot - se falso então não é vazio = {}".format(g.empty))
    # print(g.head())
    if g.empty:
        return "(3º)There is no cell with this amount of people"

    else:
        g = g.reset_index()
        gp = df[(df['index_grid'].isin(g.index_grid)) & (df['time_slot'].isin(g.time_slot))]
        gp = gp.set_index(keys=['index_grid', 'time_slot'])
        # gp = gp.loc[ list(zip(g.index_grid.tolist(), g.time_slot.tolist())) ]

        gp = gp[gp.index.isin(list(zip(g.index_grid.tolist(), g.time_slot.tolist())))]
        return gp.reset_index()


def checking_that_people_stayed_together_all_time(
    filter1, 
    min_people, 
    max_people, 
    min_time
                                                 ):
    ''' ----------------------------------------------------
    # 4º agrupando por tempo de permanencia total na célula
    '''
    filter1 = filter1.groupby(['index_grid', 'time_slot', 'id']).agg({'time_to_prev': 'sum'}).sort_values(
        ['index_grid', 'time_slot'], ascending=True)

    ''' ------------------------------------------------------
    # 5º definindo tempo de permanencia desejado em segundos
    '''
    min_time_sec = min_time * 60

    ''' ----------------------------------------------------------------
    # 6º retornando apenas pessoas com o tempo de permanencia desejado
    '''
    gp_tm = filter1[filter1.time_to_prev >= min_time_sec]
    gp_tm = gp_tm.reset_index()

    # -----
    if gp_tm.empty:
        return "(6º)There is no cell with people stand for {} minutes".format(min_time)
    # -----
    else:
        return gp_tm



def filter_by_lenght_of_stay(
    df, 
    min_time, 
    min_people, 
    max_people
                            ):

    '''
    Meeting detection 
    ----------------
    Detects meetings from a trajectory DataFrame according to 
    the minimum meeting time and limits of people per meeting

    Parameters
    ----------
    df : pandas.core.DataFrame
        Input trajectory data.

    min_time : int
        The minimum meeting time in minutes. 

    min_people : int
        The minimum number of people per meeting.

    max_people : int
        The maximum number of people per meeting.

    '''

    # 1º criar coluna time_slot
    create_distincts_time_slot_in_minute_from_datetime(df, min_time, 'time_slot')

    ''' ------------------------------
    # 2º criar coluna time_to_prev
    '''
    trajutils.create_update_dist_time_speed_features(df)

    ''' ---------------------------------------------------------------------------
    # 3º retornar apenas celulas com N pessoas na mesma célula e no mesmo time_slot
    '''

    filter1 = filter_min_max_people(df, min_people, max_people)

    if(type(filter1) == str):
        return filter1

    filter2 = checking_that_people_stayed_together_all_time(filter1, min_people, max_people, min_time)

    if(type(filter2) == str):
        return filter2

    ''' --------------------------------------------------------------------------------
    # 7º agrupando as celulas e time_slot de pessoas com o tempo de permanencia desejado
    '''
    list_id_grids = filter2.index_grid.tolist()
    list_slots = filter2.time_slot.tolist()
    list_ids = filter2.id.tolist()

    concise_meet = df.set_index(keys=['index_grid', 'time_slot', 'id'])
    concise_meet = concise_meet[concise_meet.index.isin(list(zip(list_id_grids, list_slots, list_ids)))]
    concise_meet = concise_meet.reset_index()

    ''' ---------------------------------------------------------------------------
    # 8º retornando apenas células com N pessoas com tempo de permanencia desejado
    ''' 
    gp_concise_meet = concise_meet.groupby(['index_grid', 'time_slot']).agg({'id': 'nunique'}).query(
        'id >=' + str(min_people) + ' and id <=' + str(max_people))

    # -----
    # print(gp_again.head())
    if gp_concise_meet.empty:
        return "(8º) There is no cell with people together for {} minutes".format(min_time)
    # -----

    '''
    # 9º juntando time_slots
    '''
    gp_concise_meet = gp_concise_meet.reset_index()

    ''' -------------------------------------------------------------
    # 9º retornando o passo anterior no formado do dataframe original
    '''
    list_id_grids = gp_concise_meet.index_grid.tolist()
    list_slots = gp_concise_meet.time_slot.tolist()

    df = df[(df.index_grid.isin(list_id_grids)) & (df.time_slot.isin(list_slots))]
    df = df.set_index(keys=['index_grid', 'time_slot'])
    df = df[df.index.isin(list(zip(list_id_grids, list_slots )))]

    return df        

def join_meets_with_POIs(df_, df_POIs, label_id='index', reset_index='True')

    if label_id=='index' and 'index' not in df_POIs:
        df_POIs.reset_index(inplace=True)

    df_ = df_.reset_index()
    integration.join_with_POIs(df_, df_POIs,label_id, label_POI, reset_index)
    df_ = df_.set_index(keys=['index_grid', 'time_slot'])

    return df_
