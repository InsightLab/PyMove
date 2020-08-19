LATITUDE = 'lat'
LONGITUDE = 'lon'
DATETIME = 'datetime'
TRAJ_ID = 'id'
TID = 'tid'
UID = 'user_id'

GEOHASH = 'geohash'
BIN_GEOHASH = 'bin_geohash'
LATITUDE_DECODE = 'lat_decode'
LONGITUDE_DECODE = 'lon_decode'

BASE_32 = ['0', '1', '2', '3', '4', '5',
           '6', '7', '8', '9', 'a', 'b',
           'c', 'd', 'e', 'f', 'g', 'h',
           'i', 'j', 'k', 'l', 'm', 'n',
           'o', 'p', 'q', 'r', 's', 't',
           'u', 'v', 'w', 'x', 'y', 'z']

POI = 'poi'
ID_POI = 'id_poi'
DIST_POI = 'dist_poi'
TYPE_POI = 'type_poi'
NAME_POI = 'name_poi'

EVENT_ID = 'event_id'
EVENT_TYPE = 'event_type'
DIST_EVENT = 'dist_event'

CITY = 'city'
HOME = 'home'
ADDRESS = 'formatted_address'
DIST_HOME = 'dist_home'

GEOMETRY = 'geometry'
VIOLATING = 'violating'

HOUR = 'hour'
HOUR_SIN = 'hour_sin'
HOUR_COS = 'hour_cos'
DATE = 'date'
DAY = 'day'
WEEK_END = 'weekend'
WEEK_DAYS = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday'
]
PERIOD = 'period'
DAY_PERIODS = [
    'Early morning',
    'Morning',
    'Afternoon',
    'Evening'
]
TIME_SLOT = 'time_slot'
TYPE_DASK = 'dask'
TYPE_PANDAS = 'pandas'

DIST_TO_PREV = 'dist_to_prev'
DIST_TO_NEXT = 'dist_to_next'
DIST_PREV_TO_NEXT = 'dist_prev_to_next'
TIME_TO_PREV = 'time_to_prev'
TIME_TO_NEXT = 'time_to_next'
TIME_PREV_TO_NEXT = 'time_prev_to_next'
SPEED_TO_PREV = 'speed_to_prev'
SPEED_TO_NEXT = 'speed_to_next'
SPEED_PREV_TO_NEXT = 'speed_prev_to_next'
INDEX_GRID_LAT = 'index_grid_lat'
INDEX_GRID_LON = 'index_grid_lon'
INDEX_GRID = 'index_grid'
TID_PART = 'tid_part'
TID_SPEED = 'tid_speed'
TID_TIME = 'tid_time'
TID_DIST = 'tid_dist'
SITUATION = 'situation'
SEGMENT_STOP = 'segment_stop'
STOP = 'stop'
MOVE = 'move'
POLYGON = 'polygon'

LAT_MEAN = 'lat_mean'
LON_MEAN = 'lon_mean'

OUT_BBOX = 'out_bbox'
DEACTIVATED = 'deactivated_signal'
JUMP = 'gps_jump'
BLOCK = 'block_signal'
SHORT = 'short_traj'

TB = 'TB'
GB = 'GB'
MB = 'MG'
KB = 'KB'
B = 'bytes'
COUNT = 'count'

COLORS = {
    0: '#000000',  # black
    1: '#808080',  # gray
    2: '#D3D3D3',  # lightgray
    3: '#FFFFFF',  # white
    4: '#800000',  # red maroon
    5: '#B22222',  # red fire brick
    6: '#DC143C',  # red crimson
    7: '#FF7F50',  # coral
    8: '#FF8C00',  # dark orange
    9: '#FFD700',  # gold
    10: '#FFFF00',  # yellow
    11: '#ADFF2F',  # green yellow
    12: '#9ACD32',  # yellow green
    13: '#6B8E23',  # olive drab
    14: '#808000',  # olive
    15: '#00FF00',  # lime
    16: '#008000',  # green
    17: '#3CB371',  # medium sea green
    18: '#00FF7F',  # spring green
    19: '#E0FFFF',  # pale turquoise
    20: '#00FFFF',  # aqua/cyan
    21: '#87CEFA',  # light sky blue
    22: '#00BFFF',  # deep sky blue
    23: '#1E90FF',  # dodger blue
    24: '#0000FF',  # blue
    25: '#6A5ACD',  # slate blue
    26: '#4B0082',  # indigo
    27: '#FF00FF',  # fuchsia / magenta
    28: '#EE82EE',  # violet
    29: '#8A2BE2',  # blue violet
    30: '#C71585',  # medium violet red
    31: '#FF1493',  # deep pink
    32: '#FFB6C1',  # light pink
    33: '#ffcc33',  # sunglow
    34: '#6699cc'   # blue gray
}

TILES = [
    'CartoDB positron',
    'CartoDB dark_matter',
    'Stamen Terrain',
    'Stamen Toner',
    'Stamen Watercolor',
    'OpenStreetMap'
]

USER_POINT = 'orange'
LINE_COLOR = 'blue'
POI_POINT = 'red'
EVENT_POINT = 'purple'
