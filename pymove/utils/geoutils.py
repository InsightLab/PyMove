import geohash2 as gh
import numpy as np

from pymove.utils.constants import (
    BASE_32,
    BIN_GEOHASH,
    COLORS,
    GEOHASH,
    LATITUDE,
    LATITUDE_DECODE,
    LONGITUDE,
    LONGITUDE_DECODE,
)
from pymove.utils.log import progress_bar

BINARY = [
    np.asarray(
        list('{0:05b}'.format(x)), dtype=int
    ) for x in range(0, len(BASE_32))
]


BASE_32_TO_BIN = dict(zip(BASE_32, BINARY))


def v_color(ob):
    """
    Returns '#ffcc33' if object crosses,
    otherwise it returns '#6699cc'.

    Parameters
    ----------
    ob : geometry object
        Any geometric object

    Return
    ------
    string
        Geometric object color
     """
    return COLORS[ob.is_simple + 33]


def plot_coords(ax, ob, color='r'):
    """
    Plot the coordinates of each point of the object in a 2D chart.

    Parameters
    ----------
    ax : axes.Axes
        Single axes object
    ob : geometry object
        Any geometric object
    color : string, optional, default 'r'
        Sets the geometric object color.
    """
    x, y = ob.xy
    ax.plot(x, y, 'o', color=color, zorder=1)


def plot_bounds(ax, ob, color='b'):
    """
    Plot the limites of geometric object.

    Parameters
    ----------
    ax : axes.Axes
        Single axes object
    ob : LineString, MultiLineString
        Geometric object formed by lines.
    color : string, optional, default 'b'
        Sets the geometric object color.
    """
    x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, '-', color=color, zorder=1)


def plot_line(
    ax, ob, color='r', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2
):
    """
    Plot a LineString

    Parameters
    ----------
    ax : axes.Axes
        Single axes object
    ob : LineString
        Sequence of points.
    color : string, optional, default 'r'
        Sets the line color.
    alpha : number, optional, default 0.7
        Defines the opacity of the line.
    linewidth : number, optional, default 3
        Defines the line thickness.
    solid_capstyle : string, optional, default 'round'
        Defines the style of the ends of the line.
    zorder : number, optional, default 2
        Determines the default drawing order for the axes.
    """
    x, y = ob.xy
    ax.plot(
        x, y, color=color, alpha=alpha, linewidth=linewidth,
        solid_capstyle=solid_capstyle, zorder=2
    )


def _encode(lat, lon, precision=15):
    """
    Encodes latitude/longitude to geohash, either to specified
    precision or to automatically evaluated precision.

    Parameters
    ----------
    lat : number
        Latitude in degrees.
    lon : number
        Longitude in degrees.
    precision : number, optional, default 15
        Number of characters in resulting geohash.

    Return
    ------
    string
        Geohash of supplied latitude/longitude.
    """
    return gh.encode(lat, lon, precision)


def _decode(geohash):
    """
    Decode geohash to latitude/longitude (location is approximate
    centre of geohash cell, to reasonable precision).

    Parameters
    ----------
    geohash : string
        Geohash string to be converted to latitude/longitude.

    Return
    ------
    (lat : number, lon : number)
        Geohashed location.
    """
    return gh.decode(geohash)


def _bin_geohash(lat, lon, precision=15):
    """
    Transforms a point's geohash into a binary array.

    Parameters
    ----------
    lat : number
        Latitude in degrees.
    lon : number
        Longitude in degrees.
    precision : number, optional, default 15
        Number of characters in resulting geohash.

    Return
    ------
    array
        Returns a binary geohash array
    """
    hashed = _encode(lat, lon, precision)
    return np.concatenate([BASE_32_TO_BIN[x] for x in hashed])


def _reset_and_create_arrays_none(df_, reset_index=True):
    """
    Reset the df index and create arrays of none values.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    reset_index : boolean, optional, default True
        Condition to reset the df index.

    Return
    ------
    arrays
        Returns arrays of none values, of the size of the df.
    """
    if reset_index:
        df_.reset_index(drop=True, inplace=True)

    latitudes = np.full(
        df_.shape[0], None, dtype=np.float64
    )

    longitudes = np.full(
        df_.shape[0], None, dtype=np.float64
    )

    geohash = np.full(
        df_.shape[0], None, dtype='object_'
    )

    bin_geohash = np.full(
        df_.shape[0], None, dtype=np.ndarray
    )

    return latitudes, longitudes, geohash, bin_geohash


def create_geohash_df(df_, precision=15):
    """
    Create geohash from geographic coordinates
    and integrate with df.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    precision : number, optional, default 15
        Number of characters in resulting geohash.
    """

    try:
        _, _, geohash, _ = _reset_and_create_arrays_none(df_)

        for idx, row in progress_bar(
            df_[[LATITUDE, LONGITUDE]].iterrows(), total=df_.shape[0]
        ):
            geohash[idx] = _encode(row[LATITUDE], row[LONGITUDE], precision)

        df_[GEOHASH] = geohash
        print('\n================================================')
        print('\n========> geohash feature was created. <========')
        print('\n================================================')

    except Exception as e:
        raise e


def create_bin_geohash_df(df_, precision=15):
    """
    Create trajectory geohash binaries and integrate with df.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    precision : number, optional, default 15
        Number of characters in resulting geohash.
    """
    try:
        _, _, _, bin_geohash = _reset_and_create_arrays_none(df_)

        for idx, row in progress_bar(
            df_[[LATITUDE, LONGITUDE]].iterrows(), total=df_.shape[0]
        ):
            bin_geohash[idx] = _bin_geohash(row[LATITUDE], row[LONGITUDE], precision)

        df_[BIN_GEOHASH] = bin_geohash
        print('\n================================================')
        print('\n=====> bin_geohash features was created. <======')
        print('\n================================================')

    except Exception as e:
        raise e


def decode_geohash_to_latlon(df_, label_geohash=GEOHASH, reset_index=True):
    """
    Decode feature with hash of trajectories back to
    geographic coordinates.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    label_geohash : str, optional, default 'geohash'
        The name of the feature with hashed trajectories
    reset_index : boolean, optional, default True
        Condition to reset the df index.
    """
    try:
        if label_geohash not in df_:
            raise ValueError('feature {} not in df'.format(label_geohash))

        lat, lon, _, _ = _reset_and_create_arrays_none(df_, reset_index=reset_index)

        for idx, row in progress_bar(df_[[label_geohash]].iterrows(), total=df_.shape[0]):
            lat_lon = _decode(row[label_geohash])
            lat[idx] = lat_lon[0]
            lon[idx] = lat_lon[1]

        df_[LATITUDE_DECODE] = lat
        df_[LONGITUDE_DECODE] = lon
        print('\n================================================')
        print('\n==> lat and lon decode features was created. <==')
        print('\n================================================')

    except Exception as e:
        raise e
