"""
Visualization auxiliary operations.

add_map_legend,
generate_color,
rgb,
hex_rgb,
cmap_hex_color,
get_cmap,
save_wkt

"""

from typing import List, Optional, Text, Tuple

from branca.element import MacroElement, Template
from folium import Map
from matplotlib.cm import get_cmap as _get_cmap
from matplotlib.colors import Colormap, ListedColormap, rgb2hex
from numpy.random import randint
from pandas import DataFrame

from pymove.utils.constants import COLORS, LATITUDE, LONGITUDE, TRAJ_ID


def add_map_legend(m: Map, title: Text, items: List[Tuple]):
    """
    Adds a legend for a folium map.

    Parameters
    ----------
    m : Map
        Represents a folium map.
    title : str
        Represents the title of the legend
    items : list of tuple
        Represents the color and name of the legend items

    References
    ----------
    https://github.com/python-visualization/folium/issues/528#issuecomment-421445303

    Examples
    --------
    >>> import folium
    >>> from pymove.utils.visual import add_map_legend
    >>> df
              lat          lon              datetime  id
    0   39.984094   116.319236   2008-10-23 05:53:05   1
    1   39.984198   116.319322   2008-10-23 05:53:06   1
    2   39.984224   116.319402   2008-10-23 05:53:11   1
    3   39.984211   116.319389   2008-10-23 05:53:16   2
    4   39.984217   116.319422   2008-10-23 05:53:21   2
    >>> m = folium.Map(location=[df.lat.median(), df.lon.median()])
    >>> folium.PolyLine(mdf[['lat', 'lon']], color='red').add_to(m)
    >>> pm.visual.add_map_legend(m, 'Color by ID', [(1, 'red')])
    >>> m.get_root().to_dict()
    {
        "name": "Figure",
        "id": "1d32230cd6c54b19b35ceaa864e61168",
        "children": {
            "map_6f1abc8eacee41e8aa9d163e6bbb295f": {
                "name": "Map",
                "id": "6f1abc8eacee41e8aa9d163e6bbb295f",
                "children": {
                    "openstreetmap": {
                        "name": "TileLayer",
                        "id": "f58c3659fea348cb828775f223e1e6a4",
                        "children": {}
                    },
                    "poly_line_75023fd7df01475ea5e5606ddd7f4dd2": {
                        "name": "PolyLine",
                        "id": "75023fd7df01475ea5e5606ddd7f4dd2",
                        "children": {}
                    }
                }
            },
            "map_legend": {  # legend element
                "name": "MacroElement",
                "id": "72911b4418a94358ba8790aab93573d1",
                "children": {}
            }
        },
        "header": {
            "name": "Element",
            "id": "e46930fc4152431090b112424b5beb6a",
            "children": {
                "meta_http": {
                    "name": "Element",
                    "id": "868e20baf5744e82baf8f13a06849ecc",
                    "children": {}
                }
            }
        },
        "html": {
            "name": "Element",
            "id": "9c4da9e0aac349f594e2d23298bac171",
            "children": {}
        },
        "script": {
            "name": "Element",
            "id": "d092078607c04076bf58bd4593fa1684",
            "children": {}
        }
    }
    """
    item = "<li><span style='background:%s;'></span>%s</li>"
    list_items = '\n'.join([item % (c, n) for (n, c) in items])
    template = """
    {{% macro html(this, kwargs) %}}

    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link rel="stylesheet"
        href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

      <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
      <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>

      <script>
      $( function() {{
        $( "#maplegend" ).draggable({{
                        start: function (event, ui) {{
                            $(this).css({{
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            }});
                        }}
                    }});
    }});

      </script>
    </head>
    <body>

    <div id='maplegend' class='maplegend'
        style='position: absolute; z-index:9999; border:2px solid grey;
        background-color:rgba(255, 255, 255, 0.8); border-radius:6px;
        padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>

    <div class='legend-title'> {} </div>
    <div class='legend-scale'>
      <ul class='legend-labels'>
        {}
      </ul>
    </div>
    </div>

    </body>
    </html>

    <style type='text/css'>
      .maplegend .legend-title {{
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }}
      .maplegend .legend-scale ul {{
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }}
      .maplegend .legend-scale ul li {{
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }}
      .maplegend ul.legend-labels li span {{
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }}
      .maplegend .legend-source {{
        font-size: 80%;
        color: #777;
        clear: both;
        }}
      .maplegend a {{
        color: #777;
        }}
    </style>
    {{% endmacro %}}""".format(
        title, list_items
    )

    macro = MacroElement()
    macro._template = Template(template)

    m.get_root().add_child(macro, name='map_legend')


def generate_color() -> Text:
    """
    Generates a random color.

    Returns
    -------
        Random HEX color

    Examples
    --------
    >>> from pymove.utils.visual import generate_color
    >>> print(generate_color(), type(generate_color()))
    '#E0FFFF' <class 'str'>
    >>> print(generate_color(), type(generate_color()))
    '#808000' <class 'str'>
    """
    return COLORS[randint(0, len(COLORS))]


def rgb(rgb_colors: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """
    Return a tuple of integers, as used in AWT/Java plots.

    Parameters
    ----------
    rgb_colors : tuple
        Represents a tuple with three positions that correspond to the percentage red,
        green and blue colors.

    Returns
    -------
    tuple
        Represents a tuple of integers that correspond the colors values.

    Examples
    --------
    >>> from pymove.utils.visual import rgb
    >>> print(rgb((0.1, 0.2, 0.7)), type(rgb((0.1, 0.2, 0.7))))
    (51, 178, 25) <class 'tuple'>
    >>> print(rgb((0.5, 0.4, 0.1)), type(rgb((0.5, 0.4, 0.1))))
    (102, 25, 127) <class 'tuple'>
    """
    blue = rgb_colors[0]
    red = rgb_colors[1]
    green = rgb_colors[2]
    return int(red * 255), int(green * 255), int(blue * 255)


def hex_rgb(rgb_colors: Tuple[float, float, float]) -> Text:
    """
    Return a hex str, as used in Tk plots.

    Parameters
    ----------
    rgb_colors : tuple
        Represents a tuple with three positions that correspond to the percentage red,
        green and blue colors

    Returns
    -------
    str
        Represents a color in hexadecimal format

    Examples
    --------
    >>> from pymove.utils.visual import hex_rgb
    >>> print(hex_rgb((0.1, 0.2, 0.7)), type(hex_rgb((0.1, 0.2, 0.7))))
    '#33B219' <class 'str'>
    >>> print(hex_rgb((0.5, 0.4, 0.1)), type(hex_rgb((0.5, 0.4, 0.1))))
    '#66197F' <class 'str'>
    """
    return '#%02X%02X%02X' % rgb(rgb_colors)


def cmap_hex_color(cmap: ListedColormap, i: int) -> Text:
    """
    Convert a Colormap to hex color.

    Parameters
    ----------
    cmap : ListedColormap
        Represents the Colormap
    i : int
        List color index

    Returns
    -------
    str
        Represents corresponding hex str

    Examples
    --------
    >>> from pymove.utils.visual import  cmap_hex_color
    >>> import matplotlib.pyplot as plt
    >>> jet = plt.get_cmap('jet')  # This comand generates a Linear Segmented Colormap
    >>> print(cmap_hex_color(jet, 0))
    '#000080'
    >>> print(cmap_hex_color(jet, 1))
    '#000084'
    """
    return rgb2hex(cmap(i))


def get_cmap(cmap: Text) -> Colormap:
    """
    Returns a matplotlib colormap instance.

    Parameters
    ----------
    cmap : str
        name of the colormar

    Returns
    -------
    Colormap
        matplotlib colormap

    Examples
    --------
    >>> from pymove.utils.visual import  get_cmap
    >>> print(get_cmap('Greys')
    <matplotlib.colors.LinearSegmentedColormap object at 0x7f743fc04bb0>
    """
    return _get_cmap(cmap)


def save_wkt(
    move_data: DataFrame, filename: Text, label_id: Optional[Text] = TRAJ_ID
):
    """
    Save a visualization in a map in a new file .wkt.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data
    filename : str
        Represents the filename
    label_id : str
        Represents column name of trajectory id

    Returns
    -------
        File: A file.wkt that contains geometric points that build a map visualization

    Examples
    --------
    >>> from pymove.utils.visual import save_wkt
    >>> df.head()
              lat          lon              datetime  id
    0   39.984094   116.319236   2008-10-23 05:53:05   1
    1   39.984198   116.319322   2008-10-23 05:53:06   1
    2   39.984224   116.319402   2008-10-23 05:53:11   1
    3   39.984211   116.319389   2008-10-23 05:53:16   2
    4   39.984217   116.319422   2008-10-23 05:53:21   2
    >>> save_wkt(df, 'test.wkt', 'id')
    >>> with open('test.wtk') as f:
    >>>     print(f.read())
    'id;linestring'
    '1;LINESTRING(116.319236 39.984094,116.319322 39.984198,116.319402 39.984224)'
    '2;LINESTRING(116.319389 39.984211,116.319422 39.984217)'
    """
    wtk = '%s;linestring\n' % label_id
    ids = move_data[label_id].unique()
    for id_ in ids:
        move_df = move_data[move_data[label_id] == id_]
        curr_str = '%s;LINESTRING(' % id_
        curr_str += ','.join(
            '%s %s' % (x[0], x[1])
            for x in move_df[[LONGITUDE, LATITUDE]].values
        )
        curr_str += ')\n'
        wtk += curr_str
    with open(filename, 'w') as f:
        f.write(wtk)
