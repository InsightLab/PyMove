from branca.element import MacroElement, Template
from matplotlib.cm import get_cmap as _get_cmap
from matplotlib.colors import rgb2hex
from numpy.random import randint

from pymove.utils.constants import COLORS, LATITUDE, LONGITUDE, TRAJ_ID


def add_map_legend(m, title, items):
    """
    Adds a legend for a folium map.

    Parameters
    ----------
    m : folium.map.
        Represents a folium map.
    title : string.
        Represents the title of the legend.
    items : list of tuple.
        Represents the color and name of the legend items.

    References
    ----------
    https://github.com/python-visualization/folium/issues/528#issuecomment-421445303

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

    m.get_root().add_child(macro)


def generate_color():
    """
    Generates a random color.

    Returns
    -------
    Random HEX color
    """
    return COLORS[randint(0, len(COLORS))]


def rgb(rgb_colors):
    """
    Return a tuple of integers, as used in AWT/Java plots.

    Parameters
    ----------
    rgb_colors : list
        Represents a list with three positions that correspond to the percentage red,
        green and blue colors.

    Returns
    -------
    tuple
        Represents a tuple of integers that correspond the colors values.

    Examples
    --------
    >>> from pymove.visualization.visualization import rgb
    >>> rgb([0.6,0.2,0.2])
        (51, 51, 153)
    """
    blue = rgb_colors[0]
    red = rgb_colors[1]
    green = rgb_colors[2]
    return int(red * 255), int(green * 255), int(blue * 255)


def hex_rgb(rgb_colors):
    """
    Return a hex string, as used in Tk plots.

    Parameters
    ----------
    rgb_colors : list
        Represents a list with three positions that correspond to the percentage red,
        green and blue colors.

    Returns
    -------
    String
        Represents a color in hexadecimal format.

    Examples
    --------
    >>> from pymove.visualization.visualization import hex_rgb
    >>> hex_rgb([0.6,0.2,0.2])
    '#333399'
    """
    return '#%02X%02X%02X' % rgb(rgb_colors)


def cmap_hex_color(cmap, i):
    """
    Convert a Colormap to hex color.

    Parameters
    ----------
    cmap : matplotlib.colors.ListedColormap
        Represents the Colormap.
    i : int
        List color index.

    Returns
    -------
    String
        Represents corresponding hex string.
    """
    return rgb2hex(cmap(i))


def get_cmap(cmap):
    return _get_cmap(cmap)


def save_wkt(move_data, filename, label_id=TRAJ_ID):
    """
    Save a visualization in a map in a new file .wkt.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    filename : String
        Represents the filename.
    label_id : String
        Represents column name of trajectory id.

    """

    str_ = '%s;linestring\n' % label_id
    ids = move_data[label_id].unique()
    for id_ in ids:
        move_df = move_data[move_data[label_id] == id_]
        curr_str = '%s;LINESTRING(' % id_
        curr_str += ','.join(
            '%s %s' % (x[0], x[1])
            for x in move_df[[LONGITUDE, LATITUDE]].values
        )
        curr_str += ')\n'
        str_ += curr_str
    with open(filename, 'w') as f:
        f.write(str_)
