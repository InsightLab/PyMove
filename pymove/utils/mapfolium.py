from branca.element import MacroElement, Template


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

    item = "<li><span style='background:%srs;'></span>%srs</li>"
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
