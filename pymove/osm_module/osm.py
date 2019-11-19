# from xml.etree import ElementTree as ET
# import pandas as pd
# import numpy as np
# from time import time
# from pymove import utils as ut
#
# try:
#     import overpy as ovp
# except ImportError:
#     pass
#
#
# def getinfo_osm(edges):
#     edges_unique = np.unique(edges)
#     edges_str = ','.join(map(str, edges_unique))
#     api = ovp.Overpass()
#     return api.query('[out:json];way(id:{});out tags;'.format(edges_str))
#
# def generate_df_edges(osm_result, cols=['lanes', 'maxspeed']):
#     rows_size = len(osm_result.ways)
#     cols_size = len(cols)
#     info_edges = np.empty((rows_size, cols_size), dtype=np.float64)
#     id_edges = np.empty((rows_size,), dtype=np.int64)
#
#     for i, way in enumerate(osm_result.ways):
#         id_edges[i] = way.id
#         for j, col in enumerate(cols):
#             if col in way.tags:
#                 info_edges[i, j] = way.tags[col]
#             else:
#                 info_edges[i, j] = np.nan
#
#     df_edges = pd.DataFrame(info_edges, columns=cols)
#     df_edges['osm_edge_id'] = id_edges
#     return df_edges[ list(df_edges.columns[-1:]) + list(df_edges.columns[:-1]) ]
#
# def get_way_tags_values(tree, way_id, tag_labels):
#     """
#     this is very slow.
#     e.g.
#     from xml.etree import ElementTree as ET
#     tree = ET.parse('/cloud/regis/taxi_simples/fortaleza.osm.xml')
#     get_way_tags_values(tree, 191585228, ['lanes', 'maxspeed'])
#     """
#     tags_ = tree.findall(".//way[@id='{}']/tag".format(way_id))
#     result = {}
#     for tag in tags_:
#         tag_label = tag.get('k')
#         value = tag.get('v')
#         if tag_label in tag_labels and value is not None:
#             result[tag_label] = value
#     return result
#
# def add_features_from_osm(df, osm_id_label, osm_xml_file, tag_labels, default_values=None):
#     """
#     blazing fast and recommended!!!
#     """
#     print('add_features_from_osm')
#     print('generating new dataframe from the original one...')
#     filter_ = df[osm_id_label] >= 0
#     df_new = pd.DataFrame({osm_id_label : df.loc[filter_, osm_id_label].unique()})
#
#
#     for i, tag_label in enumerate(tag_labels):
#         if default_values is None:
#             df_new[tag_label] = np.nan
#         else:
#             df_new[tag_label] = default_values[i]
#
#     if df_new.index.name != osm_id_label:
#         df_new.set_index(osm_id_label, inplace=True)
#
#     print('loading xml file...')
#     tree = ET.parse(osm_xml_file)
#     print('finding way elements...')
#     tags = tree.findall(".//way")
#     size_all = len(tags)
#     size_processed = 0
#     osmEdgeId_values = df_new.index.values
#     start_time = time()
#     curr_perc_int = 0
#     for tag in tags:
#         way_id = int(tag.get('id'))
#         if way_id in osmEdgeId_values:
#             children_tags = tag.getchildren()
#             for ctag in children_tags:
#                 if ctag.tag == 'tag':
#                     key = ctag.get('k')
#                     if key in tag_labels:
#                         df_new.at[way_id, key] = ctag.get('v')
#
#         size_processed += 1
#         curr_perc_int, _ = ut.progress_update(size_processed, size_all, start_time, curr_perc_int, step_perc=20)
#
#     df_new.reset_index(inplace=True)
#     return df_new
