import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymove.core import MoveDataFrame
from pymove.utils.utils import format_labels

LATITUDE = 'lat'
LONGITUDE = 'lon'
DATETIME = 'datetime'
TRAJ_ID = 'id'

#TODO: tirar o data do format_labels
#TODO: mover constantes para um arquivo
class PandasMoveDataFrame(pd.DataFrame): # dask sua estrutura de dados
    def __init__(self, data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id = TRAJ_ID):
        # formatar os labels que foram passados pro que usado no pymove -> format_labels
        # renomeia as colunas do dado passado pelo novo dict
        # cria o dataframe
    
        mapping_columns = format_labels(data, traj_id, latitude, longitude, datetime)
    
        tdf = data.rename(columns=mapping_columns)
        columns = tdf.columns
        
        if self._has_columns(tdf):
            self._validate_move_data_frame(tdf)
            super(PandasMoveDataFrame, self).__init__(tdf)
            
            
    def _has_columns(self, data):
        if(LATITUDE in data and LONGITUDE in data and DATETIME in data):
            return True
        return False

    def _validate_move_data_frame(self, data):
        # chama a função de validação   
            # deverá verificar se tem as colunas e os tipos
        try:
            if(data.dtypes.lat != 'float32'):
                data.lat.astype('float32') 
            if(data.dtypes.lon != 'float32'):
                data.lon.astype('float32') 
            if(data.dtypes.datetime != 'datetime64[ns]'):
                data.lon.astype('datetime64[ns]') 
        except AttributeError as erro:
            print(erro)

    @property
    def lat(self):
        if LATITUDE not in self:
            raise AttributeError("The TrajDataFrame does not contain the column '%s.'" % LATITUDE)
        return self[LATITUDE]

    @property
    def lng(self):
        if LONGITUDE not in self:
            raise AttributeError("The TrajDataFrame does not contain the column '%s.'"%LONGITUDE)
        return self[LONGITUDE]

    
    @property
    def datetime(self):
        if DATETIME not in self:
            raise AttributeError("The TrajDataFrame does not contain the column '%s.'"%DATETIME)
        return self[DATETIME]

    def head(self, num = 5):
        self.head(num)
        
   
    # def read_file(self, filename):
    #     return self.read_file(filename)

  
    # def get_user_number(self):
    #     pass

  
    # def time_interval():
    #     pass  


    # def to_csv():
    #     pass


    # def to_numpy():
    #     pass          