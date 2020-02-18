import os
import logging
import warnings
import numpy as np
import pandas as pd
from distutils import util
# to handle str to bool conversion
from ast import literal_eval

from ..utils.custom_decorator import where_exception
from ...config.result_path_config import PATH_CONFIG

warnings.filterwarnings('ignore')
logger = logging.getLogger('collect_log_helper')

ORIGINAL_DATA_DIR = PATH_CONFIG.ORIGINAL_DATA_DIRECTORY
# 'result/original_data'


def drop_columns(data, columns):
    """
    Drop Columns and Return Data without request columns

        Parameters:
        -----------
             data (pandas.DataFrame) : pandas DataFrame
             columns (str) : one single name of data's columns

        Returns:
        --------
             data (pandas.DataFrame): pandas DataFrame without 'columns'
    """
    logger.info("{} 필드를 데이터에서 제외시킵니다".format(columns))
    data = data.drop(columns=[columns], axis=1)
    return data


class PreprocessorBase:
    """
    For Preparing data, Loading transformer, Changing parameters

    Base Class of TestPreprocessor (preprocess_tester.py)
              and InspectUserRequest (preprocess_helper.py)
    """
    @staticmethod
    def load_data(file_name):
        """
        Load Data from file and Return as pd.DataFrame.

        파일이름(__.csv, __.json)을 받아서 경로에 존재하면 로드하는 함수.
        (ex. 'my_data.json', 'my_data.csv')

        Parameters:
        -----------
             file_name (str) : file name
                               (ex. 'my_data.json', 'my_data.csv')

        Returns:
        --------
             get_data (pandas.DataFrame) : DataFrame loaded from json or csv file

        파일형식이 json 인 경우,
        1) 데이터 저장 형태가 다음과 같으면
            => list like [{column->value}, … ,{column->value}
            get_data = pd.read_json(get_data, orient='records')
        2) 데이터 저장 형태가 다음과 같으면
            => separate by enter {column:value,column:value,…}
            get_data = pd.read_json(file_path, lines=True, encoding='utf-8')
        """
        try:
            if os.path.splitext(file_name)[1] == '.csv':
                file_path = os.path.join(ORIGINAL_DATA_DIR, file_name)
                get_data = pd.read_csv(file_path)
            elif os.path.splitext(file_name)[1] == '.json':
                file_path = os.path.join(ORIGINAL_DATA_DIR, file_name)
                get_data = pd.read_json(file_path, lines=True, encoding='utf-8')
            return get_data
        except Exception as e:
            where_exception(error_msg=e)
            return None

    @staticmethod
    def _get_transformer(params):
        """
        Import Transformer and Return as Original object class

        Parameters:
        -----------
             params (dict) : PrepFunctionSerializer converted to dict types

        Returns:
        --------
             transformer (object) : original transformer class from python

        """
        try:
            module_ = __import__(params['LIBRARY_NAME'])
            class_ = getattr(module_, str(params['LIBRARY_OBJECT_NAME']))
            transformer_ = getattr(class_, params['LIBRARY_FUNCTION_NAME'])
            transformer = transformer_()
            return transformer
        except AttributeError:
            module_ = __import__(params['LIBRARY_NAME'] + '.' + params['LIBRARY_OBJECT_NAME'])
            class_ = getattr(module_, params['LIBRARY_OBJECT_NAME'])
            transformer_ = getattr(class_, params['LIBRARY_FUNCTION_NAME'])
            transformer = transformer_()
            return transformer
        except Exception as e:
            where_exception(error_msg=e)

    # 요청된 전처리 조건에 맞게 파라미터 값 변경
    @staticmethod
    def _change_transformer_params(transformer, params_dict):
        """
        Change Parameters of Transformer according to params_dict

        Parameters:
        -----------
             transformer (object) : Requested Original transformer
             params_dict (dict) : 'condition' value from Request Body

        Returns:
        --------
             transformer (object) : transformer with changed parameters
                                    according to param_dict

        """
        try:
            for k, v in params_dict.items():
                if not isinstance(v, str):
                    v = str(v)
                v = v.lower()
                logger.warning('[전처리 파라미터 변경] 파라미터 변경 요청=> key="{}", value="{}"'.format(k, v))
                if '.' in v:
                    if v.replace('.', '').isdigit():  # float 인 경우
                        v = float(v)
                        setattr(transformer, k, v)
                    else:
                        setattr(transformer, k, v)
                elif v.isdigit():  # int 인 경우
                    v = int(v)
                    setattr(transformer, k, v)
                elif v == 'true' or v == 'false':  # boolean 인 경우
                    v = bool(util.strtobool(v))
                    setattr(transformer, k, v)
                elif v == 'none':  # None 인 경우
                    setattr(transformer, k, None)
                elif v.startswith('['):  # list 인 경우
                    v = v[1:-1].replace(' ', '').split(',')
                    setattr(transformer, k, v)
                elif v.startswith('('):  # tuple 인 경우
                    v = literal_eval(v)
                    setattr(transformer, k, v)
                elif v == 'nan':  # np.nan 인 경우
                    setattr(transformer, k, np.nan)
                else:
                    setattr(transformer, k, v)
            logger.warning(transformer)
            return transformer
        except Exception as e:
            logger.error('[전처리 파라미터 변경] 전처리기 파라미터 변경을 실패했습니다')
            where_exception(error_msg=e)
            return dict(error_name='ParameterSyntaxError', error_detail=v)
