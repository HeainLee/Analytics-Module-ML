import os
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
from django.http import Http404

from .preprocess_base import PreprocessorBase, drop_columns
from ..utils.custom_decorator import where_exception
from ...models.original_data import OriginalData
from ...models.preprocess_functions import PreprocessFunction
from ...serializers.serializers import OriginalDataSerializer
from ...serializers.serializers import PreprocessFunctionSerializer
from ...config.result_path_config import PATH_CONFIG

warnings.filterwarnings('ignore')
logger = logging.getLogger('collect_log_helper')

PREPROCESSED_DATA_DIR = PATH_CONFIG.PREPROCESSED_DATA
# 'result/preprocessed_data'
PREPROCESS_TRANSFORMER_DIR = PATH_CONFIG.PREPROCESS_TRANSFORMER
# 'result/preprocess_transformer '


def _error_return_dict(error_type, error_msg):
    """
    Return common error dictionary type

        Parameters:
        -----------
             error_type (str) : type of error (eg. '4102')
             error_msg (str) : detail message of the error

        Returns:
        --------
             (dict) : common error dictionary
    """
    return dict(error_type=error_type, error_msg=error_msg)


class InspectUserRequest(PreprocessorBase):
    """
    For inspecting Users' POST request from user's request body

        Attributes:
        -----------
        original_data_pk (int) : user requested ID of original Data
        data_saved_path (str) : original data's saved path
                                (eg. ['FILEPATH'] from query)
        data_loaded (pandas.DataFrame) : user requested Data for Preprocessing

    """

    def __init__(self):
        self.original_data_pk = None
        self.data_saved_path = None
        self.data_loaded = None

    @staticmethod
    def _mandatory_key_exists_preprocessed_post(element):
        """
        Check whether mandatory keys are existed

            Parameters:
            -----------
                 element (dict) : raw request information from user's request body

            Returns:
            --------
                 True (bool) : True, if all mandatory keys is satisfied
                 or
                 _key (str) : omitted key (induce 4101)
        """
        mand_key_level_one = ['original_data_sequence_pk', 'request_data']
        mand_key_level_two = ['preprocess_functions_sequence_pk', 'field_name']
        if list(element.keys()) != mand_key_level_one:
            _key = set(mand_key_level_one).difference(list(element.keys()))
            return ''.join(_key)
        else:
            for request_info_dict in element['request_data']:
                for _key in mand_key_level_two:
                    if _key not in request_info_dict.keys():
                        return _key
        return True

    def _check_request_pk(self, data_id, func_id):
        return True

    # 전처리 데이터 생성 요청에 대한 요청 파라미터 검사하는 함수
    def check_post_mode(self, request_info):

        logger.warning('[데이터 전처리] Check Request Info')

        # 필수 파라미터 검사 (4101)
        is_keys = self._mandatory_key_exists_preprocessed_post(
            element=request_info)
        if isinstance(is_keys, str):  # mandatory key name (str)
            return _error_return_dict('4101', is_keys)

        # 요청한 원본데이터 있는지 검사 (Http404/4004)
        self.original_data_pk = request_info['original_data_sequence_pk']
        all_origin_id = list(OriginalData.objects.all().values_list(
            'ORIGINAL_DATA_SEQUENCE_PK', flat=True))
        if not int(self.original_data_pk) in all_origin_id:
            raise Http404
        else:
            self.data_saved_path = OriginalDataSerializer(
                OriginalData.objects.get(pk=self.original_data_pk)).data['FILEPATH']

            if not os.path.isfile(self.data_saved_path):
                logger.error('{} 경로가 존재하지 않습니다'.format(self.data_saved_path))
                return _error_return_dict('4004', self.data_saved_path)
            else:
                self.data_loaded = super().load_data(
                    file_name=os.path.split(self.data_saved_path)[1])

        request_info_list = request_info['request_data']
        for request_info_dict in request_info_list:
            # 요청한 전처리기능이 있는지 검사 (Http404)
            pfunc_id = request_info_dict['preprocess_functions_sequence_pk']
            all_pfunc_id = list(PreprocessFunction.objects.all().values_list(
                'PREPROCESS_FUNCTIONS_SEQUENCE_PK', flat=True))
            if not int(pfunc_id) in all_pfunc_id:
                raise Http404

            # 요청한 필드명이 원본 데이터에 있는지(4102)
            field_name = request_info_dict['field_name']
            for field_name_ in field_name.replace(" ", "").split(','):
                if not field_name_ in list(self.data_loaded.columns):
                    return _error_return_dict('4102', field_name_)
        return True  # 에러 상태가 없으면 True


class PreprocessTask(InspectUserRequest):

    def __init__(self):
        self.get_pfunc = None

    # 전처리된 데이터를(pandas.DataFrame)을 json을 변환하여 저장하는 함수
    def save_preprocessed_data(self, preprocessed_data, preprocessed_data_name):
        file_path = os.path.join(PREPROCESSED_DATA_DIR, preprocessed_data_name)
        preprocessed_data.to_json(file_path, orient='index')
        # orient 옵션에 따라 저장되는 형태 달라짐 / index = dict like {index -> {column -> value}}
        return file_path, preprocessed_data_name

    # 변환기를 저장하는 함수
    def _save_transformer(self, transformer, transformer_name):
        file_path = os.path.join(PREPROCESS_TRANSFORMER_DIR, transformer_name)
        joblib.dump(transformer, file_path)
        return file_path, transformer_name

    # 학습용으로 사용될 데이터에 전처리를 수행하고 결과를 리턴하는 함수(_task_general_preprocess 함수에서 사용됨)
    ## 변환기를 학습데이터로 학습하고(fit) 전처리 적용하고(transform) 데이터에 덮어씌우는(changed_field) 과정
    ### (**전처리 방법의 동작방식에 따라 조건문 추가될 수 있음)
    def _train_data_transformer(self, data, field_name, transformer):
        try:
            field_column = data[field_name].astype(float)
        except:
            field_column = data[field_name]
        try:
            transformer.fit(field_column.values.reshape(-1, 1))
            changed_field = transformer.transform(field_column.values.reshape(-1, 1))

            # 변환기 정보를 딕셔너리로 저장하는 과정
            transformer_info = dict(function_name=type(transformer).__name__)

            # transfrom된 데이터와 원본 데이터 통합
            if type(transformer).__name__ == 'OneHotEncoder':
                # 원핫인코딩을 한 경우 통합 방법(새로운 칼럼 생성됨)
                changed_field = changed_field.toarray() \
                    if not isinstance(changed_field, np.ndarray) else changed_field
                OneHot = pd.DataFrame(
                    changed_field, columns=[field_name + "_" + str(int(i)) \
                                            for i in range(changed_field.shape[1])])
                data = pd.concat([data, OneHot], axis=1, sort=False)
                data = data.drop(field_name, axis=1)
                transformer_info['original_classes'] = list(transformer.categories_[0])
                encoded_class = transformer.transform(np.array(transformer.categories_).reshape(-1, 1))
                encoded_class = encoded_class.toarray() \
                    if not isinstance(encoded_class, np.ndarray) else encoded_class
                transformer_info['encoded_classes'] = encoded_class.tolist()

            elif type(transformer).__name__ == 'MultiLabelBinarizer':
                # MultiLabelBinarizer 경우 통합 방법(새로운 칼럼 생성됨)
                changed_field = changed_field.toarray() \
                    if not isinstance(changed_field, np.ndarray) else changed_field
                OneHot = pd.DataFrame(
                    changed_field, columns=[field_name + "_" + str(int(i)) \
                                            for i in range(changed_field.shape[1])])
                data = pd.concat([data, OneHot], axis=1, sort=False)
                data = data.drop(field_name, axis=1)
                transformer_info['original_classes'] = list(transformer.classes_)
                encoded_class = transformer.transform(np.array(transformer.classes_).reshape(-1, 1))
                encoded_class = encoded_class.toarray() \
                    if not isinstance(encoded_class, np.ndarray) else encoded_class
                transformer_info['encoded_classes'] = encoded_class.tolist()

            elif type(transformer).__name__ == 'KBinsDiscretizer':
                # KBinsDiscretizer인 경우 통합 방법(새로운 칼럼 생성됨)
                changed_field = changed_field.toarray() \
                    if not isinstance(changed_field, np.ndarray) else changed_field
                OneHot = pd.DataFrame(
                    changed_field, columns=[field_name + "_" + str(int(i)) \
                                            for i in range(changed_field.shape[1])])
                data = pd.concat([data, OneHot], axis=1, sort=False)
                data = data.drop(field_name, axis=1)

            else:  # 원핫인코딩이 또는 kbins구간화 아닌 경우 통합 방법(기존 값을 덮어씀)
                data[field_name] = changed_field
                if type(transformer).__name__ == 'LabelEncoder':
                    transformer_info['original_classes'] = list(transformer.classes_)
                    transformer_info['encoded_classes'] = list(np.unique(changed_field))
            return data, transformer, transformer_info

        except Exception as e:
            logger.error('Error Type = {} / Error Message = {}'.format(type(e), e))
            return False

    # DropColumns 처리로 해당 열 삭제 후 data과 처리정보(info_list) 반환하는 함수
    def _task_drop_columns(self, data, field_name, pfunction_pk):
        logger.info('{} 필드를 데이터에서 제외시킵니다'.format(field_name))
        info_list = []
        if len(field_name.split(',')) != 1:
            for field_name_ in field_name.replace(" ", "").split(','):
                data = drop_columns(data=data, columns=field_name_)
                info_dict = dict(
                    field_name=field_name_,
                    function_name='DropColumns',
                    function_pk=pfunction_pk,
                    file_name=None,
                    original_classes=None,
                    encoded_classes=None
                )
                info_list.append(info_dict)
        else:
            data = drop_columns(data=data, columns=field_name)
            info_dict = dict(
                field_name=field_name,
                function_name='DropColumns',
                function_pk=pfunction_pk,
                file_name=None,
                original_classes=None,
                encoded_classes=None
            )
            info_list.append(info_dict)
        return data, info_list

    # 사이킷런 전처리의 실제 수행된 결과인 data와 처리정보(info_list) 반환하는 함수
    def _task_general_preprocess(self, data, request_dict, pk, incremental_N):
        info_list = []
        field_name = request_dict['field_name']
        pfunction_pk = request_dict['preprocess_functions_sequence_pk']
        pfun_name = self.get_pfunc['PREPROCESS_FUNCTIONS_NAME']
        transformer = super()._get_transformer(params=self.get_pfunc)

        if 'condition' in request_dict.keys():  # 파라미터 수정을 요청한 경우
            request_condition = request_dict['condition']
            transformer = super()._change_transformer_params(
                transformer=transformer, params_dict=request_condition)
            logger.info('{} 전처리 기능의 변경된 파라미터 {} 가 적용됩니다' \
                        .format(pfun_name, transformer.get_params()))
        else:
            logger.info('{} 전처리 기능의 기본 파라미터 {} 가 적용됩니다' \
                        .format(pfun_name, transformer.get_params()))

        if len(field_name.split(',')) != 1:  # 전처리 요청한 필드 개수가 2개 이상인 경우
            field_names = field_name.replace(" ", "").split(',')
            for field_name_ in field_names:
                field_name_ = field_name_.strip()
                logger.info('{} 필드에 {} 전처리 수행을 요청했습니다' \
                            .format(field_name_, pfun_name))
                data, fitted_transformer, fitted_info = self._train_data_transformer(
                    data=data, field_name=field_name_, transformer=transformer)
                incremental_N += 1
                logger.info('{} 필드에 적용된 {} 전처리기 저장 ===> T_{}_{}.pickle' \
                            .format(field_name_, pfun_name, pk, incremental_N))

                file_path, file_name = self._save_transformer(
                    transformer=fitted_transformer,
                    transformer_name='T_{}_{}.pickle'.format(pk, incremental_N))
                info_dict = dict(
                    field_name=field_name_,
                    function_name=pfun_name,
                    function_pk=pfunction_pk,
                    file_name=file_name,
                    original_classes=None,
                    encoded_classes=None
                )
                if 'original_classes' in fitted_info.keys():
                    info_dict['original_classes'] = fitted_info['original_classes']
                    info_dict['encoded_classes'] = fitted_info['encoded_classes']

                info_list.append(info_dict)

        else:  # 전처리 요청한 필드 개수가 1개인 경우
            logger.info('{} 필드에 {} 전처리 수행을 요청했습니다' \
                        .format(field_name, pfun_name))
            data, fitted_transformer, fitted_info = self._train_data_transformer(
                data=data, field_name=field_name, transformer=transformer)
            incremental_N += 1
            logger.info('{} 필드에 적용된 {} 전처리기 저장 ===> T_{}_{}.pickle'. \
                        format(field_name, pfun_name, pk, incremental_N))

            file_path, file_name = self._save_transformer(
                transformer=fitted_transformer,
                transformer_name='T_{}_{}.pickle'.format(pk, incremental_N))
            info_dict = dict(
                field_name=field_name,
                function_name=pfun_name,
                function_pk=pfunction_pk,
                file_name=file_name,
                original_classes=None,
                encoded_classes=None
            )
            if 'original_classes' in fitted_info.keys():
                info_dict['original_classes'] = fitted_info['original_classes']
                info_dict['encoded_classes'] = fitted_info['encoded_classes']
            info_list.append(info_dict)

        return data, info_list, incremental_N

    # 전처리 데이터 생성 결과(data, preprocessed_data_info_list)를 반환하는 함수
    def task_result(self, data, user_request_dict, pk):
        preprocessed_data_info_list = []
        incremental_N = 0

        try:
            for get_request_dict in user_request_dict:
                field_name = get_request_dict['field_name']
                pfunction_pk = get_request_dict['preprocess_functions_sequence_pk']

                self.get_pfunc = PreprocessFunctionSerializer(
                    PreprocessFunction.objects.get(pk=pfunction_pk)).data

                if self.get_pfunc['PREPROCESS_FUNCTIONS_NAME'] == 'DropColumns':
                    data, info_list = self._task_drop_columns(
                        data=data, field_name=field_name, pfunction_pk=pfunction_pk)
                    [preprocessed_data_info_list.append(result) for result in info_list]

                else:
                    data, info_list, incremental_N = self._task_general_preprocess(
                        data=data, request_dict=get_request_dict, pk=pk, incremental_N=incremental_N)
                    [preprocessed_data_info_list.append(result) for result in info_list]
            return data, preprocessed_data_info_list

        except Exception as e:
            logger.error('전처리 데이터 생성에 실패했습니다')
            where_exception(e)
            return False
