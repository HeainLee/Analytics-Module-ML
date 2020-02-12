import os
import sys
import json
import joblib
import logging
import decimal
import warnings
import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from pandas.core.common import flatten
from distutils import util  # to handle str to bool conversion

from ...models.original_data import OriginalData
from ...models.preprocess_functions import PreprocessFunction
from ...serializers.serializers import OriginalDataSerializer
from ...serializers.serializers import PreprocessFunctionSerializer
from ...config.result_path_config import PATH_CONFIG

warnings.filterwarnings('ignore')
logger = logging.getLogger('collect_log_helper')

ORIGINAL_DATA_DIR = PATH_CONFIG.ORIGINAL_DATA_DIRECTORY  # 'result/original_data'
PREPROCESSED_DATA_DIR = PATH_CONFIG.PREPROCESSED_DATA  # 'result/preprocessed_data'
PREPROCESS_TRANSFORMER_DIR = PATH_CONFIG.PREPROCESS_TRANSFORMER  # 'result/preprocess_transformer '


# 전처리 기능 중에서 사용자가 선택한 특정 컬럼을 삭제하는 기능을 수행하는 함수(사이킷런에서 해당 전처리 기능은 미제공)
def drop_columns(data, columns):
    logger.info("{} 필드를 데이터에서 제외시킵니다".format(columns))
    data = data.drop(columns=[columns], axis=1)
    return data


class PreparePreprocess:

    # 파일이름(__.csv, __.json)을 받아서 경로에 존재하면 로드하는 함수
    def load_data(self, file_name):
        if os.path.splitext(file_name)[1] == '.csv':
            try:
                get_data = os.path.join(ORIGINAL_DATA_DIR, file_name)
                get_data = pd.read_csv(get_data)
                return get_data
            except:
                logger.error('{} 파일이 존재하지 않습니다'.format(file_name))
                return None

        elif os.path.splitext(file_name)[1] == '.json':
            # 데이터 형태에 따라 읽어들이는 코드 변경될 수 있음
            try:
                get_data = os.path.join(ORIGINAL_DATA_DIR, file_name)
                # 데이터 저장 형태가 다음과 같으면 아래 코드 ==> list like [{column->value}, … ,{column->value}]
                # get_data = pd.read_json(get_data, orient='records')
                # 데이터 저장 형태가 다음과 같으면 아래 코드 ==> seperate by enter {column:value,column:value,…}
                get_data = pd.read_json(get_data, lines=True, encoding='utf-8')
                return get_data
            except:
                logger.error('{} 파일이 존재하지 않습니다'.format(file_name))
                return None

    # 변환기를 로드해서 리턴하는 함수
    def _get_transformer(self, params):
        try:
            module_ = __import__(params['LIBRARY_NAME'])
            class_ = getattr(module_, str(params['LIBRARY_OBJECT_NAME']))
            transformer_ = getattr(class_, params['LIBRARY_FUNCTION_NAME'])
            transformer = transformer_()
        except:
            module_ = __import__(params['LIBRARY_NAME'] + '.' + params['LIBRARY_OBJECT_NAME'])
            class_ = getattr(module_, params['LIBRARY_OBJECT_NAME'])
            transformer_ = getattr(class_, params['LIBRARY_FUNCTION_NAME'])
            transformer = transformer_()
        return transformer

    # 요청된 전처리 조건에 맞게 파라미터 값 변경
    def _change_transformer_params(self, transformer, params_dict):
        try:
            for k, v in params_dict.items():
                if not isinstance(v, str):
                    v = str(v)
                v = v.lower()
                logger.error('[전처리 파라미터 변경] 파라미터 변경 요청... ===> key="{}", value="{}"'.format(k, v))
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
            return transformer
        except Exception as e:
            logger.error('[전처리 파라미터 변경] 전처리기 파라미터 변경을 실패했습니다')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error('Error Type = {}, Error Occured File Name = {}, Error Line = {}, Error Message = {}'. \
                         format(exc_type, fname, exc_tb.tb_lineno, exc_obj))
            return dict(error_name='ParameterSyntaxError', error_detail=v)


class TestPreprocess(PreparePreprocess):

    def _float_to_str(self, float_type):
        converted_value = decimal.Context().create_decimal(repr(float_type))
        return format(converted_value, 'f')

    # 전처리 테스트 요청의 필수 파라미터 검사하는 함수
    def _mandatory_key_exists_original_patch(self, element):
        mand_key_level_one = 'request_test'
        mand_key_level_two = ['preprocess_functions_sequence_pk', 'field_name']
        if mand_key_level_one not in element.keys():
            return mand_key_level_one
        else:
            for request_info_dict in element['request_test']:
                for _key in mand_key_level_two:
                    if _key not in request_info_dict.keys():
                        return _key
        return True

    # 전처리 테스트를 수행하고 NUM 개수만큼만 결과를 리턴하는 함수 
    def _test_transformer(self, field, field_name, transformer):
        transformer_name = type(transformer).__name__
        try:
            NUM = 5  # 전처리된 결과를 보여줄 row 갯수
            transformer.fit(field.values.reshape(-1, 1))  # 필드가 하나일 때
            changed_field = transformer.transform(field.values.reshape(-1, 1))
            # 결과를 반환하기 위해 결과 타입 변환 및 통합
            if transformer_name == 'OneHotEncoder' or transformer_name == 'MultiLabelBinarizer':
                if isinstance(changed_field, csr_matrix):
                    changed_field = changed_field.toarray()
                changed_field = list(list(map(self._float_to_str, i)) for i in changed_field[:NUM])
                changed_field = list(map(lambda x: str(x), changed_field))
            elif transformer_name == 'LabelEncoder' or transformer_name == 'OrdinalEncoder':
                changed_field = list(map(lambda  x: str(x), list(flatten(changed_field[:NUM]))))
            else: # numerical 전처리의 경우에 해당
                changed_field = list(map(lambda x: "%.4f" % x, list(flatten(changed_field[:NUM]))))
            changed_field = dict(zip(range(0, len(changed_field)), changed_field))
            return changed_field

        except Exception as e:
            logger.error('{} 전처리 기능 fit&transform 도중 에러가 발생했습니다'.format(transformer_name))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error('Error Type = {}, Error Occured File Name = {}, Error Line = {}, Error Message = {}'. \
                         format(exc_type, fname, exc_tb.tb_lineno, exc_obj))
            return_error = {"error_name": "PreprocessTestError", "error_detail": field_name + ',' + transformer_name + ',' + str(exc_obj)}
            return return_error

    # 전처리 테스트 요청에 대한 요청 파라미터 검사하는 함수
    def check_request_body_original_patch(self, request_info, request_data, pk):
        logger.info('[전처리 테스트] 요청 ID [{}]의 필수 파라미터 검사...'.format(pk))
        # 필수 파라미터 검사
        is_keys = self._mandatory_key_exists_original_patch(element=request_info)
        if is_keys != True:
            return dict(error_type='4101', error_msg=is_keys)

        request_info_list = request_info['request_test']
        for request_info_dict in request_info_list:
            # logger.error('전처리 테스트 요청 정보 ===> {}'.format(request_info_dict))
            # 요청한 전처리 기능이 있는지 검사
            pfunc_id = request_info_dict['preprocess_functions_sequence_pk']
            all_pfunc_id = list(PreprocessFunction.objects.all().values_list(
                'PREPROCESS_FUNCTIONS_SEQUENCE_PK', flat=True))
            if not int(pfunc_id) in all_pfunc_id:
                return dict(error_type='4004', error_msg=pfunc_id)

            # 요청한 필드명이 원본 데이터에 있는지 검사
            field_name = request_info_dict['field_name']
            for field_name_ in field_name.replace(" ", "").split(','):
                if not field_name_ in list(request_data.columns):
                    return dict(error_type='4102', error_msg=field_name_)
            # 에러 상태가 없으면 True
        return True

    # 전처리 테스트 요청 받아서 수행하는 코드(each column return 5 values)
    def test_result(self, data, user_request_dict, pk):
        # user_request_dict=request.data['request_test']
        logger.info('[전처리 테스트] 요청 ID [{}]의 테스트 결과 생성 중...'.format(pk))
        TOTAL_LIST = []
        for get_request_dict in user_request_dict:
            field_name = get_request_dict['field_name']  # 전처리 요청한 컬럼명
            pfunction_pk = get_request_dict['preprocess_functions_sequence_pk']  # 전처리 기능 PK

            get_pfunc = PreprocessFunctionSerializer(
                PreprocessFunction.objects.get(pk=pfunction_pk)).data
            pfun_name = get_pfunc['PREPROCESS_FUNCTIONS_NAME']

            if pfun_name == 'DropColumns':
                logger.info('[전처리 테스트] {} 필드를 제외합니다'.format(field_name))
                if len(field_name.split(',')) != 1:
                    for field_name_ in field_name.replace(" ", "").split(','):
                        result_json = dict(
                            field_name=field_name_,
                            function_name=pfun_name,
                            function_parameter="None",
                            test_result=['drop_columns'] * 5)
                        TOTAL_LIST.append(result_json)
                else:
                    result_json = dict(
                        field_name=field_name,
                        function_name=pfun_name,
                        function_parameter="None",
                        test_result=['drop_columns'] * 5)
                    TOTAL_LIST.append(result_json)

            else:
                transformer = super()._get_transformer(params=get_pfunc)
                if 'condition' in get_request_dict.keys():  # 파라미터 수정을 요청한 경우
                    request_condition = get_request_dict['condition']
                    transformer = super()._change_transformer_params(
                        transformer=transformer, params_dict=request_condition)
                    if isinstance(transformer, dict) and 'error_name' in transformer.keys():
                        return dict(error_name='ParameterSyntaxError',
                                    error_detail=transformer['error_detail'])
                    else:
                        logger.info('[전처리 테스트] 요청 ID [{}] {} 전처리 기능의 변경된 파라미터 {} 가 적용됩니다' \
                                    .format(pk, pfun_name, transformer.get_params()))
                else:
                    logger.info('[전처리 테스트] 요청 ID [{}] {} 전처리 기능의 기본 파라미터 {} 가 적용됩니다' \
                                .format(pk, pfun_name, transformer.get_params()))
                    request_condition = None

                if len(field_name.split(',')) != 1:  # 전처리 테스트 요청한 필드 개수가 2개 이상인 경우
                    field_names = field_name.replace(" ", "").split(',')
                    for field_name_ in field_names:
                        field_name_ = field_name_.strip()
                        try:
                            field_column = data[field_name_].astype(float)
                        except:
                            field_column = data[field_name_]
                        logger.info('[전처리 테스트] {} 필드의 {} 전처리를 수행중...' \
                                    .format(field_name_, pfun_name))
                        changed_field = self._test_transformer(
                            field=field_column, field_name=field_name_, transformer=transformer)

                        # {"error_name":"PreprocessTestError","error_detail":field_name+','+transformer_name}
                        if type(changed_field) == dict and 'error_name' in changed_field.keys():
                            return changed_field

                        else:
                            logger.info('[전처리 테스트] {} 필드의 전처리 수행 결과... {}' \
                                        .format(field_name_, changed_field))
                            result_json = dict(
                                field_name=field_name_,
                                function_name=pfun_name,
                                function_parameter=request_condition,
                                test_result=changed_field)
                            TOTAL_LIST.append(result_json)
                else:
                    try:
                        field_column = data[field_name].astype(float)
                    except:
                        field_column = data[field_name]
                    logger.info('[전처리 테스트] {} 필드의 {} 전처리를 수행중...' \
                                .format(field_name, pfun_name))
                    changed_field = self._test_transformer(
                        field=field_column, field_name=field_name, transformer=transformer)

                    # {"error_name":"PreprocessTestError","error_detail":field_name+','+transformer_name}
                    if type(changed_field) == dict and 'error_name' in changed_field.keys():
                        return changed_field

                    else:
                        logger.info('[전처리 테스트] {} 필드의 전처리 수행 결과... {}' \
                                    .format(field_name, changed_field))
                        result_json = dict(
                            field_name=field_name,
                            function_name=pfun_name,
                            function_parameter=request_condition,
                            test_result=changed_field)
                        TOTAL_LIST.append(result_json)

        return TOTAL_LIST


class CreatePreprocessedData(TestPreprocess):

    def __init__(self):
        self.original_data_pk = None
        self.origianl_data = None
        self.data_saved_path = None
        self.data_loaded = None

    # 전처리 데이터 생성 요청의 필수 파라미터 검사하는 함수
    def _mandatory_key_exists_preprocessed_post(self, element):
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

    # 전처리 데이터 생성 요청에 대한 요청 파라미터 검사하는 함수
    def check_request_body_preprocessed_post(self, request_info):
        # 필수 파라미터 검사
        is_keys = self._mandatory_key_exists_preprocessed_post(
            element=request_info)
        if is_keys != True:
            return dict(error_type='4101', error_msg=is_keys)

        # 요청한 원본데이터 있는지 검사
        self.original_data_pk = request_info['original_data_sequence_pk']
        all_origin_id = list(OriginalData.objects.all().values_list(
            'ORIGINAL_DATA_SEQUENCE_PK', flat=True))
        if not int(self.original_data_pk) in all_origin_id:
            return dict(error_type='4004',
                        error_msg=str(self.original_data_pk) + ',original_data')
        else:
            self.origianl_data = OriginalDataSerializer(
                OriginalData.objects.get(pk=self.original_data_pk)).data
            self.data_saved_path = self.origianl_data['FILEPATH']
            if not os.path.isfile(self.data_saved_path):
                logger.error('{} 경로가 존재하지 않습니다'.format(self.data_saved_path))
                return dict(error_type='4004',
                            error_msg=str(self.data_saved_path) + ',file_not_found')
            else:
                self.data_loaded = super().load_data(
                    file_name=os.path.split(self.data_saved_path)[1])

        request_info_list = request_info['request_data']
        for request_info_dict in request_info_list:
            # 요청한 전처리기능이 있는지 검사
            pfunc_id = request_info_dict['preprocess_functions_sequence_pk']
            all_pfunc_id = list(PreprocessFunction.objects.all().values_list(
                'PREPROCESS_FUNCTIONS_SEQUENCE_PK', flat=True))
            if not int(pfunc_id) in all_pfunc_id:
                return dict(error_type='4004',
                            error_msg=str(pfunc_id) + ',preprocess_function')

            # 요청한 필드명이 원본 데이터에 있는지
            field_name = request_info_dict['field_name']
            for field_name_ in field_name.replace(" ", "").split(','):
                if not field_name_ in list(self.data_loaded.columns):
                    return dict(error_type='4102', error_msg=field_name_)
            # 에러 상태가 없으면 True
        return True


class PreprocessTask(CreatePreprocessedData):

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
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error('Error Type = {}, Error Occured File Name = {}, Error Line = {}, Error Message = {}'. \
                         format(exc_type, fname, exc_tb.tb_lineno, exc_obj))
            return False


class DataSummary:

    def __init__(self, path):
        self.data = self.get_data(path)
        self.columns = self.data.columns

    def get_data(self, path):
        if os.path.splitext(path)[1] == '.csv':
            df_data = pd.read_csv(path)
        elif os.path.splitext(path)[1] == '.json':
            if 'O' in path:
                df_data = pd.read_json(path, lines=True, encoding='utf-8').fillna("None").sort_index()
            elif 'P' in path:
                df_data = pd.read_json(path, orient='index').sort_index()
        return df_data

    def columns_info(self):
        columns_list = list(self.columns)
        return str(columns_list)

    def sample_info(self):
        sample_data = self.data.loc[:4].to_json()
        sample_data = json.loads(sample_data)
        return str(sample_data)

    def size_info(self):
        amount = self.data.shape[0]
        return amount

    def categorical_(self, col_name):
        col_data = self.data[col_name]
        if len(set(col_data)) == len(col_data):
            return "count", {"elements": "unique", "frequency": [str(len(col_data))]}
        elif len(set(col_data)) < 3:
            dic = dict(col_data.value_counts())
            return "pie", {"elements": list(dic.keys()), "frequency": list(map(str, dic.values()))}
        else:
            dic = dict(col_data.value_counts())
            return "bar", {"elements": list(dic.keys()), "frequency": list(map(str, dic.values()))}

    def numerical_(self, col_name):
        (freq, bins, patches) = plt.hist(self.data[col_name])
        bins_means = []
        for i in range(len(bins) - 1):
            bins_means.append(np.mean([bins[i], bins[i + 1]]))
        return "histogram", {"bins_means": list(map(str, np.round(bins_means, decimals=3))),
                             "frequency": list(map(str, np.round(freq, decimals=3)))}

    def statistics_info(self):
        self.graph_types = []
        self.compact_datas = []
        self.data_statistics = []
        self.column_dtypes = self.data.dtypes.replace('object', 'string')
        for i, j in enumerate(self.column_dtypes):
            try:
                if j in ["float64", "float32", "int64", "int32"]:  # numerical일 경우
                    self.column_dtypes[i] = 'numerical'
                    self.graph_type, self.compact_data = self.numerical_(self.columns[i])
                else:  # categorical 일 경우
                    self.column_dtypes[i] = 'categorical'
                    self.graph_type, self.compact_data = self.categorical_(self.columns[i])

                self.graph_types.append(self.graph_type)
                self.compact_datas.append(self.compact_data)
            except:
                logger.error("Cant Extract Graph Data"+self.columns[i])
                self.graph_types.append("")
                self.compact_datas.append({})

        for name, dtype, graph_type, compact_data in zip(
                self.columns, self.column_dtypes, self.graph_types, self.compact_datas):
            single_column_info = {'name': name, 'type': dtype, 'graph_type': graph_type, 'compact_data': compact_data}
            self.data_statistics.append(single_column_info)
        self.data_statistics = json.dumps(self.data_statistics)
        del (self.column_dtypes)
        return self.data_statistics
