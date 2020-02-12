import os
import json
import copy
import joblib
import inspect
import logging
import warnings
import numpy as np
import pandas as pd
from distutils import util
from ast import literal_eval
from django.http import Http404
from sklearn import base
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimators_dtypes

from ...config.result_path_config import PATH_CONFIG
from ...models.algorithm import Algorithm
from ...models.original_data import OriginalData
from ...models.preprocessed_data import PreprocessedData
from ...serializers.serializers import ALGOSerializer
from ...serializers.serializers import OriginalDataSerializer
from ...serializers.serializers import PreprocessedDataSerializer
from ..utils.custom_decorator import where_exception

warnings.filterwarnings('ignore')
logger = logging.getLogger('collect_log_helper')

ORIGINAL_DATA_DIR = PATH_CONFIG.ORIGINAL_DATA_DIRECTORY  # 'result/original_data'
PREPROCESSED_DATA_DIR = PATH_CONFIG.PREPROCESSED_DATA  # 'result/preprocessed_data'
MODEL_DIR = PATH_CONFIG.MODEL  # 'result/model'


class PrepareModelTrain:
    """
    For preparing data, inspect data, get estimator, change parameters

    """

    # 원본데이터를 로드하는 함수
    @staticmethod
    def _load_original_data(file_name):
        """
        @type file_name: str
        @type loaded_data: pandas.core.frame.DataFrame
        @param file_name: file name (ex. 'O_1.json', 'O_2.csv')
        @return: pandas DataFrame converted from json or csv file
        """
        try:
            if os.path.splitext(file_name)[1] == '.csv':
                file_path = os.path.join(ORIGINAL_DATA_DIR, file_name)
                loaded_data = pd.read_csv(file_path)
            elif os.path.splitext(file_name)[1] == '.json':
                file_path = os.path.join(ORIGINAL_DATA_DIR, file_name)
                # 데이터 저장 형태 : separated by enter {column: value,column: value,..}
                loaded_data = pd.read_json(file_path, lines=True, encoding='utf-8')
            return loaded_data
        except Exception as e:
            where_exception(error_msg=e)
            return None

    # 전처리된데이터를 로드하는 함수
    def _load_preprocessed_data(self, file_name):
        if os.path.splitext(file_name)[1] == '.csv':
            try:
                file_path = os.path.join(PREPROCESSED_DATA_DIR, file_name)
                get_csv_data = pd.read_csv(file_path)
                return get_csv_data
            except:
                logger.warning('{} 파일이 존재하지 않습니다'.format(file_name))
                return None

        elif os.path.splitext(file_name)[1] == '.json':
            try:
                file_path = os.path.join(PREPROCESSED_DATA_DIR, file_name)
                get_json_data = json.loads(open(file_path, 'r').read())
                json_to_df = pd.DataFrame.from_dict(get_json_data, orient='index')
                json_to_df.index = json_to_df.index.astype(int)
                json_to_df = json_to_df.sort_index()
                return json_to_df
            except:
                logger.warning('{} 파일이 존재하지 않습니다'.format(file_name))
                return None

    # 요청한 데이터의 변수 타입이 수치형인지 검사하는 함수
    def _inspect_data(self, data_set):
        for dtype in data_set.dtypes:
            if dtype == 'object':
                return False
            else:
                return True

    # 모델을 로드해서 리턴하는 함수
    def _get_estimator(self, params):
        module_ = __import__(params['LIBRARY_NAME'] +
                             '.'+params['LIBRARY_OBJECT_NAME'])
        class_ = getattr(module_, params['LIBRARY_OBJECT_NAME'])
        clf_ = getattr(class_, params['LIBRARY_FUNCTION_NAME'])
        clf = clf_()
        return clf

    # 요청된 모델 조건에 맞게 파라미터 값 변경
    def _change_params(self, model, param):
        try:
            for k, v in param.items():
                if not isinstance(v, str):
                    v = str(v)
                v = v.lower()
                logger.error('변경 요청한 파라미터 {} = {}'.format(k, v))
                if '.' in v:
                    if v.replace('.', '').isdigit(): # float인 경우
                        v = float(v)                
                        setattr(model, k, v)
                    else:
                        setattr(model, k, v)
                elif v.isdigit(): # int인 경우
                    v = int(v)
                    setattr(model, k, v)
                elif v.startswith('-'): # neg int인 경우
                    v = int(v)
                    setattr(model, k, v)
                elif v == 'none': # None인 경우
                    setattr(model, k, None)
                elif v == 'true' or v == 'false': # boolen인 경우
                    v = bool(util.strtobool(v))
                    setattr(model, k, v)
                elif v.startswith('{'): # dict인 경우
                    v = literal_eval(v)
                    setattr(model, k, v)
                else:
                    setattr(model, k, v)
            return model
        except Exception as e:
            logger.error('Error Type = {} / Error Message = {}'.format(type(e), e))
            return model

class InspectUserRequest(PrepareModelTrain):

    def __init__(self):
        self.clf = None
        self.library_name = None
        self.function_usage = None
        self.model_param = None
        self.train_param = None
        self.train_data = None
        self.train_data_dict = None
        self.train_data_type = None

    # 필수 파라미터 검사 (4101)
    def _mandatory_key_exists_models_post(self, element):
        mand_key = ['algorithms_sequence_pk', 'train_data', 'train_parameters']
        for key in mand_key:
            if key not in element.keys():
                return key
            if key == 'train_data':
                get_train_data_info = element['train_data'].keys()
                is_original_data = bool(
                    'original_data_sequence_pk' in get_train_data_info)
                is_preprocessed_data = bool(
                    'preprocessed_data_sequence_pk' in get_train_data_info)
                if is_original_data == False and is_preprocessed_data == False:
                    return key
        return True

    # 요청한 알고리즘 ID와 데이터 ID가 있는지 검사 (4004)
    def _check_request_pk(self, algo_id, data_id, data_type):
        if not int(algo_id) in list(Algorithm.objects.all().values_list(\
            'ALGORITHM_SEQUENCE_PK', flat=True)):
            # raise Http404
            return False
        else:
            # USAGE 확인(regression, classification)
            user_request_algorithm = ALGOSerializer(
                Algorithm.objects.get(pk=algo_id)).data
            self.library_name = user_request_algorithm['LIBRARY_NAME']
            self.function_usage = user_request_algorithm['LIBRARY_FUNCTION_USAGE']
            self.clf = super()._get_estimator(params=user_request_algorithm)

        if data_type == 'original_data_sequence_pk':
            self.train_data_type = 'original'
            if not int(data_id) in list(OriginalData.objects.all().values_list(\
                'ORIGINAL_DATA_SEQUENCE_PK', flat=True)):
                # raise Http404
                return False
            else:
                self.train_data_dict = OriginalDataSerializer(
                    OriginalData.objects.get(pk=data_id)).data
                data_path = self.train_data_dict['FILEPATH']
                if not os.path.isfile(data_path):
                    logger.error('{} 경로가 존재하지 않습니다'.format(data_path))
                    return dict(error_type='4004', error_msg=data_path)
                else:
                    self.train_data = super()._load_original_data(
                        file_name=os.path.split(data_path)[1])

        elif data_type == 'preprocessed_data_sequence_pk':
            self.train_data_type = 'preprocessed'
            if not int(data_id) in list(PreprocessedData.objects.all().values_list(\
                'PREPROCESSED_DATA_SEQUENCE_PK', flat=True)):
                # raise Http404
                return False
            else:
                self.train_data_dict = PreprocessedDataSerializer(
                    PreprocessedData.objects.get(pk=data_id)).data
                data_path = self.train_data_dict['FILEPATH']
                if not os.path.isfile(data_path):
                    logger.error('{} 경로가 존재하지 않습니다'.format(data_path))
                    return dict(error_type='4004', error_msg=data_path)
                else:
                    self.train_data = super()._load_preprocessed_data(
                        file_name=os.path.split(data_path)[1])
        return True
    
    # model_parameters 검사 (4012/4000)
    # 예) {"max_features": "log2", "min_samples_split":3}
    def _check_model_parameters(self, model_param_dict):
        if model_param_dict: # model_parameters가 있는 경우
            model_param_list = list(model_param_dict.keys())
            logger.info('사용자가 변경을 요청한 모델 파라미터 {}'.format(model_param_list))
            for name in model_param_list:
                if not hasattr(self.clf, name):
                    logger.error('{}는 요청 가능한 모델 파라미터가 아닙니다'.format(name))
                    return dict(error_type='4102', error_msg=name)
            self.model_param = self.clf.get_params()
            self.model_param.update(model_param_dict)
            for k, v in self.model_param.items():
                if isinstance(v, bool) or isinstance(v, type(None)):
                    self.model_param[k] = str(v)

            try:
                self.clf = super()._change_params(model=self.clf, param=model_param_dict)
                logger.info('모델 파라미터 적용 결과 {}'.format(self.clf.get_params()))
                check_estimators_dtypes(name=type(self.clf).__name__, estimator_orig=self.clf)
            except Exception as e:
                logger.error('모델 파라미터 에러 발생 {}'.format(e))
                return 'model_param_error'
                         
        else: # model_parameters가 없는 경우
            self.model_param = self.clf.get_params()
            for k, v in self.model_param.items():
                if isinstance(v, bool) or isinstance(v, type(None)):
                    self.model_param[k] = str(v)
            logger.info('모델 파라미터를 요청하지 않았으므로, 모델의 기본 파라미터가 적용됩니다')
        return True

    # train_parameters 검사 (4101, 4102)
    # 예)  "y": "target"
    def _check_train_parameters(self, train_param_dict):
        if train_param_dict: # train_parameters가 있는 경우
            train_param_list = list(train_param_dict.keys())
            logger.info('사용자가 요청한 학습 파라미터 {}'.format(train_param_list))
            for name in train_param_list:
                if name not in inspect.getfullargspec(self.clf.fit).args:
                    logger.error('{}는 요청 가능한 학습 파라미터가 아닙니다'.format(name))
                    return dict(error_type='4102', error_msg=name)

            # 지도학습에서 train_parameter에 y 값이 있는지 검사 (4101)
            if self.library_name == 'sklearn' and \
             (self.function_usage == 'regression' or 'classification'):
                if 'y' not in train_param_list:
                    logger.error('지도학습에서 y 학습 파라미터는 필수입니다')
                    return dict(error_type='4101', error_msg='y')

            # 지도학습에서 요청한 train_parameter의 y값이 유휴한 값인지 검사 (4102)
            if train_param_dict['y'] not in self.train_data.columns:
                logger.error('{}는 유효한 값이 아닙니다'.format(train_param_dict['y']))
                return dict(error_type='4102', error_msg=train_param_dict['y'])

            self.train_param = {} # Command필드에 저정한 train_parameters정보 생성
            for k, v in inspect.signature(self.clf.fit).parameters.items():
                if v.default is not inspect.Parameter.empty:
                    self.train_param[k] = v.default
                else:
                    if k == 'X':
                        except_value = train_param_dict['y']
                        train_data_columns = list(self.train_data.columns)
                        train_data_columns.remove(except_value)
                        self.train_param[k] = train_data_columns
                    else:
                        self.train_param[k] = None
            self.train_param.update(train_param_dict)
            for k, v in self.train_param.items():
                if isinstance(v, bool) or isinstance(v, type(None)):
                    self.train_param[k] = str(v)
        return True

    # 모델 학습 요청에 대한 요청 파라미터 검사하는 함수
    def check_request_body_models_post(self, request_info, pk):
        logger.info('요청 ID [{}]의 필수 파라미터를 검사...'\
            .format(pk))
        # 필수 파라미터 검사 (4101)
        is_keys = self._mandatory_key_exists_models_post(element=request_info)
        if is_keys != True:
            return dict(error_type='4101', error_msg=is_keys)

        logger.info('요청 ID [{}]의 요청 알고리즘 및 데이터 유효성 검사...'\
            .format(pk))
        # 요청한 알고리즘 ID와 데이터 ID가 있는지 검사 (4004)
        algorithm_id = request_info['algorithms_sequence_pk']
        train_data_type = list(request_info['train_data'].keys())[0]
        train_data_id = list(request_info['train_data'].values())[0]
        is_valid = self._check_request_pk(
            algo_id=algorithm_id, data_id=train_data_id, data_type=train_data_type)
        if is_valid != True:
            if isinstance(is_valid, dict) and is_valid['error_type'] == '4004':
                return dict(error_type='4004', error_msg=is_valid['error_msg']) # file_not_found
            else:
                raise Http404 # algorithm 또는 data ID가 없는 경우 -- resource_not_found
        
        logger.info('요청 ID [{}]의 [model_parameters] 파라미터 검사...'\
            .format(pk))
        # model_parameters 검사 (4012/4013)
        get_model_param = request_info['model_parameters'] if \
        'model_parameters' in list(request_info.keys()) else False
        is_valid = self._check_model_parameters(
            model_param_dict=get_model_param)
        if is_valid != True:
            if isinstance(is_valid, dict) and is_valid['error_type'] == '4102':
                return dict(error_type=is_valid['error_type'], error_msg=is_valid['error_msg'])
            elif isinstance(is_valid, str) and is_valid == 'model_param_error':
                return dict(error_type='4103', error_msg=is_valid)

        logger.info('요청 ID [{}]의 [train_parameters] 파라미터를 검사...'\
            .format(pk))
        # train_parameters 검사 (4101, 4102)
        get_train_param = request_info['train_parameters']

        is_valid = self._check_train_parameters(
            train_param_dict=get_train_param)
        if is_valid != True:
            return dict(error_type=is_valid['error_type'], 
                error_msg=is_valid['error_msg'])
                
        # 요청한 데이터의 변수 타입이 수치형인지 검사 (4022)
        # 나중에 model 학습 부분에서의 에러 발생에 대비한 검사 항목이며, 언제든 변경될 수 있음
        if not super()._inspect_data(data_set=self.train_data):
            logger.error('학습 또는 적용 데이터 타입은 numerical이어야 합니다')
            return dict(error_type='4022', 
                error_msg='Data is not suitable for the algorithm')
        return True

    # 모델 중지/재시작/테스트 요청 필수 파라미터 검사하는 함수
    def check_patch_mode(self, request_info):
        valid_patch_mode = ['STOP', 'TEST', 'RESTART'] # 'LOAD', 'UNLOAD' 
        # 필수 body parameters인 mode를 요청하지 않은 경우
        if 'mode' not in request_info.keys():
            return dict(error_type='4101', error_msg='mode')
        # mode가 valid_patch_mode가 아닌 경우
        if request_info['mode'] not in valid_patch_mode:
            return dict(error_type='4102', error_msg=request_info['mode'])
        # mode가 TEST인데 test_data_path를 요청하지 않은 경우
        if request_info['mode'] == 'TEST':
            if 'test_data_path' not in request_info.keys():
                return dict(error_type='4101', error_msg='test_data_path')
            # test_parameters(target, test_type)를 요청하지 않은 경우 
            # if 'test_parameters' not in request_info.keys():
            #     return dict(error_type='4101', error_msg='test_parameters')
            # if 'target' not in request_info['test_parameters'].keys():
            #     return dict(error_type='4101', error_msg='target')
        return True


class SklearnTrainTask(PrepareModelTrain):

    def _evaluate_cv5(self, estimator, data_set, target_value):
        x_data = data_set.drop([target_value], axis=1)
        y_data = data_set[target_value]
        kfold = KFold(n_splits=5, shuffle=True, random_state=2019)
        cv_scores = cross_val_score(estimator, X=x_data, y=y_data, cv=kfold)
        # cv_scores.means()
        return list(cv_scores)

    def _evaluate_holdout(self, estimator, data_set, target_value):
        x_data = data_set.drop([target_value], axis=1)
        y_data = data_set[target_value]
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.3, random_state=2019)
        clf = estimator.fit(X=x_train, y=y_train)
        y_pred = clf.predict(X=x_test)
        holdout_score = clf.score(x_test, y_test)
        return holdout_score

    def _train_final_model(self, estimator, data_set, target_value):
        x_data = data_set.drop([target_value], axis=1)
        train_columns = list(x_data.columns.values)
        y_data = data_set[target_value]
        final_model = estimator.fit(X=x_data, y=y_data)
        return final_model, train_columns

    def _save_estimator(self, estimator, estimator_name):
        file_path = os.path.join(MODEL_DIR, estimator_name)
        joblib.dump(estimator, file_path)
        return file_path, estimator_name

    # RESTART모드에서 command 검사 
    def check_params(self, params_dict):
        valid_params = {}        
        for param_key, param_value in params_dict.items():
            if isinstance(param_value, str):
                param_value = param_value.lower()
                if param_value == 'true' or param_value == 'false':
                    param_value = bool(util.strtobool(param_value))
                elif param_value == 'None':
                    param_value = None
            valid_params[param_key] = param_value
        return valid_params

    def model_task_result(self, algo_pk, data_path, model_param, train_param, pk):
        if 'preprocessed_data' in data_path:
            data = super()._load_preprocessed_data(
                file_name=os.path.split(data_path)[1])
        elif 'original_data' in data_path:
            data = super()._load_original_data(
                file_name=os.path.split(data_path)[1])
        logger.info('[{}] 경로에서 학습 데이터를 로드했습니다'.format(data_path))

        # 학습에 사용될 알고리즘 import해서 불러오기
        algo = ALGOSerializer(Algorithm.objects.get(pk=algo_pk)).data
        clf = super()._get_estimator(params=algo)
        model_name = type(clf).__name__

        # 사용자 요청에 따라 모델 파라미터 변경
        logger.error('모델 파라미터 {}'.format(model_param))
        if not model_param == None:
            clf = super()._change_params(model=clf, param=model_param)
        logger.info('요청 ID [{}]의 학습 모델 = {}'.format(pk, clf))
        if base.is_regressor(clf) == True or base.is_classifier(clf) == True:
            logger.info('요청 ID [{}]에 의해 모델 [{}]의 교차검증을 수행합니다'\
                .format(pk, model_name))
            cv_score = self._evaluate_cv5(
                estimator=clf, data_set=data, target_value=train_param['y'])
            logger.info('요청 ID [{}]에 의해 모델 [{}]의 홀드아웃 검증을 수행합니다'\
                .format(pk, model_name))
            holdout_score = self._evaluate_holdout(
                estimator=clf, data_set=data, target_value=train_param['y'])
            logger.info('요청 ID [{}]에 의해 모델 [{}]의 최종모델을 학습합니다'\
                .format(pk, model_name))
            final_model, train_columns = self._train_final_model(
                estimator=clf, data_set=data, target_value=train_param['y'])
            file_path, file_name = self._save_estimator(
                estimator=final_model, estimator_name='M_{}.pickle'.format(pk))
        logger.info('요청 ID [{}]의 학습된 모델 저장 => M_{}.pickle'.format(pk, pk))

        model_info = dict(
            model_name=algo['ALGORITHM_NAME'], 
            model_param=clf.get_params(), 
            model_train_columns=train_columns)
        validation_info = dict(
            validation_score=cv_score, 
            holdout_score=holdout_score)

        final_result = dict(
            file_path=file_path, 
            file_name=file_name, 
            model_info=model_info, 
            validation_info=validation_info
            )

        return final_result
