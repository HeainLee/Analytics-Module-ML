#데이터 적용 테스트에서 필요한 함수 (train_model_view.py)
#모델이 전처리된 데이터로 학습된경우, SUMMARY 정보를 받아와서 동일하게 테스트 데이터도 전처리를 수행하는 함수

import os
import joblib
import logging
import numbers
import numpy as np
import pandas as pd 
from ast import literal_eval 
from django.shortcuts import get_object_or_404

from ..data_preprocess.preprocess_base import drop_columns
from ...models.preprocessed_data import PreprocessedData
from ...serializers.serializers import PreprocessedDataSerializer
from ...config.result_path_config import PATH_CONFIG

logger = logging.getLogger('collect_log_helper')

PREPROCESS_TRANSFORMER_DIR = PATH_CONFIG.PREPROCESS_TRANSFORMER # 'result/preprocess_transformer'
MODEL_DIR = PATH_CONFIG.MODEL  # 'result/model'


class ModelPerformance:

    def __init__(self, model_info, test_data_path, y_data=None):

        self.y_data = y_data
        self.model_info = model_info
        self.test_data_path = test_data_path
        # self.model_session = model_session

    # 전처리기 로드하는 함수
    def _load_transformer(self, file_name):
        file_path = os.path.join(PREPROCESS_TRANSFORMER_DIR, file_name)
        transformer = joblib.load(file_path)
        return transformer

    # 모델 로드하는 함수
    def _load_estimator(self, model_name):
        file_path = os.path.join(MODEL_DIR, model_name)
        clf = joblib.load(file_path)
        return clf

    # Train Data와 동일한 변환기로 Test Data에 전처리를 수행하는 함수
    def _test_data_transformer(self, data_set, pdata_summary):
        test_data_columns = list(data_set.columns.values)
        train_pdata_summary = literal_eval(pdata_summary) # str => list

        # 학습된 데이터의 전처리 정보를 읽어서 차례대로 동일하게 수행하는 코드
        for preprocess_info_dict in train_pdata_summary:
            field_name = preprocess_info_dict['field_name']
            func_name = preprocess_info_dict['function_name']
            function_pk = preprocess_info_dict['function_pk']
            file_name = preprocess_info_dict['file_name']
            logger.info('테스트 데이터 전처리 수행중...')
            logger.info('필드명 [{}] 전처리 기능 [{}]'.format(field_name, func_name))

            if field_name in test_data_columns:
                if func_name == 'DropColumns':
                    data_set = drop_columns(data_set, field_name)
                else:
                    transformer = self._load_transformer(file_name=file_name)
                    changed_field = transformer.transform(
                        data_set[field_name].values.reshape(-1,1))

                    # transfrom된 데이터와 원본 데이터 통합
                    if func_name=='OneHotEncoder' or func_name=='KBinsDiscretizer':
                        # 원핫인코딩을 또는 kbins구간화한 경우 통합 방법(새로운 칼럼 생성됨)
                        changed_field = changed_field.toarray()\
                        if not isinstance(changed_field, np.ndarray) else changed_field
                        new_columns = pd.DataFrame(
                            changed_field, columns = [field_name+"_"+str(int(i)) \
                            for i in range(changed_field.shape[1])])
                        data_set = pd.concat([data_set, new_columns], axis=1, sort=False)
                        data_set = data_set.drop(field_name, axis=1)
                    else: 
                        # 원핫인코딩 또는 kbins구간화 아닌 경우 통합 방법(기존 값을 덮어씀)
                        data_set[field_name] = changed_field
            else:
                return False
        return data_set

    #모델학습에서 사용한 데이터와 테스트 데이터이 컬럼이 일치하는지 확인하는 함수
    def _check_train_columns(self, data_set, train_summary, target_data=None):
        if target_data == None:
            test_data_columns = list(data_set.columns.values)
            test_data_columns.sort()
            train_data_summary = literal_eval(train_summary)
            train_data_columns = train_data_summary['model_train_columns']
            train_data_columns.sort()
            if test_data_columns == train_data_columns:
                return True
            else:
                logger.info('테스트 데이터 컬럼명 {}'.format(test_data_columns))
                logger.info('모델이 학습한 데이터 컬럼명 {}'.format(train_data_columns))
                return False
        else:
            test_data_columns = list(data_set.columns.values)
            test_data_columns.remove(target_data)
            test_data_columns.sort()
            train_data_summary = literal_eval(train_summary)
            train_data_columns = train_data_summary['model_train_columns']
            train_data_columns.sort()
            if test_data_columns == train_data_columns:
                return True
            else:
                logger.info('테스트 데이터 컬럼명 {}'.format(test_data_columns))
                logger.info('모델이 학습한 데이터 컬럼명 {}'.format(train_data_columns))
                return False            
    
    # 예측값 또는 스코어를 출력하는 함수
    def get_test_result(self, target=None):
        try:
            pk = self.model_info['MODEL_SEQUENCE_PK']
            if self.test_data_path.endswith('.csv'):
                test_data = pd.read_csv(self.test_data_path)
            elif self.test_data_path.endswith('.json'):
                test_data = pd.read_json(self.test_data_path, lines=True, encoding='utf-8')
            logger.info('모델 ID [{}]에 적용할 테스트 데이터를 로드했습니다'.format(pk))

            pdata_info = get_object_or_404(
                PreprocessedData, pk=self.model_info['PREPROCESSED_DATA_SEQUENCE_FK2'])
            pdata_serial = PreprocessedDataSerializer(pdata_info).data
            pdata_test = self._test_data_transformer(
                data_set=test_data, pdata_summary=pdata_serial['SUMMARY'])

            if isinstance(pdata_test, bool): # 오류 발생시 False 반환
                logger.error(
                    '테스트 데이터 컬럼에 모델 ID [{}]가 학습한 컬럼이 존재하지 않습니다'.format(pk))
                error_response = {'type': 4022, 'detail':'Data is not suitable for the model'}
                return error_response
            else:
                logger.info(
                    '테스트 데이터에 모델 ID [{}]가 학습한 데이터와 동일한 전처리를 수행합니다'.format(pk))

            is_same_columns = self._check_train_columns(
                data_set=pdata_test,
                train_summary=self.model_info['TRAIN_SUMMARY'],
                target_data=target
                )

            if not is_same_columns:
                logger.error(
                    '모델 ID [{}]가 학습한 데이터와 데스트 데이터의 컬럼이 일치하지 않습니다'.format(pk))
                error_response = {'type': 4022, 'detail':'Data is not suitable for the model'}
                return error_response

            logger.info('{}'.format(list(pdata_test.columns)))
            X_ = drop_columns(pdata_test, target)
            y_ = np.array(pdata_test[target]).reshape(-1,1)
            model_load = self._load_estimator(model_name=self.model_info['FILENAME'])
            score_ = model_load.score(X=X_, y=y_)
            predict_ = model_load.predict(X=X_)
            if isinstance(predict_[0], numbers.Integral):
                result_response = {'score':'%.3f' % score_, 'predict':predict_}
                return result_response
            else:
                result_response = [ '%.3f' % elem for elem in predict_]
                result_response = {'score':'%.3f' % score_, 'predict':result_response}
                return result_response
        except Exception as e:
            logger.error('Error Occured {}'.format(e))
