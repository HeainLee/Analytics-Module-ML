import os
import glob
import joblib
import logging
import numbers
import numpy as np
import pandas as pd
from ast import literal_eval
from django.conf import settings

from ..data_preprocess.preprocess_helper import drop_columns

logger = logging.getLogger('collect_log_helper')


class BatchTestResult:

    def __init__(self, batch_service, test_data_path):
        self.batch_manager_id = batch_service['BATCH_SERVICE_SEQUENCE_PK']
        self.model_summary = batch_service['MODEL_SUMMARY']
        self.model_command = batch_service['MODEL_COMMAND']
        self.pdata_summary = batch_service['PREPROCESSED_DATA_SUMMARY']
        self.model_sandbox_pk = batch_service['MODEL_SANDBOX_SEQUENCE_FK1']
        self.trans_sandbox_pk = batch_service['PREPROCESSED_DATA_SANDBOX_SEQUENCE_FK2']

        self.nfs_dir = settings.ANALYTICS_MANAGER_NFS # /ANALYTICS_MANAGER_NFS/batchServer
        self.test_data_path = test_data_path
        self.nfs_batch_info_dir = os.path.join(self.nfs_dir, 'batchService_{}'.format(self.batch_manager_id))
        self.nfs_model_path = os.path.join(self.nfs_batch_info_dir, 'M_{}.pickle'.format(self.model_sandbox_pk))
        self.nfs_trans_path = glob.glob(
            os.path.join(self.nfs_batch_info_dir, 'T_{}_*.pickle'.format(self.trans_sandbox_pk)))

    # 전처리기 로드하는 함수
    def _load_transformer(self, file_name):
        file_path = os.path.join(self.nfs_batch_info_dir, file_name)
        transformer = joblib.load(file_path)
        return transformer

    # 모델 로드하는 함수
    def _load_estimator(self, model_path):
        clf = joblib.load(model_path)
        return clf

    # Train Data와 동일한 변환기로 Test Data에 전처리를 수행하는 함수
    def _test_data_transformer(self, data_set, pdata_summary):
        test_data_columns = list(data_set.columns.values)
        train_pdata_summary = literal_eval(pdata_summary)  # str => list

        # 학습된 데이터의 전처리 정보를 읽어서 차례대로 동일하게 수행하는 코드
        try:
            for preprocess_info_dict in train_pdata_summary:
                field_name = preprocess_info_dict['field_name']
                func_name = preprocess_info_dict['function_name']
                file_name = preprocess_info_dict['file_name']
                logger.info('테스트 데이터 전처리 수행중...')
                logger.info('필드명 [{}] 전처리 기능 [{}]'.format(field_name, func_name))

                if field_name in test_data_columns:
                    if func_name == 'DropColumns':
                        data_set = drop_columns(data_set, field_name)
                    else:
                        transformer = self._load_transformer(file_name=file_name)
                        changed_field = transformer.transform(
                            data_set[field_name].values.reshape(-1, 1))

                        # transfrom된 데이터와 원본 데이터 통합
                        if func_name == 'OneHotEncoder' or func_name == 'KBinsDiscretizer':
                            # 원핫인코딩을 또는 kbins구간화한 경우 통합 방법(새로운 칼럼 생성됨)
                            changed_field = changed_field.toarray() \
                                if not isinstance(changed_field, np.ndarray) else changed_field
                            new_columns = pd.DataFrame(
                                changed_field,
                                columns=[field_name + "_" + str(int(i)) for i in range(changed_field.shape[1])])
                            data_set = pd.concat([data_set, new_columns], axis=1, sort=False)
                            data_set = data_set.drop(field_name, axis=1)
                        else:
                            # 원핫인코딩 또는 kbins구간화 아닌 경우 통합 방법(기존 값을 덮어씀)
                            data_set[field_name] = changed_field
                else:
                    return dict(error_type='field_name', error_msg=field_name)
            return data_set
        except Exception as e:
            logger.error('Error Occured {}'.format(e))

    # 모델학습에서 사용한 데이터와 테스트 데이터이 컬럼이 일치하는지 확인하는 함수
    def _check_train_columns(self, data_set, train_summary, target_data):
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

    # 배치 서비스 요청에 대한 요청 파라미터 검사하는 함수
    def check_request_batch_path(self):
        check_list = [self.test_data_path, self.nfs_model_path, self.nfs_trans_path[0]]
        for check_path in check_list:
            logger.info('경로 확인 중... [{}]'.format(check_path))
            if not os.path.isfile(check_path):
                logger.error('{} 경로가 존재하지 않습니다'.format(check_path))
                return dict(error_type='4004', error_msg=check_path)
        return True

    # 예측값 또는 스코어를 출력하는 함수
    def get_batch_test_result(self):
        try:
            logger.info('배치 서비스 ID 는 [{}]입니다'.format(self.batch_manager_id))

            # 테스트 데이터 로드
            if self.test_data_path.endswith('.csv'):
                test_data = pd.read_csv(self.test_data_path)
            elif self.test_data_path.endswith('.json'):
                test_data = pd.read_json(self.test_data_path, lines=True, encoding='utf-8')
            logger.info('배치 서비스 ID [{}]에 적용할 테스트 데이터를 로드했습니다'.format(self.batch_manager_id))

            # 테스트 데이터 전처리
            pdata_test = self._test_data_transformer(data_set=test_data, pdata_summary=self.pdata_summary)

            if isinstance(pdata_test, dict):  # check_result 타입이 dict이면 에러 메시지를 반환한 것!
                if pdata_test['error_type'] == 'field_name':
                    logger.error('데이터의 컬럼 [{}]은 적용할 수 없습니다'.format(pdata_test['error_msg']))
                    error_response = {'type': 4022, 'detail': 'Data is not suitable for the model'}
                    return error_response
            else:
                logger.info('배치 서비스 ID [{}]의 테스트 데이터 전처리를 수행했습니다'.format(self.batch_manager_id))

            target = literal_eval(self.model_command)['train_parameters']['y']
            is_same_columns = self._check_train_columns(
                data_set=pdata_test,
                train_summary=self.model_summary,
                target_data=target
            )

            if not is_same_columns:
                logger.error(
                    '배치 서비스 ID [{}]의 모델이 학습한 데이터와 데스트 데이터의 컬럼이 일치하지 않습니다' \
                        .format(self.batch_manager_id))
                error_response = {'type': 4022, 'detail': 'Data is not suitable for the model'}
                return error_response

            # 모델 로드
            model_load = self._load_estimator(model_path=self.nfs_model_path)
            logger.info('배치 서비스 ID [{}]의 모델을 로드했습니다'.format(self.batch_manager_id))

            # 모델 테스트 결과
            X_ = drop_columns(pdata_test, target)
            y_ = np.array(pdata_test[target]).reshape(-1, 1)
            score_ = model_load.score(X=X_, y=y_)
            predict_ = model_load.predict(X=X_)
            logger.info('배치 서비스 ID [{}]의 테스트 결과를 반환합니다'.format(self.batch_manager_id))

            if isinstance(predict_[0], numbers.Integral):
                result_response = {'score': '%.3f' % score_, 'predict': predict_}
                return result_response
            else:
                result_response = ['%.3f' % elem for elem in predict_]
                result_response = {'score': '%.3f' % score_, 'predict': result_response}
                return result_response
        except Exception as e:
            logger.error('Error Occured {}'.format(e))
