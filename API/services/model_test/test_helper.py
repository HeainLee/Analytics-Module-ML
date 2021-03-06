import logging
import numbers
import numpy as np
import pandas as pd
from ast import literal_eval
from django.shortcuts import get_object_or_404

from ..utils.custom_decorator import where_exception
from ..data_preprocess.preprocess_base import PreprocessorBase
from ...models.preprocessed_data import PreprocessedData
from ...serializers.serializers import PreprocessedDataSerializer

logger = logging.getLogger("collect_log_helper")


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


class ModelPerformance(PreprocessorBase):
    """
    Return model test result

        Attributes:
        -----------
            y_data (str) : target name of data
                           (get from model query 'COMMAND')
            model_info (dict) : Model query info (eg. TrainModelSerializer)
            test_data_path (str) : real test data saved path
    """

    def __init__(self, model_info, test_data_path, y_data=None):
        self.y_data = y_data
        self.model_info = model_info
        self.test_data_path = test_data_path

    # 모델학습에서 사용한 데이터와 테스트 데이터이 컬럼이 일치하는지 확인하는 함수
    @staticmethod
    def _check_train_columns(data_set, train_summary, target_data=None):
        if target_data is None:
            test_data_columns = list(data_set.columns.values)
            test_data_columns.sort()
            train_data_summary = literal_eval(train_summary)
            train_data_columns = train_data_summary["model_train_columns"]
            train_data_columns.sort()
        else:
            test_data_columns = list(data_set.columns.values)
            test_data_columns.remove(target_data)
            test_data_columns.sort()
            train_data_summary = literal_eval(train_summary)
            train_data_columns = train_data_summary["model_train_columns"]
            train_data_columns.sort()
        if test_data_columns == train_data_columns:
            return True
        else:
            return False

    # Train Data 와 동일한 변환기로 Test Data 에 전처리를 수행하는 함수
    def _test_data_transformer(self, data_set, pdata_summary):
        test_data_columns = list(data_set.columns.values)
        train_pdata_summary = literal_eval(pdata_summary)  # str => list

        # 학습된 데이터의 전처리 정보를 읽어서 차례대로 동일하게 수행하는 코드
        for preprocess_info_dict in train_pdata_summary:
            field_name = preprocess_info_dict["field_name"]
            func_name = preprocess_info_dict["function_name"]
            file_name = preprocess_info_dict["file_name"]
            logger.info(f"[모델 테스트] {func_name} applied to {field_name}")

            if field_name not in test_data_columns:
                return False
            else:
                if func_name == "DropColumns":
                    data_set = super()._drop_columns(data_set, field_name)
                else:
                    transformer = super()._load_pickle(
                        base_path="PREPROCESS_TRANSFORMER_DIR", file_name=file_name
                    )
                    changed_field = transformer.transform(
                        data_set[field_name].values.reshape(-1, 1)
                    )
                    changed_field = super()._to_array(changed_field)

                    # transform 된 데이터와 원본 데이터 통합(NEW) - preprocess_helper.py 참고
                    if len(changed_field.shape) == 2 and changed_field.shape[1] == 1:
                        if func_name == "Normalizer":
                            logger.warning("Not working in this version!!!")
                        else:
                            data_set[field_name] = changed_field
                    elif len(changed_field.shape) == 1:  # LabelEncoder
                        data_set[field_name] = changed_field
                    else:
                        col_name = super()._new_columns(
                            field_name=field_name, after_fitted=changed_field
                        )
                        new_columns = pd.DataFrame(changed_field, columns=col_name)
                        data_set = pd.concat(
                            [data_set, new_columns], axis=1, sort=False
                        )
                        data_set = data_set.drop(field_name, axis=1)
        return data_set

    # 예측값 또는 스코어를 출력하는 함수
    def get_test_result(self, target=None):
        try:
            pk = self.model_info["MODEL_SEQUENCE_PK"]
            if self.test_data_path.endswith(".csv"):
                test_data = pd.read_csv(self.test_data_path)
            elif self.test_data_path.endswith(".json"):
                test_data = pd.read_json(
                    self.test_data_path, lines=True, encoding="utf-8"
                )
            logger.info(f"[모델 테스트] Model ID [{pk}] Data Load!")

            pdata_info = get_object_or_404(
                PreprocessedData, pk=self.model_info["PREPROCESSED_DATA_SEQUENCE_FK2"]
            )
            pdata_serial = PreprocessedDataSerializer(pdata_info).data
            pdata_test = self._test_data_transformer(
                data_set=test_data, pdata_summary=pdata_serial["SUMMARY"]
            )

            if isinstance(pdata_test, bool):  # 오류 발생시 False 반환
                logger.error(f"[모델 테스트] Model ID [{pk}] Check Columns Name")
                return _error_return_dict("4022", "Data is not suitable for the model")
            is_same_columns = self._check_train_columns(
                data_set=pdata_test,
                train_summary=self.model_info["TRAIN_SUMMARY"],
                target_data=target,
            )

            if not is_same_columns:
                logger.error(f"[모델 테스트] Model ID [{pk}] Check Columns Name")
                return _error_return_dict("4022", "Data is not suitable for the model")
            X_ = super()._drop_columns(pdata_test, target)
            y_ = np.array(pdata_test[target]).reshape(-1, 1)
            model_load = super()._load_pickle(
                base_path="MODEL_DIR", file_name=self.model_info["FILENAME"]
            )
            score_ = model_load.score(X=X_, y=y_)
            predict_ = model_load.predict(X=X_)

            if isinstance(predict_[0], numbers.Integral):
                result_response = {"score": "%.3f" % score_, "predict": predict_}
                return result_response
            else:
                result_response = ["%.3f" % elem for elem in predict_]
                result_response = {"score": "%.3f" % score_, "predict": result_response}
                return result_response
        except Exception as e:
            where_exception(error_msg=e)
