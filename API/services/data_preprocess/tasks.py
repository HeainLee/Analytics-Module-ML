'''
celery로 처리할 작업을 정의
스마트시티 분석 모듈에서는 전처리된 데이터를 생성하는 작업을 처리할 때 사용 
@shared_task 데코레이터 = 해당 함수에 대한 요청이 들어오며 작업을 할당
'''
from __future__ import absolute_import
from multiprocessing import current_process

try:
    current_process()._config
except AttributeError:
    current_process()._config = {'semprefix': '/mp'}

import os
import sys
import logging
import datetime
from celery import shared_task
from smartcity.celery import app

from ...serializers.serializers import PreprocessedDataSerializer
from ...models.preprocessed_data import PreprocessedData
from .preprocess_helper import PreprocessTask, DataSummary

logger = logging.getLogger('collect_log_task')


@shared_task(name='preprocess_tasks.transformer_fit', bind=True, ignore_result=False, track_started=True)
def transformer_fit(self, data_saved_path=None, pfunction_info=None, pk=None):
    logger.info('요청 ID [{}]의 전처리 작업이 진행중입니다'.format(pk))
    back_job = make_result(data_saved_path=data_saved_path, pfunction_info=pfunction_info, pk=pk)
    Pdata_info = PreprocessedData.objects.get(pk=pk)
    fit_infomation = {}

    if back_job == False:
        fit_infomation['PROGRESS_STATE'] = 'fail'
        fit_infomation['PROGRESS_END_DATETIME'] = datetime.datetime.now()
        logger.error('요청 ID [{}]의 전처리 작업이 실패했습니다'.format(pk))
        serializer = PreprocessedDataSerializer(Pdata_info, data=fit_infomation, partial=True)

    else:
        fit_infomation['FILEPATH'] = str(back_job['FILEPATH'])  # 생성된 데이터의 위치
        fit_infomation['FILENAME'] = str(back_job['FILENAME'])  # 생성된 데이터의 파일명
        fit_infomation['SUMMARY'] = str(back_job['SUMMARY'])  # 전처리를 수행한 것에 대한 정보 모음
        fit_infomation['COLUMNS'] = back_job['COLUMNS']
        fit_infomation['AMOUNT'] = back_job['AMOUNT']
        fit_infomation['SAMPLE_DATA'] = back_job['SAMPLE_DATA']
        fit_infomation['STATISTICS'] = back_job['STATISTICS']
        fit_infomation['PROGRESS_STATE'] = 'success'
        fit_infomation['PROGRESS_END_DATETIME'] = datetime.datetime.now()
        logger.info('요청 ID [{}]의 전처리 작업이 완료되었습니다'.format(pk))
        serializer = PreprocessedDataSerializer(
            Pdata_info, data=fit_infomation, partial=True)
    if serializer.is_valid():
        serializer.save()
        return 'async_task_finished'
    else:
        logger.error('요청 ID [{}]의 전처리 데이터 저장이 실패했습니다'.format(pk))
        return 'save_failed'


def make_result(data_saved_path=None, pfunction_info=None, pk=None):
    asyn_task = PreprocessTask()
    try:
        data = asyn_task.load_data(file_name=os.path.split(data_saved_path)[1])
        data, preprocessed_data_info_list = asyn_task.task_result(
            data=data, user_request_dict=pfunction_info['request_data'], pk=pk)
        save_data_path, save_data_name = asyn_task.save_preprocessed_data(
            preprocessed_data=data, preprocessed_data_name='P_{}.json'.format(pk))

        data_summary = DataSummary(save_data_path)

        final_result = dict(
            FILEPATH=save_data_path,
            FILENAME=save_data_name,
            SUMMARY=preprocessed_data_info_list,
            COLUMNS=data_summary.columns_info(),
            AMOUNT=data_summary.size_info(),
            SAMPLE_DATA=data_summary.sample_info(),
            STATISTICS=data_summary.statistics_info()
        )
        return final_result
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error('{}, {}, {}'.format(exc_type, fname, exc_tb.tb_lineno))
        logger.error('Error Type = {} / Error Message = {}'.format(type(e), e))
        return False
