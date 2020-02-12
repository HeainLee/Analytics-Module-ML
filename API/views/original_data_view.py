#################[데이터 원본(학습데이터) 관리]#################
import os
import shutil
import logging
from django.conf import settings
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from ..models.original_data import OriginalData
from ..models.preprocess_functions import PreprocessFunction
from ..serializers.serializers import OriginalDataSerializer
from ..services.data_preprocess.preprocess_helper import DataSummary, TestPreprocess
from ..services.utils.custom_response import CustomErrorCode
from ..config.result_path_config import PATH_CONFIG

logger = logging.getLogger('collect_log_view')
error_code = CustomErrorCode()


class OriginalDataView(APIView):
    def post(self, request):
        base_directory = settings.NIFI_RESULT_DIRECTORY  # /home/centos/NIFI_RESULT
        data_save_path = PATH_CONFIG.ORIGINAL_DATA_DIRECTORY  # 'result/original_data'
        if 'data_path' not in request.data.keys():
            return Response(error_code.MANDATORY_PARAMETER_MISSING_4101(error_msg='data_path'),
                            status=status.HTTP_400_BAD_REQUEST)

        request_data_path = os.path.join(base_directory, request.data['data_path'])
        if os.path.isfile(request_data_path):
            file_name = os.path.split(request_data_path)[1]
            file_ext = os.path.splitext(file_name)[1][1:]
        else:
            return Response(error_code.FILE_NOT_FOUND_4004(path_info=request_data_path),
                            status=status.HTTP_404_NOT_FOUND)

        query = OriginalData.objects.all()
        if query.exists():
            get_pk_new = OriginalData.objects.latest('ORIGINAL_DATA_SEQUENCE_PK').pk + 1
        else:
            get_pk_new = 1

        save_file_name = os.path.join(data_save_path, 'O_{}.{}'.format(get_pk_new, file_ext))
        save_file_name = shutil.copy(request_data_path, save_file_name)
        data_summary = DataSummary(save_file_name)

        data_info = dict(
            NAME=os.path.splitext(file_name)[0],
            FILEPATH=save_file_name,
            FILENAME=os.path.splitext(os.path.split(save_file_name)[1])[0],
            EXTENSION=file_ext,
            COLUMNS=data_summary.columns_info(),
            STATISTICS=data_summary.statistics_info(),
            SAMPLE_DATA=data_summary.sample_info(),
            AMOUNT=data_summary.size_info(),
        )

        serializer = OriginalDataSerializer(data=data_info)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        queryset = OriginalData.objects.all().order_by('ORIGINAL_DATA_SEQUENCE_PK')
        serializer = OriginalDataSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class OriginalDataDetailView(APIView):
    def get(self, request, pk):
        origin_data = get_object_or_404(OriginalData, pk=pk)
        serializer = OriginalDataSerializer(origin_data)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def patch(self, request, pk):
        # 지정한 원본데이터에 요청한 전처리를 수행하고 json으로 넘겨줌
        origin_data = get_object_or_404(OriginalData, pk=pk)
        serializer = OriginalDataSerializer(origin_data).data
        data_path = serializer['FILEPATH']
        if not os.path.isfile(data_path):
            logger.error('{} 경로가 존재하지 않습니다'.format(data_path))
            return Response(error_code.FILE_NOT_FOUND_4004(path_info=data_path))

        test_preprocess = TestPreprocess()
        user_request_data = test_preprocess.load_data(file_name=os.path.split(data_path)[1])

        check_result = test_preprocess.check_request_body_original_patch(
            request_info=request.data, request_data=user_request_data, pk=pk)
        if not check_result:
            error_type = check_result['error_type']
            error_msg = check_result['error_msg']
            if error_type == '4004':
                get_object_or_404(PreprocessFunction, pk=error_msg)
            elif error_type == '4102':
                return Response(error_code.INVALID_PARAMETER_TYPE_4102(error_msg),
                                status=status.HTTP_400_BAD_REQUEST)
            elif error_type == '4101':
                return Response(error_code.MANDATORY_PARAMETER_MISSING_4101(error_msg),
                                status=status.HTTP_400_BAD_REQUEST)

        # 전처리 테스트 수행 (요청을 하나씩 처리해서 한꺼번에 결과 반환)
        logger.info('요청 ID [{}]의 전처리 테스트를 시작합니다'.format(pk))
        RESULT_LIST = test_preprocess.test_result(
            data=user_request_data, user_request_dict=request.data['request_test'], pk=pk)

        # {"error_name":"PreprocessTestError", "error_detail":field_name+','+transformer_name}
        ## 전처리 기능 fit/transform하는 도중 발생한 에러(test_transformer)
        if type(RESULT_LIST) == dict and 'error_name' in RESULT_LIST.keys():
            if RESULT_LIST['error_name'] == 'PreprocessTestError':
                return Response(error_code.INVALID_PREPROCESS_CONDITION_4104(
                    error_msg=RESULT_LIST['error_detail']),
                    status=status.HTTP_400_BAD_REQUEST)
            elif RESULT_LIST['error_name'] == 'ParameterSyntaxError':
                return Response(error_code.INVALID_PARAMETER_TYPE_4102(
                    error_msg=RESULT_LIST['error_detail']),
                    status=status.HTTP_400_BAD_REQUEST)

        logger.info('요청 ID [{}]의 전처리 테스트가 완료되었습니다'.format(pk))
        return Response(RESULT_LIST, status=status.HTTP_200_OK)

    def delete(self, request, pk):
        # DELETE_FLAG == True로 전환하고, 저장된 파일 삭제 (db의 instance는 삭제하지 않음)
        # 이미 DELETE_FLAG가 True인 경우, Conflict(4009) 에러 반환
        origin_data = get_object_or_404(OriginalData, pk=pk)
        serializer = OriginalDataSerializer(origin_data)
        if serializer.data['DELETE_FLAG']:
            return Response(error_code.CONFLICT_4009(mode='DELETE', error_msg='deleted'),
                            status=status.HTTP_409_CONFLICT)
        else:
            if os.path.isfile(serializer.data['FILEPATH']):
                os.remove(serializer.data['FILEPATH'])
                serializer = OriginalDataSerializer(
                    origin_data, data=dict(DELETE_FLAG=True), partial=True)
                if serializer.is_valid():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_200_OK)
            else:
                return Response(error_code.FILE_NOT_FOUND_4004(
                    path_info=serializer.data['FILEPATH']),
                    status=status.HTTP_404_NOT_FOUND)
