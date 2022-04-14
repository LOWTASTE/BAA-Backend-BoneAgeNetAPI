from django.shortcuts import render

# Create your views here.
from django.views.decorators.csrf import csrf_exempt
from rest_framework import generics, permissions, status
from rest_framework.parsers import JSONParser

from BAA.models import BAAModel, BAAModel_Data, Algorithm
from BAA.serializers import *
from rest_framework.views import APIView
from rest_framework.response import Response
from BAA.Predict.PredFunction_inceptionForOCN import PredFunction_inceptionForOCN
import json
from BAA.Train.TrainFunction_inceptionForOCN import TrainFunction_inceptionForOCN
from django.core.cache import cache


class BAAModelAPI(generics.ListCreateAPIView):
    permission_classes = (permissions.AllowAny,)
    queryset = BAAModel.objects.all()
    serializer_class = BAAModelSerializer

    # def post(self, request, *args, **kwargs):
    #     baaModelInfo = json.loads(request.body)
    #
    #     if baaModelInfo.get('algorithm'):
    #         algorithm = baaModelInfo.get('algorithm')
    #         if algorithm == 1:
    #             TrainFunction_inceptionForOCN(baaModelInfo)
    #     return Response(" algorithm not exist ", status=status.HTTP_400_BAD_REQUEST)


class BAAModel_DataAPI(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = (permissions.AllowAny,)
    queryset = BAAModel_Data.objects.all()
    serializer_class = BAAModel_DataSerializer


class AlgorithmAPI(generics.ListCreateAPIView):
    permission_classes = (permissions.AllowAny,)
    queryset = Algorithm.objects.all()
    serializer_class = AlgorithmSerializer


class PredictAPI(APIView):
    permission_classes = (permissions.AllowAny,)
    parser_classes = (JSONParser,)
    """
    预测接口:表单数据
    """

    @staticmethod
    def post(request):
        if request.method == 'POST':
            predictInfo = json.loads(request.body)
            pic_url = predictInfo.get('pic_url')
            model_id = predictInfo.get('model_id')
            print("GenerateResult")
            result = PredFunction_inceptionForOCN(pic_url, model_id)
            if result is None:
                result = 'failed'
                return Response(result)
            result = result[0]
            return Response(result)


class TrainAPI(APIView):
    permission_classes = (permissions.AllowAny,)
    parser_classes = (JSONParser,)
    """
    预测接口:json数据
    """

    @staticmethod
    def post(request):
        baaModelInfo = json.loads(request.body)
        if baaModelInfo.get('algorithm'):
            algorithm = baaModelInfo.get('algorithm')
            if algorithm == 1:
                TrainFunction_inceptionForOCN(baaModelInfo)
                return Response(" Model OK ")
        return Response(" algorithm not exist ", status=status.HTTP_400_BAD_REQUEST)


class TestAPI(APIView):
    permission_classes = (permissions.AllowAny,)
    # parser_classes = (JSONParser,)
    """
    预测接口:表单数据
    """

    @staticmethod
    def post(request):
        # if request.POST:
        # pic_url = request.POST.get('pic_url')
        # model_id = request.POST.get('model_id')
        # print(pic_url)
        # print(model_id)
        # print("GenerateResult")
        # path = BAAModel.objects.get(pk=model_id).path
        # print(path)
        # # result = PredFunction_inceptionForOCN(pic_url, model_id)
        # return Response(path)

        print(cache.set("Test", "Test"))
        print(cache.get("Test"))
        print(cache.get("Nope"))
        return Response("OK")
