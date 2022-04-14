from django.contrib.auth.models import User
from rest_framework import serializers
from BAA.models import BAAModel, BAAModel_Data, Algorithm
# from BAA.DTO.models import BAAModelInfo


class BAAModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = BAAModel
        fields = ('id', 'created', 'epoch', 'name', 'accuracy_rate', 'algorithm', 'path', 'description')


class BAAModel_DataSerializer(serializers.ModelSerializer):
    class Meta:
        model = BAAModel_Data
        fields = ('id', 'model', 'pic_url')


class AlgorithmSerializer(serializers.ModelSerializer):
    class Meta:
        model = Algorithm
        fields = ('id', 'name')


# class BAAModelInfoSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = BAAModelInfo
#         fields = ('epoch', 'name', 'algorithm', 'path', 'train_pic_list', 'train_labels', 'train_male_list')
