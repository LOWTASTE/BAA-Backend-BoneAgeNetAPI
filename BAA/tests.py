# # import json
# #
# # from django.test import TestCase
# #
# # # Create your tests here.
# # from BAA.Train.TrainFunction_inceptionForOCN import TrainFunction_inceptionForOCN
# #
# # baaModel = {
# #     "epoch": 2,
# #     "train_pic_list": [
# #         "Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/1380.png",
# #         "Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/1377.png",
# #         "Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/1382.png",
# #         "Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/1383.png",
# #         "Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/1384.png",
# #         "Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/1385.png",
# #         "Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/1387.png",
# #         "Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/1389.png",
# #         "Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/1390.png",
# #         "Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/1391.png",
# #         "Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/1394.png",
# #         "Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/1408.png",
# #         "Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/1409.png"
# #     ],
# #     "train_labels": [
# #         120,
# #         180,
# #         138,
# #         150,
# #         156,
# #         36,
# #         138,
# #         138,
# #         156,
# #         180,
# #         57,
# #         126,
# #         149
# #     ],
# #     "train_male_list": [
# #         "True",
# #         "False",
# #         "True",
# #         "True",
# #         "True",
# #         "True",
# #         "True",
# #         "True",
# #         "True",
# #         "True",
# #         "True",
# #         "True",
# #         "True"
# #     ],
# #     "algorithm": 1
# # }
# #
# # j = json.dumps(baaModel)
# # print(baaModel)
# # TrainFunction_inceptionForOCN(baaModel)
#
#
# # # MLP for Pima Indians Dataset Serialize to JSON and HDF5
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.models import model_from_json
# import numpy
# import os
#
# # load json and create model
#
# from utils import func_utils
#
# # 加载权重与模型
# json_file = open('CUTmodel.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# # print(loaded_model_json)
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights('CUTmodel-60epoch.h5')
# print("Loaded model from disk")
#
#
# weights = loaded_model.layers[-1].get_weights()[0]
# func_utils.GAPAttention(loaded_model, weights, 'Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/')
#

# from PIL import Image
#
# from utils.func_utils import load_image
#
# input_pic = 'Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/1385.png'
# # input_pic = pic_url
# pil_img = Image.open(input_pic)
# print(pil_img)
# print("sss"* 10)
# print(load_image(input_pic))

# from BAA.utils.crop_img import crop_img_default
# from PIL import Image
# import numpy as np
# print("传入格式")
# input_pic = 'Q:/Proj_Python/DataSet/Bone+Age+Training+Set/boneage-training-dataset/1385.png'
# input = Image.open(input_pic)
# print(input)
# print("剪切后格式")
# cut_array = crop_img_default(input_pic)
# print(cut_array)
# print("转换格式")
# pil_img = Image.fromarray(np.uint8(cut_array))
# print(pil_img)
# print("正确预测格式")
# currect = Image.open("E:/Data/60epoch/hand/1385.png")
# print(currect)
# # print('cut pic finished')

from django.core.cache import cache
# from django_redis import get_redis_connection  # 视图中连接并操作
# conn = get_redis_connection("default")
test = 520
# conn.
print(cache.set('Test', test))
print(cache.get('Test'))
print(cache.get('Nope'))

