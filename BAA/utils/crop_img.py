from BAA.utils.func_utils import GAPAttentionForOne
from tensorflow.keras.models import model_from_json

from django.core.cache import cache


# 生成

def crop_img_default(img_path):
    # if cache.get('loaded_model') is None:
    #     json_file = open('E:/PROJ_ALL/PROJ_PY/Machine-Learning/tutorial/BAA/Models/CUTmodel.json', 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     # print(loaded_model_json)
    #     loaded_model = model_from_json(loaded_model_json)
    #     # load weights into new model
    #     loaded_model.load_weights('E:/PROJ_ALL/PROJ_PY/Machine-Learning/tutorial/BAA/Models/CUTmodel-60epoch.h5')
    #     weights = loaded_model.layers[-1].get_weights()[0]
    #     print("Loaded model from disk")
    #     cache.set('loaded_model', loaded_model)
    #     cache.set('weights', weights)
    # return GAPAttentionForOne(cache.get('loaded_model'), cache.get('weights'), img_path)

    json_file = open('E:/PROJ_ALL/PROJ_PY/Machine-Learning/tutorial/BAA/Models/CUTmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # print(loaded_model_json)
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('E:/PROJ_ALL/PROJ_PY/Machine-Learning/tutorial/BAA/Models/CUTmodel-60epoch.h5')
    weights = loaded_model.layers[-1].get_weights()[0]
    print("Loaded model from disk")
    # cache.set('loaded_model', loaded_model)
    # cache.set('weights', weights)

    return GAPAttentionForOne(loaded_model, weights, img_path)
