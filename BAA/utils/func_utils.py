# from visualization import *
import numpy as np
import cv2
from tensorflow.keras import backend as K
# from keras.preprocessing import image
import os
from BAA.utils.crop_patches import find_max_component, crop
import uuid

def ShowAttentionV1(model, image_path):
    file_list = os.listdir(image_path)
    file_list.sort()
    for filename in file_list:
        print(filename)
        filepath = image_path + filename
        image = load_image(filepath)
        image = image / 255.0
        gender = 1.0
        gender = np.asarray(gender)
        gender = np.expand_dims(gender, axis=0)
        layer = K.function([model.layers[0].input], [model.layers[196].output])
        FeatureMap = layer([image, gender])[0]
        print(FeatureMap.shape)
        FeatureMap = np.squeeze(FeatureMap, axis=0)
        FeatureMap = np.abs(FeatureMap)
        heatmap = np.mean(FeatureMap, axis=2)
        heatmap = heatmap / np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        print(heatmap.shape)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        SaveImg(filename, filepath, heatmap)
    print('********** Done ***********')


def GAPAttention(model, weights, image_path):
    file_list = os.listdir(image_path)
    file_list.sort()
    for filename in file_list:
        filepath = image_path + filename
        print(filepath)
        image = load_image(filepath)
        image = image / 255.0
        gender = 1.0
        gender = np.asarray(gender)
        gender = np.expand_dims(gender, axis=0)
        layer = K.function([model.layers[0].input], [model.layers[1].get_output_at(-1), model.layers[-1].output])
        GAP, prediction = layer([image, gender])
        GAP = np.squeeze(GAP, axis=0)
        print(GAP.shape)
        index = np.argmax(prediction)
        print(index)
        # weight = weights[:,index]
        weight = np.mean(weights[:, index - 5:index + 5], axis=1)
        heatmap = np.zeros((GAP.shape[0], GAP.shape[1]))
        for k in range(GAP.shape[2]):
            heatmap = heatmap + weight[k] * GAP[:, :, k]
        heatmap = heatmap / np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        print(heatmap.shape)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        SaveImg(filename, filepath, heatmap)
    print('********** All Done ***********')


def SaveImg(filename, filepath, heatmap):
    img = cv2.imread(filepath)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    AttentionImg = 0.5 * heatmap + img
    cv2.imwrite('E:/Data/60epoch/heatmap/' + filename, heatmap)
    cv2.imwrite('E:/Data/60epoch/attentionImg/' + filename, AttentionImg)
    # cv2.imwrite('E:/Data/Test/heatmap/' + filename, heatmap)
    # cv2.imwrite('E:/Data/Test/attentionImg/' + filename, AttentionImg)
    print("* save done *")


def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (300, 300))
    x = np.asarray(img, dtype=np.float32)
    # img = image.load_img(path, target_size=(448, 448))
    # print (img.shape)
    # x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


def softlabel(label, num_class):
    softlabel = np.zeros((len(label), num_class))
    ratio = 1.0 / 50
    for i in range(len(label)):
        for j in range(num_class):
            softlabel[i, j] = 1.0 - ratio * np.abs(j - label[i])
    softlabel = np.maximum(softlabel, 0)
    return softlabel


def GaussLabel(label, num_class):
    sigma = 15.0
    GaussLabel = np.zeros((len(label), num_class))
    x = np.array(range(num_class)) + 1
    for k in range(len(label)):
        GaussLabel[k, :] = np.exp(-(x - label[k]) ** 2 / (2.0 * sigma ** 2))
    return GaussLabel


def TestMAE(model, test_data, test_label, test_gender):
    test_gender = np.array(test_gender)
    test_gender = np.expand_dims(test_gender, axis=1)
    layer = K.function([model.layers[0].input, model.layers[3].input], [model.layers[-1].output])
    predictions = layer([test_data, test_gender])
    predictions = np.array(predictions)
    predictions = np.squeeze(predictions, axis=0)
    print(predictions.shape)
    predict_label = np.argmax(predictions, axis=1)
    test_label = np.argmax(test_label, axis=1)
    print(predict_label)
    print(test_label)
    TestMAE = np.mean(np.abs(predict_label - test_label))
    return TestMAE


def DataAugment(x_train):
    x_train_Aug = np.zeros(x_train.shape)
    for i in range(x_train.shape[0]):
        for j in range(3):
            img = x_train[i, :, :, j]
            img = RandomMask(img)
            img = RandomMask(img)
            if np.random.random() > -1:
                x_train_Aug[i, :, :, j] = img
            else:
                x_train_Aug[i, :, :, j] = x_train[i, :, :, j]
    return x_train_Aug


def RandomMask(img):
    m, n = img.shape
    m = int(m / 6)
    n = int(n / 6)
    i, j = np.random.randint(0, 6, 2)
    img[i * m:(i + 1) * m, j * n:(j + 1) * n] = np.random.random()
    return img


def GAPAttentionForOne(model, weights, image_path):
    # filepath = image_path + filename
    # 生成heatmap
    print(image_path)
    image = load_image(image_path)
    image = image / 255.0
    gender = 1.0
    gender = np.asarray(gender)
    gender = np.expand_dims(gender, axis=0)
    layer = K.function([model.layers[0].input], [model.layers[1].get_output_at(-1), model.layers[-1].output])
    GAP, prediction = layer([image, gender])
    GAP = np.squeeze(GAP, axis=0)
    index = np.argmax(prediction)
    weight = np.mean(weights[:, index - 5:index + 5], axis=1)
    heatmap = np.zeros((GAP.shape[0], GAP.shape[1]))
    for k in range(GAP.shape[2]):
        heatmap = heatmap + weight[k] * GAP[:, :, k]
    heatmap = heatmap / np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # SaveImg(filename, filepath, heatmap)

    # 整合
    image_tmp = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (image_tmp.shape[1], image_tmp.shape[0]))
    AttentionImg = 0.5 * heatmap + image_tmp

    # 生成UUID保存heatmap
    heatmap_path = 'E:/PROJ_ALL/PROJ_PY/Machine-Learning/tutorial/BAA/temp/' + str(uuid.uuid1()) + '.png'
    cv2.imwrite(heatmap_path, heatmap)

    gray_img = cv2.imread(image_path, 0)
    # gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    gray_heatmap = cv2.imread(heatmap_path, 0)

    # 通过heatmap生成mask
    ret, mask = cv2.threshold(gray_heatmap, 40, 255, cv2.THRESH_BINARY)

    # mask 如果为0矩阵直接保留原图
    if np.count_nonzero(mask) > 1:
        print("find_max_component")
        mask = find_max_component(mask)
    else:
        print("img")
        mask = gray_img

    # 根据mask裁剪出图片
    croped_img = crop(gray_img, mask)
    os.remove(heatmap_path)
    print('&&&&&&&&remove done&&&&&&&&')
    return croped_img
