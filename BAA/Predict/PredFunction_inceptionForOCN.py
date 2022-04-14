from PIL import Image
import torchvision.transforms as T
from torch.autograd import Variable as V
import torch
from BAA.Nets.incepForOCN import inception_v3
from BAA.models import BAAModel
from BAA.utils.crop_img import crop_img_default
import numpy as np
import os

# 原版
# base_bone_dir = os.path.join('..', 'input', 'rsna-bone-age')
# dataPath = os.path.join(base_bone_dir, 'boneage-training-dataset', 'boneage-training-dataset')
# labelPath = os.path.join(base_bone_dir, 'boneage-training-dataset.csv')


def PredFunction_inceptionForOCN(pic_url, model_id):
    model = inception_v3(pretrained=False, aux_logits=False)
    model.train(False)
    # 找出预测模型所在路径并加载
    path = BAAModel.objects.get(pk=model_id).path
    print(path)
    print('model loading')
    # path = 'E:\\PROJ_ALL\\PROJ_PY\\Machine-Learning\\tutorial\\BAA\\Models\\model.pth'
    model.load_state_dict(torch.load(path))
    print('load model finished')

    # print(crop_img_default(pic_url))
    # 注意力图处理传入的图片
    pil_img = Image.fromarray(np.uint8(crop_img_default(pic_url)))
    # pil_img = pil_img.convert("L")
    print('cut pic finished')

    # # input_pic = '../../dataSet/Bone+Age+Training+Set/boneage-training-dataset/1377.png'
    # input_pic = pic_url
    # pil_img = Image.open(input_pic)

    # array = np.asarray(pil_img)
    # data = t.from_numpy(array)
    transform2 = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize([0.5, ], [0.5, ])
    ])
    data = transform2(pil_img)
    data = V(data.cuda())
    data = V(torch.unsqueeze(data, dim=0).float(), requires_grad=False)
    model = model.cuda()
    print(data.shape)
    output = model(data)
    output_left = output[:, :77]
    output_mid = output[:, 77:154]
    output_right = output[:, 154:]
    pred_left = output_left.cpu().data.max(1, keepdim=True)[1].cuda()
    pred_mid = output_mid.cpu().data.max(1, keepdim=True)[1].cuda()
    pred_right = output_right.cpu().data.max(1, keepdim=True)[1].cuda()
    pred_left = pred_left.cpu().float().numpy()
    pred_mid = pred_mid.cpu().float().numpy()
    pred_right = pred_right.cpu().float().numpy()
    pred_age = pred_left + pred_right + pred_mid + 1
    return pred_age
