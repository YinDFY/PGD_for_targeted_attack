# -*- coding: utf-8 -*-
import cv2
import time
import pickle
import os
import shutil
from numpy import *
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import Pretrained_FR_Models.irse as irse
import Pretrained_FR_Models.facenet as facenet
import Pretrained_FR_Models.ir152 as ir152
device = "cuda:3"
def cos_simi(emb_before_pasted, emb_target_img):
    """
    :param emb_before_pasted: feature embedding for the generated adv-makeup face images
    :param emb_target_img: feature embedding for the victim target image
    :return: cosine similarity between two face embeddings
    """
    return torch.mean(torch.sum(torch.mul(emb_target_img, emb_before_pasted), dim=1)
                      / emb_target_img.norm(dim=1) / emb_before_pasted.norm(dim=1))

def cal_target_loss(before_pasted, target_img, model_name,input_size):
    """
    :param before_pasted: generated adv-makeup face images
    :param target_img: victim target image
    :param model_name: FR model for embedding calculation
    :return: cosine distance between two face images
    """


    # Obtain FR model
    fr_model =model_name

    before_pasted_resize = F.interpolate(before_pasted, size=input_size, mode='bilinear')
    target_img_resize = F.interpolate(target_img, size=input_size, mode='bilinear')

    # Inference to get face embeddings
    emb_before_pasted = fr_model(before_pasted_resize)
    emb_target_img = fr_model(target_img_resize).detach()

    # Cosine loss computing
    cos_loss = 1 - cos_simi(emb_before_pasted, emb_target_img)

    return cos_loss


def PGD_Attack(img_before,target,model_name,eps=0.3,alpha = 2/255,iters = 40):
    input_size = (112, 112)
    fr_model = ir152.IR_152((112, 112))
    if model_name == 'ir152':
        input_size = (112, 112)
        fr_model = ir152.IR_152((112, 112))
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/ir152.pth'))
    elif model_name == 'irse50':
        input_size = (112, 112)
        fr_model = irse.Backbone(50, 0.6, 'ir_se')
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/irse50.pth'))
    elif model_name == 'mobile_face':
        input_size = (112, 112)
        fr_model = irse.MobileFaceNet(512)
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/mobile_face.pth'))
    elif model_name == 'facenet':
        input_size = (160, 160)
        fr_model = facenet.InceptionResnetV1(num_classes=8631, device='cuda')
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/facenet.pth'))
    fr_model.to(device)
    fr_model.eval()
    img_before_resize = F.interpolate(img_before, size=input_size, mode='bilinear')
    target_resize = F.interpolate(target, size=input_size, mode='bilinear')
    ori_images = img_before_resize.data
    for i in range(iters):
        img_before_resize.requires_grad = True
        output_attacker = fr_model(img_before_resize)
        output_target = fr_model(target_resize)
        loss = 1 - cos_simi(output_target,output_attacker)
        loss.backward()
        adv_images = img_before_resize + alpha*img_before_resize.grad.sign()
        eta = torch.clamp(adv_images - ori_images,min = -eps,max = eps)
        img_before = torch.clamp(ori_images + eta ,min = 0,max = 1).detach()
    return img_before

def preprocess(im, mean, std, device):
    mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=device).view(1, -1, 1, 1)
    im = (im - mean) / std
    return im


if __name__ == "__main__":
    path = './Datasets_Makeup/before_aligned_600'
    target_img =cv2.imread("./Datasets_Makeup/target_aligned_600/Camilla_Parker_Bowles_0002.jpg")
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB) / 255
    target_img = torch.from_numpy(target_img).permute(2, 0, 1).to(torch.float32).to(device)
    target_img = preprocess(target_img, 0.5, 0.5, device)
    output_dir = "./facenet/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    i = 0
    for root,dirs,files in os.walk(path,topdown=True):
        for name in files:
            file_path = os.path.join(root,name)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
            img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32).to(device)
            img = preprocess(img, 0.5, 0.5, device)
            attacked_img = PGD_Attack(img,target_img,"facenet")
            print(name)
            torchvision.utils.save_image(attacked_img / 2 + 0.5, './facenet/'+str(i) + '.png', nrow=1)
            i = i +1