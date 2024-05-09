# -*- coding: utf-8 -*-
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn.functional as F
import Pretrained_FR_Models.irse as irse
import Pretrained_FR_Models.facenet as facenet
import Pretrained_FR_Models.ir152 as ir152
import arcface_torch.backbones.iresnet as arc
from utils import *

"""
if torch.cuda.is_available():
    device = torch.device("cuda:4")         
else:
    device = torch.device("cpu")          
"""
output_dir_list = ["./ir152/", "./facenet/", "./ir152/", "./irse50/"]
model_list = [["irse50", "facenet", "mobile_face"], ["irse50", "mobile_face", "ir152"], ["irse50", "mobile_face", "facenet"],
              ["ir152", "mobile_face", "facenet"]]


def cos_simi(emb_before_pasted, emb_target_img):
    """
    :param emb_before_pasted: feature embedding for the generated adv-makeup face images
    :param emb_target_img: feature embedding for the victim target image
    :return: cosine similarity between two face embeddings
    """
    return torch.mean(torch.sum(torch.mul(emb_target_img, emb_before_pasted), dim=1)
                      / emb_target_img.norm(dim=1) / emb_before_pasted.norm(dim=1))


def cal_target_loss(before_pasted, target_img, model_name):
    fr_model = None
    input_size = None
    if model_name == 'ir152':
        input_size = (112, 112)
        fr_model = ir152.IR_152((112, 112))
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/ir152.pth'))
    if model_name == 'irse50':
        input_size = (112, 112)
        fr_model = irse.Backbone(50, 0.6, 'ir_se')
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/irse50.pth'))
    if model_name == 'mobile_face':
        input_size = (112, 112)
        fr_model = irse.MobileFaceNet(512)
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/mobile_face.pth'))
    if model_name == 'facenet':
        input_size = (160, 160)
        fr_model = facenet.InceptionResnetV1(num_classes=8631, device=device)
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/facenet.pth'))
    if model_name == "arc":
        input_size = (112, 112)
        fr_model = arc.iresnet100()
        fr_model.load_state_dict(torch.load("./arcface_torch/pretrained_model/arcface.pth"))
    if model_name == "cosface":
        input_size = (112, 112)
        fr_model = arc.iresnet100()
        fr_model.load_state_dict(torch.load("./arcface_torch/pretrained_model/cosface.pth"))
    fr_model.to(device)
    fr_model.eval()

    before_pasted_resize = F.interpolate(before_pasted, size=input_size, mode='bilinear')
    target_img_resize = F.interpolate(target_img, size=input_size, mode='bilinear')

    # Inference to get face embeddings
    emb_before_pasted = fr_model(before_pasted_resize)
    emb_target_img = fr_model(target_img_resize).detach()

    # Cosine loss computing
    cos_loss = cos_simi(emb_before_pasted, emb_target_img)
    # cos_loss.requires_grad = True
    return cos_loss


def PGD_Attack(img_before, target, model_names, eps=0.02, alpha=0.5 / 255, iters=5):
    img_adv = img_before.clone().detach().requires_grad_(True)
    for i in range(iters):
        # Zero out the gradients
        img_adv.requires_grad = True

        # Calculate losses for each model
        total_loss = 0
        for model_name in model_names:
            cos_loss = cal_target_loss(img_adv, target, model_name)
            total_loss += cos_loss

        # Backward and gradient calculation
        total_loss.backward()

        # Update the image
        grad = img_adv.grad.detach().sign()
        img_adv = img_adv + alpha * grad
        img_adv = img_before + torch.clamp(img_adv - img_before, min=-eps, max=eps)
        img_adv = img_adv.detach()
        img_adv = torch.clamp(img_adv, 0,1)

        print(total_loss.data)

    return img_adv


avg_ssim = 0.0
avg_psnr = 0.0
if __name__ == "__main__":
    path = './before_aligned_600'
    target_img = cv2.imread("./target_aligned_600/Camilla_Parker_Bowles_0002.jpg") / 255.0
    target_img = torch.from_numpy(target_img).permute(2, 0, 1).to(torch.float32).to(device).unsqueeze(0)
    ssim_list = []
    psnr_list = []

    i = 0
    th_dict = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
               'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878),
               'cosface': (0.18, 0.28, 1), 'arc': (0.18, 0.28, 1)}
    sim_dict = {}
    sim_dict["cosface"] = []
    sim_dict["facenet"] = []
    sim_dict["arc"] = []
    sim_dict["mobile_face"] = []
    sim_dict["ir152"] = []
    sim_dict["irse50"] = []
    for root, dirs, files in os.walk(path, topdown=True):
        for index in range(1):
            output_dir = output_dir_list[index]
            print("Black Attack:", output_dir.split("/")[1])
            m = model_list[index]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for name in files:
                file_path = os.path.join(root, name)
                img = cv2.imread(file_path) / 255.0
                print(name)
                img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32).to(device).unsqueeze(0)

                attacked_img = PGD_Attack(img, target_img, m)

                for model in ["irse50", "facenet", "ir152", "mobile_face"]:
                    cos = cal_target_loss(attacked_img, target_img, model)
                    sim_dict[model].append(cos.item())

                sim = calculate_ssim(img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0,
                                     attacked_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0)
                psr = calculate_psnr(img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0,
                                     attacked_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0)
                print(sim,psr)
                avg_ssim = avg_ssim + sim
                avg_psnr = avg_psnr + psr
                img_np = attacked_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0
                cv2.imwrite(output_dir + str(i) + '.png', img_np)
                i = i + 1
    for key, values in sim_dict.items():
        th01, th001, th0001 = th_dict[key]
        total = len(values)
        success01 = 0
        success001 = 0
        success0001 = 0
        for v in values:
            if v > th01:
                success01 += 1
            if v > th001:
                success001 += 1
            if v > th0001:
                success0001 += 1
        if total != 0:
            print(key, " attack success(far@0.1) rate: ", success01 / total)
            print(key, " attack success(far@0.01) rate: ", success001 / total)
            print(key, " attack success(far@0.001) rate: ", success0001 / total)
    print("SSIM:", avg_ssim / i)

    print("PSRN:", avg_psnr / i)
