import cv2
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
import skimage
from illumination.models import models
from illumination.options.train_options import TrainOptions
from illumination.utils.saw_utils import srgb_to_rgb, rgb_to_chromaticity, resize_img_arr, load_img_arr


def decom_single_image(image):
    """

    :param image: ndarry, [0,1], RGB, [w,h,c]
    :return:
    """
    # load model
    opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    model = models.create_model(opt)
    model.switch_to_eval()

    # load image
    saw_img = resize_img_arr(image)

    # prediction
    saw_img = np.transpose(saw_img, (2, 0, 1))
    input_ = torch.from_numpy(saw_img).unsqueeze(0).contiguous().float()
    input_images = Variable(input_.cuda(), requires_grad=False)
    prediction_R, prediction_S = model.netG.forward(input_images)

    '''Visualization'''  # https://github.com/zhengqili/CGIntrinsics/issues/1
    # the outputs of the network actually are logged
    prediction_R = torch.exp(prediction_R.data[0, :, :, :])
    prediction_S = torch.exp(prediction_S.data[0, :, :, :])

    # calc chromaticity
    srgb_img = input_images[0, :, :, :]
    rgb_img = srgb_to_rgb(srgb_img.cpu().numpy().transpose(1, 2, 0))
    rgb_img[rgb_img < 1e-4] = 1e-4
    chromaticity = rgb_to_chromaticity(rgb_img)
    chromaticity = torch.from_numpy(np.transpose(chromaticity, (2, 0, 1))).contiguous().float()

    # actual R
    p_R = torch.mul(chromaticity, prediction_R.cpu())
    p_R_np = p_R.cpu().numpy()
    p_R_np = np.transpose(p_R_np, (1, 2, 0))
    # p_R_np = cv2.cvtColor(p_R_np, cv2.COLOR_BGR2RGB)
    # S
    p_S_np = np.transpose(prediction_S.cpu().numpy(), (1, 2, 0))
    p_S_np = np.squeeze(p_S_np, axis=2)

    return p_R_np, p_S_np


def excu(photo_ids, img_dir):
    # load model
    opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    model = models.create_model(opt)
    model.switch_to_eval()

    count = 0
    total_num_img = len(photo_ids)

    for photo_id in photo_ids:
        print("photo_id ", count, photo_id, total_num_img)

        # load image
        img_path = img_dir + str(photo_id) + ".png"
        saw_img = load_img_arr(img_path)
        saw_img = resize_img_arr(saw_img)

        # prediction
        saw_img = np.transpose(saw_img, (2, 0, 1))
        input_ = torch.from_numpy(saw_img).unsqueeze(0).contiguous().float()
        input_images = Variable(input_.cuda(), requires_grad=False)
        prediction_R, prediction_S = model.netG.forward(input_images)

        '''Visualization'''  # https://github.com/zhengqili/CGIntrinsics/issues/1
        # the outputs of the network actually are logged
        prediction_R = torch.exp(prediction_R.data[0, :, :, :])
        prediction_S = torch.exp(prediction_S.data[0, :, :, :])

        # calc chromaticity
        srgb_img = input_images[0, :, :, :]
        rgb_img = srgb_to_rgb(srgb_img.cpu().numpy().transpose(1, 2, 0))
        rgb_img[rgb_img < 1e-4] = 1e-4
        chromaticity = rgb_to_chromaticity(rgb_img)
        chromaticity = torch.from_numpy(np.transpose(chromaticity, (2, 0, 1))).contiguous().float()

        # actual R
        p_R = torch.mul(chromaticity, prediction_R.cpu())
        p_R_np = p_R.cpu().numpy()
        p_R_np = np.transpose(p_R_np, (1, 2, 0))
        p_R_np = cv2.cvtColor(p_R_np, cv2.COLOR_BGR2RGB)
        # S
        p_S_np = np.transpose(prediction_S.cpu().numpy(), (1, 2, 0))
        p_S_np = np.squeeze(p_S_np, axis=2)

        # show
        cv2.imshow('R ' + str(photo_id), p_R_np)
        cv2.imshow('S ' + str(photo_id), p_S_np)
        np_img = input_images.data[0, :, :, :].cpu().numpy()
        cv2.imshow('I ' + str(photo_id), np_img.transpose(1, 2, 0)[..., ::-1])  # RGB2BGR
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    photo_ids = [1]
    img_dir = "../"

    excu(photo_ids, img_dir)
