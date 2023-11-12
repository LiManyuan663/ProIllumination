import numpy as np
import os

import pydensecrf.densecrf as dcrf


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')


def unnormalization(img):
    img_tmp = np.transpose(img, axes=[1, 2, 0])
    img_tmp *= (0.229, 0.224, 0.225)
    img_tmp += (0.485, 0.456, 0.406)
    img_tmp *= 255.0

    return img_tmp


def get_FPM_FNM(image,mask,premask,labels_dst1,labels_dst2):
    image = np.array(image.data.cpu())
    mask = np.array(mask.data.cpu())
    premask = np.array(premask.data.cpu())
    labels_dst1 = np.array(labels_dst1.data.cpu())
    labels_dst2 = np.array(labels_dst2.data.cpu())
    FPM = []
    FNM = []
    for i in range(len(image)):
        image_i = np.ascontiguousarray(unnormalization(image[i]).astype('uint8'))
        mask_i = mask[i].squeeze().astype('uint8')
        premask_i = (premask[i]*255).squeeze().astype('uint8')
        labels_dst1_i = labels_dst1[i].squeeze().astype('uint8')
        labels_dst2_i = labels_dst2[i].squeeze().astype('uint8')

        premask_i = crf_refine(image_i,premask_i)
        FNM.append(np.bitwise_or(np.bitwise_and(premask_i==0,mask_i>0),labels_dst2_i==1)*1.)
        FPM.append(np.bitwise_or(np.bitwise_and(premask_i>0,mask_i<1),labels_dst1_i==1)*1.)
    FPM = np.array(FPM)
    FNM = np.array(FNM)

    return FPM,FNM