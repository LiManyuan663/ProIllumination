import numpy as np
import argparse
from tqdm import tqdm
import yaml
from attrdict import AttrMap
import cv2

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from shadowRemoval.data_manager import TestDataset
from shadowRemoval.utils import gpu_manage, save_image, heatmap
# from untitled1 import Generator
from shadowRemoval.SpA_Former import Generator


def predict(config, args):
    gpu_manage(args)
    dataset = TestDataset(args.filename, config.in_ch, config.out_ch)
    data_loader = DataLoader(dataset=dataset, num_workers=config.threads, batch_size=1, shuffle=False)

    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)

    param = torch.load(args.pretrained)
    gen.load_state_dict(param, False)

    if args.cuda:
        gen = gen.cuda(0)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            x = Variable(batch[0])
            if args.cuda:
                x = x.cuda()

            att, out = gen(x)

            h = 1
            w = 1
            c = 3
            p = config.width
            q = config.height

            allim = np.zeros((h, w, c, p, q))
            out_ = out.cpu().numpy()[0]
            out_rgb = np.clip(out_[:3], 0, 1)

            allim[0, 0, :] = out_rgb * 255
            allim = allim.transpose(0, 3, 1, 4, 2)
            allim = allim.reshape((h * p, w * q, c))
            allim_display = np.clip(allim, 0, 255).astype(np.uint8)
            return allim_display

def predict_remove_shadow(filename):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./shadowRemoval/config.yml')
    parser.add_argument('--filename', type=str, default=filename)
    parser.add_argument('--out_dir', type=str, default='./SpAGAN-FFT-Transformer')
    parser.add_argument('--pretrained', type=str, default='./shadowRemoval/pretrained_models/ISTD/gen_model_epoch_160.pth')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--gpu_ids', type=int, default=[0])
    parser.add_argument('--manualSeed', type=int, default=0)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)
    config = AttrMap(config)
    return predict(config, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,  default='./config.yml')
    parser.add_argument('--test_dir', type=str,  default='./data/ISTDSpANet/A')
    parser.add_argument('--out_dir', type=str,  default='./SpAGAN-FFT-Transformer')
    parser.add_argument('--pretrained', type=str, default='./pretrained_models/ISTD/gen_model_epoch_160.pth')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--gpu_ids', type=int, default=[0])
    parser.add_argument('--manualSeed', type=int, default=0)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)
    config = AttrMap(config)

    predict(config, args)
