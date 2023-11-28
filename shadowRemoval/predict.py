import os
import sys
import torch
torch.backends.cudnn.benchmark = True
import cv2
import torchvision.transforms as transforms
from PIL import Image
import kornia
from shadowRemoval.model import *
import argparse
import shadowRemoval.options as options
import shadowRemoval.utils as utils

######### parser ###########
# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
# print(dir_name)
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
# print(opt)
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

transform = transforms.Compose(
    [transforms.Resize((256, 256), Image.Resampling.BICUBIC),
    transforms.ToTensor()]
)

def predict_remove_shadow(img_root,mask_root):
    nety = Uformer2(img_size=opt.train_ps, embed_dim=opt.embed_dim, win_size=opt.win_size,
                    token_projection=opt.token_projection, token_mlp=opt.token_mlp).cuda().eval()
    nety.load_state_dict(torch.load('shadowRemoval/output3/net1.pth'))
    netcr = Uformer3(img_size=opt.train_ps, embed_dim=opt.embed_dim, win_size=opt.win_size,
                     token_projection=opt.token_projection, token_mlp=opt.token_mlp).cuda().eval()
    netcr.load_state_dict(torch.load('shadowRemoval/output3/net2.pth'))
    model_restoration = utils.get_arch(opt)
    model_restoration = model_restoration.cuda().eval()
    model_restoration.load_state_dict(torch.load('shadowRemoval/output3/netG_3.pth'))

    with torch.no_grad():
        # input_list = sorted(os.listdir('DataSet/ISTD_Dataset+/test/test_A'))
        # num = len(input_list)

        # for i in range(num):
        # print('Processing image: %s' % i)
        img = Image.open(img_root)
        mask = Image.open(mask_root)
        # label = Image.open('Dataset/ISTD_Dataset+/test/test_C/' + input_list[i])

        img = transform(img).unsqueeze(0).cuda()
        mask = transform(mask).unsqueeze(0).cuda()
        # label = transform(label).unsqueeze(0).cuda()

        a, b, img = torch.split(kornia.color.rgb_to_ycbcr(img), 1, dim=1)
        # q, w, label = torch.split(kornia.color.rgb_to_ycbcr(label), 1, dim=1)

        # img, mask, label, a, b = img.cuda(), mask.cuda(), label.cuda(), a.cuda(), b.cuda()
        img, mask, a, b = img.cuda(), mask.cuda(), a.cuda(), b.cuda()

        input_before = torch.cat((a, mask), 1)
        before = nety(input_before).unsqueeze(0)
        before = torch.clamp(before, 0, 1).detach()

        input_second = torch.cat((before, b, mask), 1)
        second = netcr(input_second).unsqueeze(0)
        second = torch.clamp(second, 0, 1).detach()

        input_ = torch.cat((before, second, img, mask), 1)
        restored = model_restoration(input_).unsqueeze(0)
        restored = torch.clamp(restored, 0, 1)

        res2 = torch.cat((before, second, restored), 1)
        res2 = kornia.color.ycbcr_to_rgb(res2)

        res2 = (res2.cpu().data.numpy().transpose((0, 2, 3, 1)))[0, :, :, :] * 255.
        # res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('shadowRemoval/res_gt/res.png', cv2.cvtColor(res2, cv2.COLOR_RGB2BGR))
        res_display = np.clip(res2, 0, 255).astype(np.uint8)
        return res_display

if __name__ == '__main__':

    nety = Uformer2(img_size=opt.train_ps, embed_dim=opt.embed_dim, win_size=opt.win_size,
                    token_projection=opt.token_projection, token_mlp=opt.token_mlp).cuda().eval()
    nety.load_state_dict(torch.load('output3/net1.pth'))
    netcr = Uformer3(img_size=opt.train_ps, embed_dim=opt.embed_dim, win_size=opt.win_size,
                     token_projection=opt.token_projection, token_mlp=opt.token_mlp).cuda().eval()
    netcr.load_state_dict(torch.load('output3/net2.pth'))
    model_restoration = utils.get_arch(opt)
    model_restoration = model_restoration.cuda().eval()
    model_restoration.load_state_dict(torch.load('output3/netG_3.pth'))

    with torch.no_grad():
        input_list = sorted(os.listdir('DataSet/ISTD_Dataset+/test/test_A'))
        num = len(input_list)


        for i in range(num):
            print('Processing image: %s' % i)
            img = Image.open('Dataset/ISTD_Dataset+/test/test_A/' + input_list[i])
            mask = Image.open('Dataset/ISTD_Dataset+/test/test_B/' + input_list[i])
            # label = Image.open('Dataset/ISTD_Dataset+/test/test_C/' + input_list[i])


            img = transform(img).unsqueeze(0).cuda()
            mask = transform(mask).unsqueeze(0).cuda()
            # label = transform(label).unsqueeze(0).cuda()

            a, b, img = torch.split(kornia.color.rgb_to_ycbcr(img), 1, dim=1)
            # q, w, label = torch.split(kornia.color.rgb_to_ycbcr(label), 1, dim=1)

            # img, mask, label, a, b = img.cuda(), mask.cuda(), label.cuda(), a.cuda(), b.cuda()
            img, mask, a, b = img.cuda(), mask.cuda(), a.cuda(), b.cuda()


            input_before = torch.cat((a, mask), 1)
            before = nety(input_before).unsqueeze(0)
            before = torch.clamp(before, 0, 1).detach()

            input_second = torch.cat((before, b, mask), 1)
            second = netcr(input_second).unsqueeze(0)
            second = torch.clamp(second, 0, 1).detach()

            input_ = torch.cat((before, second, img, mask), 1)
            restored = model_restoration(input_).unsqueeze(0)
            restored = torch.clamp(restored, 0, 1)

            res2 = torch.cat((before, second, restored), 1)
            res2 = kornia.color.ycbcr_to_rgb(res2)



            res2 = (res2.cpu().data.numpy().transpose((0, 2, 3, 1)))[0, :, :, :] * 255.


            cv2.imwrite('res_gt/' + input_list[i], cv2.cvtColor(res2, cv2.COLOR_RGB2BGR))

