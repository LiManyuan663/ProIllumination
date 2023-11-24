import numpy as np
from torch import nn
import torch
from PIL import Image
from torchvision import transforms

# from misc import crf_refine
from model import DSDNet


class DSDShadowDetection(nn.Module):
    def __init__(self):
        super(DSDShadowDetection, self).__init__()
        self.net = DSDNet()
        self.net.load_state_dict(torch.load('./models/5000_SBU.pth', map_location='cpu'))
        self.img_transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.net.eval()

    def forward(self, input):
        w, h = input.size
        premask = self.net(self.img_transform(input).unsqueeze(0)).squeeze()
        premask = np.array(self.premask_transform(premask, (h, w)))
        # premask = crf_refine(np.array(input), premask)
        return premask

    def premask_transform(self, input, size):
        input = transforms.ToPILImage()(input)
        input = transforms.Resize(size)(input)
        return input


if __name__ == '__main__':
    path = './test/chaomian.jpg'
    input = Image.open(path)
    net = DSDShadowDetection()
    premask = net(input)
    input.show()
    Image.fromarray(premask).show()
