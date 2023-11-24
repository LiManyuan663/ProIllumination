from skimage.segmentation import slic, mark_boundaries
from skimage.color import rgb2gray, rgb2lab
from skimage import io, filters
import numpy as np
from PIL import Image
from shadow_detect_model import DSDShadowDetection


class ShadowSuperPixel():
    def __init__(self, image_path, args):
        self.DSD = DSDShadowDetection()
        self.image_path = image_path
        self.mask_from_cnn = args['mask_from_cnn']

    def forward(self, image_index, args):
        self.image = io.imread(image_index)
        self.mask = self.DSD(Image.fromarray(self.image)) == 255
        if len(self.mask.shape) > 2:
            self.seg = slic(rgb2lab(self.image), args['super_seg_count'], args['super_compactness'])
        else:
            self.seg = slic(self.image, args['super_seg_count'], args['super_compactness'])
        self._full_mask()

    def forward_2(self, image_id, args):
        self.image = io.imread(self.image_path[image_id][0])
        if self.mask_from_cnn:
            self.mask = self.DSD(Image.fromarray(self.image)) == 255
        else:
            self.mask = io.imread(self.image_path[image_id][1], as_gray=True)
            thre = filters.threshold_otsu(self.mask)
            self.mask = self.mask > thre
        if len(self.mask.shape) > 2:
            self.seg = slic(rgb2lab(self.image), args['super_seg_count'], args['super_compactness'])
        else:
            self.seg = slic(self.image, args['super_seg_count'], args['super_compactness'])
        self._full_mask()

    def _full_mask(self):
        unique_seg = np.unique(self.seg)
        for i in range(len(unique_seg)):
            sub_seg = self.seg == unique_seg[i]
            shadow_num = np.sum(np.bitwise_and(self.mask, sub_seg))
            unshadow_num = np.sum(sub_seg) - shadow_num
            if shadow_num > unshadow_num:
                self.mask = np.bitwise_or(
                    self.mask,
                    sub_seg
                )
            else:
                self.mask = np.bitwise_and(
                    self.mask,
                    np.bitwise_not(sub_seg)
                )

    def add_mask(self, coor):
        # 添加阴影
        segvalue = self.seg[coor[1], coor[0]]
        self.mask = np.bitwise_or(
            self.mask,
            np.equal(self.seg, segvalue)
        )

    def sub_mask(self, coor):
        # 消除阴影
        segvalue = self.seg[coor[1], coor[0]]
        self.mask = np.bitwise_and(
            self.mask,
            np.bitwise_not(np.equal(self.seg, segvalue))
        )

    def draw_segimg(self):
        # 返回带有分割边界的图像
        '''with return'''
        imgWithBoundary = (mark_boundaries(self.image, self.seg) * 255).astype('uint8')
        return Image.fromarray(imgWithBoundary)

    def draw_maskimg(self):
        # 返回带有阴影模板的图像
        '''with return'''
        if len(self.image.shape) == 3:
            imgWithMask = self.image + (np.stack([self.mask, self.mask, self.mask], axis=-1) * 50).astype('uint8')
        else:
            imgWithMask = self.image + (self.mask * 50).astype('uint8')
        return Image.fromarray(imgWithMask)


def super_pixel(path, cord):
    shadowmask = ShadowSuperPixel(path, path)
    shadowmask.calseg(100)
    seg_ori = shadowmask.drawbundary()
    io.imshow(seg_ori)
    io.show()

    shadowmask.labelmask((100, 100))
    seg_label = shadowmask.drawbundary()
    io.imshow(seg_label)
    io.show()

    shadowmask.labelmask((100, 200))
    seg = shadowmask.drawbundary()
    io.imshow(seg)
    io.show()


if __name__ == "__main__":
    # super_pixel('test/lssd2.jpg',1)
    print(np.bitwise_and(False, False))
