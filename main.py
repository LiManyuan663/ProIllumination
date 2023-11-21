from window_v2 import ShadowLabeler
from window_v3 import ProIllumination
import os
import sys
from PyQt5.QtWidgets import QApplication

args = {
    'mask_from_cnn': True,
    # 'mask_from_cnn': False,
    'super_seg_count': 1000,
    'super_compactness': 10,
    'super_sigma': 1,

}


def create_window_v2(image_root, args):
    shadow_labler = ShadowLabeler(image_root, args)


def create_window_v3():
    app = QApplication(sys.argv)
    window = ProIllumination(None)
    window.show()
    sys.exit(app.exec_())


def sortImg(img_list):
    img_int_list = [int(f) for f in img_list]
    sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
    return [img_list[i] for i in sort_index]


def read_dataset(dataset_root):
    image_root = []
    input_folder, label_folder, img_ext, label_ext = 'images', 'labels', '.jpg', '.png'
    root = dataset_root
    video_list = os.listdir(os.path.join(root, input_folder))
    num_video_frame = 0
    for vid_id, video in enumerate(video_list):
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, input_folder, video)) if
                    f.endswith(img_ext)]  # no ext
        img_list = sortImg(img_list)
        clip_file_names = []
        for img in img_list:
            # videoImgGt: (img, gt, video start index, video length)
            videoImgGt = (
                os.path.join(root, input_folder, video, img + img_ext),
                os.path.join(root, label_folder, video, img + label_ext),
                num_video_frame,
                len(img_list),
                vid_id
            )
            image_root.append(videoImgGt)
        num_video_frame += len(img_list)
    return image_root


if __name__ == "__main__":
    test_dataset = './data'
    image_root = read_dataset(dataset_root=test_dataset)

    # image_root = ['test/scu.jpg','test/lssd2.jpg']
    create_window_v2(image_root, args)
    # create_window_v3()
