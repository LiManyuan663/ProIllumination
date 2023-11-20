from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from draw_super_pixel import ShadowSuperPixel
import numpy as np

class ProIllumination():
    def __init__(self, args):
        self.master = Tk()
        self.master.geometry("1280x720")
        self.master.title("ProIllumination")

        # 参数
        self.args = args

        # 读入图像的列表和当前索引
        self.image_root = None
        self.image_index = -1

        # 超像素分割工具
        self.shadow_mask = None
        # 分割结果
        self.raw_image = None
        # 标记结果
        self.label_result = None

        # 显示鼠标坐标
        self.xy_text = StringVar()

        # 创建一个用于放置按钮的框架（顶部功能栏）
        self.frame_top = Frame(self.master, bg='white', bd=5, borderwidth=4)
        self.frame_top.pack(side="top", fill=X)

        # 左侧功能栏
        self.frame_buttons = Frame(self.master, width=200, relief=RIDGE, bg='white', bd=5, borderwidth=4)
        self.frame_buttons.pack(side="left", anchor=N, fill=Y, ipadx=2, expand=False)

        # 显示原图的窗口
        self.frame_raw_image = Frame(self.master, relief=RIDGE, bg='grey', bd=5, borderwidth=4)
        self.frame_raw_image.pack(side=LEFT, anchor=N, fill=BOTH, expand=True)

        self.canves = Canvas(self.frame_raw_image)
        self.canves.pack(fill=BOTH, expand=True)

        # 创建 frame_label_result 框架
        self.frame_label_result = Frame(self.master, relief=RIDGE, bg='grey', bd=5, borderwidth=4)
        self.frame_label_result.pack(side=RIGHT, anchor=N, fill=BOTH, expand=True)

        self.canves_result = Canvas(self.frame_label_result)
        self.canves_result.pack(fill=BOTH, expand=True)

        # 初始化按钮
        self.init_buttons()

        self.master.mainloop()

    def init_buttons(self):
        # 创建按钮并放置在顶部功能栏中
        read_image_button = Button(self.frame_top, text='Read Image', command=self.update_raw_image)
        read_image_button.pack(side="left", anchor=N)

        reload_image_button = Button(self.frame_top, text='Reload Image', command=self.reload_image_seg)
        reload_image_button.pack(side="left", anchor=N, padx=10)

        save_image_button = Button(self.frame_top, text='Save Result')
        save_image_button.pack(side="left", anchor=N)

        quit_button = Button(self.frame_top, text='Exit', command=self.client_exit)
        quit_button.pack(side="right", anchor=N)

        # 功能按钮放置在左侧功能栏
        shadow_detect_button = Button(self.frame_buttons, text='Detect Shadow',command = self.detect_shadow)
        shadow_detect_button.pack(side="top", anchor=N, pady=5, fill=X)

        shadow_remove_button = Button(self.frame_buttons, text='Remove Shadow')
        shadow_remove_button.pack(side="top", anchor=N, pady=5, fill=X)

        shadow_interact_button = Button(self.frame_buttons, text='Shadow Interact')
        shadow_interact_button.pack(side="top", anchor=N, pady=5, fill=X)

        illu_predict_button = Button(self.frame_buttons, text='Illumination Predict', command=self.reload_image_seg)
        illu_predict_button.pack(side="top", anchor=N, pady=5, fill=X)

        # save_image_button = Button(self.frame_buttons, text='Reload Video', command=self.reload_video)
        # save_image_button.pack(side="top", anchor=N, pady=5)
        #
        # save_image_button = Button(self.frame_buttons, text='Next Video', command=self.next_video)
        # save_image_button.pack(side="top", anchor=N, pady=5)

        # # 坐标名称及显示坐标
        # label_xytitle = Label(self.frame_buttons, text='标注位置坐标x,y')
        # label_xytitle.pack(side="top", anchor=N)
        #
        # label = Label(self.frame_buttons, textvariable=self.xy_text, fg='blue')
        # label.pack(side="top", anchor=N)
        #
        # # 设置超像素分割的细粒度的滑块
        # self.s_super_pixel = Scale(self.frame_buttons, from_=50, to=3000, label='超像素分割细粒度',
        #                            length=150,
        #                            orient=HORIZONTAL, command=self.set_super_n_segement)
        # self.s_super_pixel.set(1500)
        # self.s_super_pixel.pack(side="top", anchor=N, pady=5)

    # def set_super_n_segement(self, event):
    #     temp_val = self.s_super_pixel.get()
    #     self.args['super_seg_count'] = temp_val
    #     print('super_seg_count is set to {}'.format(self.args['super_seg_count']))

    # def init_image_seg(self):
    #     self.image_index += 1
    #     self.init_image()
    #     self.init_frame_label()
    #     self.save_shadow_mask()
    def update_raw_image(self):
        file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            print("Selected file:", file_path)
            # 使用PIL库加载图像
            self.image_root = file_path
            raw_image = Image.open(file_path)
            # 将图像转换为PhotoImage对象
            raw_image_tk = ImageTk.PhotoImage(raw_image)
            # 更新raw_image变量，以便后续保存
            self.raw_image = raw_image_tk
            self.canves_sample = self.canves.create_image(0, 0, anchor=NW, image=self.raw_image)
            # 更新canves上的图像
            self.canves.itemconfig(self.canves_sample, image=raw_image_tk)
            self.shadow_mask = ShadowSuperPixel(image_root, args)

    def detect_shadow(self):
        self.shadow_mask.forward(self.image_root, self.args) # DSD 似乎仍然不行
        self.canves.bind('<Button-1>', self.onLeftButtonDown)
        self.canves.bind('<Button-3>', self.onRightButtonDown)
        self.label_result = ImageTk.PhotoImage(Image.fromarray((self.shadow_mask.mask * 255).astype('uint8')))
        self.canves_result_sample = self.canves_result.create_image(0, 0, anchor=NW, image=self.label_result)

    def reload_image_seg(self):
        pass
        # self.init_image()
        # self.init_frame_label()
        # self.save_shadow_mask()

    # def reload_video(self):
    #     current_video_begin = self.image_root[self.image_index][2]
    #     self.image_index = current_video_begin
    #     self.init_image()
    #     self.init_frame_label()
    #     self.save_shadow_mask()
    #
    # def next_video(self):
    #     current_video_begin = self.image_root[self.image_index][2]
    #     current_video_length = self.image_root[self.image_index][3]
    #     next_video_begin = current_video_begin + current_video_length
    #
    #     if next_video_begin > len(self.image_root):
    #         next_video_begin = 0
    #     self.image_index = next_video_begin
    #     self.init_image()
    #     self.init_frame_label()
    #     self.save_shadow_mask()

    # def save_shadow_mask(self):
    #     self.shadow_mask_result = Image.fromarray((255 * self.shadow_mask.mask).astype('uint8'))
    #     splited_name = str(self.image_root[self.image_index][0]).split('\\')
    #     fold_name, base_name = splited_name[-2], splited_name[-1][:-4]
    #     mask_path = os.path.join('./label', fold_name)
    #
    #     if not os.path.exists(mask_path):
    #         try:
    #             os.mkdir(mask_path)
    #             print("Create file successes!")
    #         except:
    #             print("Create file failed!")
    #     try:
    #         self.shadow_mask_result.save(os.path.join(mask_path, base_name + '.png'))
    #         print("Save {0} OK  !!!".format(base_name))
    #     except:
    #         print("Save {0} failed !!!".format(base_name))

    def client_exit(self):
        exit()

    # def init_frame_label(self):
    #     self.canves_sample = self.canves.create_image(0, 0, anchor=NW, image=self.seg_result)
    #     self.canves_result_sample = self.canves_result.create_image(0, 0, anchor=NW, image=self.label_result)
    #     self.canves.bind('<Button-1>', self.onLeftButtonDown)
    #     self.canves.bind('<Button-3>', self.onRightButtonDown)
    #
    def onLeftButtonDown(self, event):
        temp_coor = [event.x, event.y]
        self.shadow_mask.add_mask(temp_coor)
        self.seg_result = ImageTk.PhotoImage(self.shadow_mask.draw_maskimg())
        self.canves.itemconfig(self.canves_sample, image=self.seg_result)
        self.label_result = ImageTk.PhotoImage(Image.fromarray((self.shadow_mask.mask * 255).astype('uint8')))
        self.canves_result.itemconfig(self.canves_result_sample, image=self.label_result)
        self.xy_text.set(str(temp_coor))
        print(temp_coor)

    def onRightButtonDown(self, event):
        temp_coor = [event.x, event.y]
        self.shadow_mask.sub_mask(temp_coor)
        self.seg_result = ImageTk.PhotoImage(self.shadow_mask.draw_maskimg())
        self.canves.itemconfig(self.canves_sample, image=self.seg_result)
        self.label_result = ImageTk.PhotoImage(Image.fromarray((self.shadow_mask.mask * 255).astype('uint8')))
        self.canves_result.itemconfig(self.canves_result_sample, image=self.label_result)
        self.xy_text.set(str(temp_coor))
        print(temp_coor)

    # def init_image(self):
    #     print('processing the {}th image'.format(self.image_index))
    #     self.shadow_mask.forward(self.image_index, self.args)
    #     self.seg_result = ImageTk.PhotoImage(Image.fromarray(self.shadow_mask.image))
    #     self.label_result = ImageTk.PhotoImage(Image.fromarray((self.shadow_mask.mask * 255).astype('uint8')))

if __name__ == "__main__":
    image_root = 'data'
    args = {
        'mask_from_cnn': False,
        'super_seg_count': 1000,
        'super_compactness': 10,
        'super_sigma': 1,
    }
    proIllumination = ProIllumination(args)
