from tkinter import *
from PIL import Image, ImageTk
import os
from draw_super_pixel import ShadowSuperPixel
import numpy as np


class ProIllumination():
    def __init__(self, image_root, args):
        self.master = Tk()
        self.master.geometry("1280x720")
        self.master.title("Label Shadow")

        # 参数
        self.args = args

        # 读入图像的列表和当前索引
        self.image_root = image_root
        self.image_index = -1

        # 超像素分割工具
        self.shadow_mask = ShadowSuperPixel(image_root, args)
        # 分割结果
        self.seg_result = None
        # 标记结果
        self.label_result = None

        # 显示鼠标坐标
        self.xy_text = StringVar()

        self.frame_buttons = Frame(self.master, height=80, width=60, relief=RIDGE, bg='white', bd=5,
                                   borderwidth=4)
        self.frame_buttons.pack(side="top", anchor=N, fill=BOTH, ipady=2, expand=False)


        w = 1280 /2
        h = 720/2
        self.frame_label = Frame(self.master, height=w, width=h, relief=RIDGE, bg='grey', bd=5, borderwidth=4)
        self.frame_label.pack(side="left", anchor=N, fill=BOTH, ipady=2, expand=False)

        self.canves = Canvas(self.frame_label, width=w, height=h)
        self.canves.pack()

        self.frame_label_result = Frame(self.master, height=w, width=h, relief=RIDGE, bg='grey', bd=5,
                                        borderwidth=4)
        self.frame_label_result.pack(side="right", anchor=N, fill=BOTH, ipady=2, expand=False)
        self.canves_result = Canvas(self.frame_label_result, width=w, height=h)
        self.canves_result.pack()

        # 　初始化按钮
        self.init_buttons()

        self.master.mainloop()

    def init_buttons(self):
        read_image_button = Button(self.frame_buttons, text='Read Image', command=self.init_image_seg)
        read_image_button.pack(side="left", anchor=N)

        reload_image_button = Button(self.frame_buttons, text='Reload Image', command=self.reload_image_seg)
        reload_image_button.pack(side="left", anchor=N)

        save_image_button = Button(self.frame_buttons, text='Save ShadowMask', command=self.save_shadow_mask)
        save_image_button.pack(side="left", anchor=N)

        save_image_button = Button(self.frame_buttons, text='Reload Video', command=self.reload_video)
        save_image_button.pack(side="left", anchor=N)

        save_image_button = Button(self.frame_buttons, text='Next Video', command=self.next_video)
        save_image_button.pack(side="left", anchor=N)

        quit_button = Button(self.frame_buttons, text='Exit', command=self.client_exit)
        # quit_button.place(x=740, y=0)
        quit_button.pack(side="right", anchor=N)

        # 坐标名称及显示坐标
        label_xytitle = Label(self.frame_buttons, text='标注位置坐标x,y')
        label_xytitle.pack(side="left", anchor=N)

        label = Label(self.frame_buttons, textvariable=self.xy_text, fg='blue')
        label.pack(side="left", anchor=N)

        # 设置超像素分割的细粒度的滑块
        self.s_super_pixel = Scale(self.frame_buttons, from_=50, to=3000, label='超像素分割细粒度',
                                   length=300,
                                   orient=HORIZONTAL, command=self.set_super_n_segement)
        self.s_super_pixel.set(1500)
        # self.s_super_pixel.place(x=150, y=50)
        self.s_super_pixel.pack(side="left", anchor=N)

    def set_super_n_segement(self, event):
        temp_val = self.s_super_pixel.get()
        self.args['super_seg_count'] = temp_val
        print('super_seg_count is set to {}'.format(self.args['super_seg_count']))

    def init_image_seg(self):
        # while(1):
        self.image_index += 1
        self.init_image()
        # shadow_num = np.sum(self.shadow_mask.mask)
        # unshadow_num = np.sum(1-1*self.shadow_mask.mask)
        # # if shadow_num/unshadow_num>0.001:
        #     break

        self.init_frame_label()
        self.save_shadow_mask()

    def reload_image_seg(self):
        self.init_image()
        self.init_frame_label()
        self.save_shadow_mask()

    def reload_video(self):
        current_video_begin = self.image_root[self.image_index][2]
        self.image_index = current_video_begin
        self.init_image()
        self.init_frame_label()
        self.save_shadow_mask()

    def next_video(self):
        current_video_begin = self.image_root[self.image_index][2]
        current_video_length = self.image_root[self.image_index][3]
        next_video_begin = current_video_begin + current_video_length

        if next_video_begin > len(self.image_root):
            next_video_begin = 0
        self.image_index = next_video_begin
        self.init_image()
        self.init_frame_label()
        self.save_shadow_mask()

    def save_shadow_mask(self):
        self.shadow_mask_result = Image.fromarray((255 * self.shadow_mask.mask).astype('uint8'))
        # self.canves_result.itemconfig(self.canves_result_sample, image=ImageTk.PhotoImage(self.shadow_mask_result))
        splited_name = str(self.image_root[self.image_index][0]).split('\\')
        # splited_name = str(self.image_root[self.image_index][0]).split('/')
        print(splited_name)
        fold_name, base_name = splited_name[-2], splited_name[-1][:-4]
        mask_path = os.path.join('./label', fold_name)

        if not os.path.exists(mask_path):
            try:
                os.mkdir(mask_path)
                print("Create file successes!")
            except:
                print("Create file failed!")
        try:
            self.shadow_mask_result.save(os.path.join(mask_path, base_name + '.png'))
            print("Save {0} OK  !!!".format(base_name))
        except:
            print("Save {0} failed !!!".format(base_name))

    def client_exit(self):
        exit()

    def init_frame_label(self):
        # self.frame_label.bind('<Button-1>', self.onLeftButtonDown)

        self.canves_sample = self.canves.create_image(0, 0, anchor=NW, image=self.seg_result)
        self.canves_result_sample = self.canves_result.create_image(0, 0, anchor=NW, image=self.label_result)
        self.canves.bind('<Button-1>', self.onLeftButtonDown)
        self.canves.bind('<Button-3>', self.onRightButtonDown)
        # self.canves.pack()

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

    # 图像分割初始化函数，需要绘制下一张图片时调用
    def init_image(self):
        print('processing the {}th image'.format(self.image_index))
        self.shadow_mask.forward(self.image_index, self.args)
        self.seg_result = ImageTk.PhotoImage(Image.fromarray(self.shadow_mask.image))
        self.label_result = ImageTk.PhotoImage(Image.fromarray((self.shadow_mask.mask * 255).astype('uint8')))

if __name__ == "__main__":
    image_root = 'data'
    args = {
        # 'mask_from_cnn': True,
        'mask_from_cnn': False,
        'super_seg_count': 1000,
        'super_compactness': 10,
        'super_sigma': 1,

    }
    proIllumination = ProIllumination(image_root, args)