## ShadowLabel

### 运行环境
 
Ubuntu：  
Python 3  
`pip install -r requirements.txt`

Win：
推荐用conda配置，  
Python 3.6    
`pip install -r requirements_win.txt`
  
`pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html`  
Pydensecrf   
链接：https://pan.baidu.com/s/1oY4YactjbNrlKKCCuoXrAw 
提取码：umk1    
`pip install ./xxxxxxxx-win_amd64.whl`  

### 阴影检测模型参数文件  
5000.pth    
链接：https://pan.baidu.com/s/1VeGscHhiuI-tLHquZbHnpw 
提取码：rmyi
下载到根目录就可以


### 运行命令

`python main.py`


### To Do list

* 实现 main.py 文件夹下面的 read_dataset 函数:

1. 输入原始数据集路径所在路径;
2. 实现功能图片的读取,利用`shadowDetectModel.py`里面的`DSDShadowDetection`类实现阴影的检测;
3. 利用检测到的阴影计算阴影区域(白色)占全图面积比,小于20则判断此图片为无效数据;
4. 返回当前图片路径为 `python list` 格式.


### 各部分说明

主程序入口: `main.py`  
窗体程序: `window_v2.py`  
超像素分割与绘制: `draw_super_pixel.py`  
阴影检测程序:  
`shadow_detect_model.py`  
`misc.py`  
`resnext`

