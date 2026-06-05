import gradio as gr
from ultralytics import YOLO
import torch
import torch.nn as nn
from ultralytics.nn import tasks
from ultralytics.nn.modules import  DWConv  , Conv
import math
import torch.nn.functional as F
import os

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup , reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x

        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        """
        Khởi tạo module ECA.
        :param channels: Số lượng kênh đầu vào (C).
        :param gamma: Tham số trong công thức tính k (thường là 2).
        :param b: Tham số trong công thức tính k (thường là 1).
        """
        super(ECA, self).__init__()
        self.channels = channels
        t = int(abs((math.log(channels, 2) / gamma) + (b / gamma)))
        
        k = t if t % 2 == 1 else t + 1

        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=k, 
            padding=(k - 1) // 2, 
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = x.mean((2, 3), keepdim=True)
        
        y = y.squeeze(-1).transpose(-1, -2)

        y = self.conv(y) 

        y = self.sigmoid(y) 

        y = y.transpose(-1, -2).unsqueeze(-1)

        return x * y 

class DCPBlock(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=1.0):
        """
        DCPBlock với Multi-scale Depthwise Convolution (3x3 và 7x7)
        """
        super().__init__()
        c_ = int(c2 * e) 
        
        self.dw7 = DWConv(c1, c1, k=7, s=1)     
        
        self.dw3 = DWConv(c1, c1, k=3, s=1)   
        
        self.coratt = CoordAtt(c1, c1)  
        
        self.pw = nn.Conv2d(c1, c2, 1)  
        
        self.add = shortcut and c1 == c2

    def forward(self, x):
        x_multi_scale = self.dw7(x) + self.dw3(x)
        
        y = self.pw(self.coratt(x_multi_scale))
        
        return x + y if self.add else y

class NVQ(nn.Module):
    def __init__(self, c1, c2, n=1, residual=True, e=0.5, shortcut=True):
        """
          args :
            c1 : input channels 
            c2 : output channels 
            n : number of c2f block 
            residual : using shortcut connection 
            g :
            e : scale hidden channels 
        """
        super().__init__()
        self.c_ = int(c2 * e) 
        
        self.cv1 = Conv(c1, self.c_, 1, 1)
      
        self.cv2 = Conv((1 + n) * self.c_, c2, 1) 

        self.res_flag = residual and (c1 == c2)

        # self.gamma = nn.Parameter(1.0 * torch.ones(c2), requires_grad=True) if self.res_flag else None
        
        self.m = nn.ModuleList(
            DCPBlock(self.c_, self.c_, shortcut) for _ in range(n)
        )

        self.eca = ECA(channels= c1)

    def forward(self, x):
        y_list = [self.cv1(x)]
        for m in self.m:
            y_list.append(m(y_list[-1]))
            
        out = self.cv2(torch.cat(y_list, 1))
      
        out = self.eca(out)

        # if self.gamma is not None:
        #     return x + self.gamma.view(1, -1, 1, 1) * out
            
        return x + out

tasks.NVQ = NVQ

# 1. Load model 
# Đảm bảo đường dẫn chính xác: 'models/ultimate_model.pt'
try:
    model = YOLO('models/ultimate_model.pt') #
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def predict_banana(image):
    results = model.predict(image , conf=0.4, iou=0.5, max_det=100)
    result = results[0]
    img_with_boxes = result.plot(labels=False)
    return img_with_boxes

demo = gr.Interface(
    fn=predict_banana,
    inputs=gr.Image( label="Tải ảnh buồng chuối"),
    outputs=gr.Image( label="Kết quả nhận diện"),
    title="Banana Bunch Detection",
    examples=[f"images/{f}" for f in os.listdir("images/") if f.endswith(('.jpg', '.png'))] if os.path.exists("images/") else None
)
# 3. Chạy App
if __name__ == "__main__":
    demo.launch()


# cd bunch-banana-detectioncls
# 