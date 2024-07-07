import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

if __package__:
    from .element import Conv, C3, SPPF, Detect
else:
    from element import Conv, C3, SPPF, Detect

from utils.torch_utils import make_divisible
from utils.autoanchor import check_anchor_order


CFG_SCALE = {
    # [depth, width]
    'n': [0.33, 0.25],
    's': [0.33, 0.50],
    'm': [0.67, 0.75],
    'l': [1.00, 1.00],
    'x': [1.33, 1.25],
}           # 0   1    2    3    4    5    6     7    8     9     10   11   12   13  14   15    16   17
CFG_WIDTH = [64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024, 512, 512, 256, 256, 256, 512, 512, 1024]
CFG_DEPTH = [3, 6, 9, 3, 3, 3, 3, 3]
ANCHORS = [
  [[10, 13], [16, 30], [33, 23]],
  [[30, 61], [62, 45], [59, 119]],
  [[116, 90], [156, 198], [373, 326]],
]


class YOLOv5(nn.Module):
    def __init__(self, scale='n', num_classes=80, in_channels=3):
        super().__init__()
        depth, width = CFG_SCALE[scale]
        ch = [make_divisible(x * width, 8) for x in CFG_WIDTH]
        n = [max(round(x * depth), 1) for x in CFG_DEPTH]
        
        # backbone
        self.layer1 = Conv(in_channels, ch[0], k=6, s=2, p=2, act=True) # P1/2
        self.layer2 = Conv(ch[0], ch[1], k=3, s=2, act=True) # P2/4
        self.layer3 = C3(ch[1], ch[2], n[0])
        self.layer4 = Conv(ch[2], ch[3], k=3, s=2, act=True) # P3/8
        self.layer5 = C3(ch[3], ch[4], n[1])
        self.layer6 = Conv(ch[4], ch[5], k=3, s=2, act=True) # P4/16
        self.layer7 = C3(ch[5], ch[6], n[2])
        self.layer8 = Conv(ch[6], ch[7], k=3, s=2, act=True) # P5/32
        self.layer9 = C3(ch[7], ch[8], n[3])
        self.sppf = SPPF(ch[8], ch[9], 5)

        # head
        self.upsample = nn.Upsample(None, 2, 'nearest')
        self.layer10 = Conv(ch[9], ch[10], k=1, s=1, act=True)
        self.layer11 = C3(ch[10] + ch[6], ch[11], n[4], False)
        self.layer12 = Conv(ch[11], ch[12], k=1, s=1, act=True)
        self.layer13 = C3(ch[12] + ch[4], ch[13], n[5], False) # P3
        self.layer14 = Conv(ch[13], ch[14], k=3, s=2, act=True) 
        self.layer15 = C3(ch[14] + ch[12], ch[15], n[6], False) # P4
        self.layer16 = Conv(ch[15], ch[16], k=3, s=2, act=True)
        self.layer17 = C3(ch[16] + ch[10], ch[17], n[7], False) # P5
        self.head = Detect(nc=num_classes, anchors=ANCHORS, ch=(ch[13], ch[15], ch[17]))

        s = 256
        m = self.head
        m.inplace = True
        m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, 3, s, s))])
        check_anchor_order(m)
        m.anchors /= m.stride.view(-1, 1, 1) # map on grid
        m.bias_init()
        self.initialize_weights()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        C1 = self.layer5(x)
        x = self.layer6(C1)
        C2 = self.layer7(x)
        x = self.layer8(C2)
        x = self.layer9(x)
        x = self.sppf(x)

        P1 = self.layer10(x)
        x = self.upsample(P1)
        x = torch.cat((x, C2), dim=1)
        x = self.layer11(x)
        P2 = self.layer12(x)
        x = self.upsample(P2)
        x = torch.cat((x, C1), dim=1)
        P3 = self.layer13(x) # P3/8-small
        x = self.layer14(P3)
        x = torch.cat((x, P2), dim=1)
        P4 = self.layer15(x) # P4/16-medium
        x = self.layer16(P4)
        x = torch.cat((x, P1), dim=1)
        P5 = self.layer17(x) # P5/32-large
        y = self.head([P3, P4, P5])
        return y

    def initialize_weights(self):
        """Initialize model weights to random values."""
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True


if __name__ == "__main__":
    import torch
    from utils.torch_utils import model_info

    model = YOLOv5(scale='s', num_classes=80)
    # print(model)
    x = torch.randn(1, 3, 640, 640)
    output = model(x)
    for y in output:
        print(y.shape)
    
    model_info(model, input_size=640)
