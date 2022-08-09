import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
# from .model import lrelu

def updown(x, size, mode='bicubic'):
    out = F.interpolate(x, size=size, mode=mode, align_corners=True)
    return out

class Fusion_Block(nn.Module):
    def __init__(self, channels):
        super(Fusion_Block, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
        )

        self.Fusion_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=channels*4, out_channels=channels*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=channels*2, out_channels=channels*2, kernel_size=3, stride=1, padding=1),
            Attention(channels*2),
        )

    def forward(self, pan, ms, fusion):
        pan_layer = self.input_layer(pan)
        ms_layer = self.input_layer(ms)
        pan_1 = pan + pan_layer
        ms_1 = ms + ms_layer
        fusion_input = torch.cat([ms, pan, fusion], dim=1)
        out_fea1 = self.Fusion_layer1(fusion_input)

        out = fusion + out_fea1
        return pan_1, ms_1, out

class Attention(nn.Module):  # CSSA
    def __init__(self, channels, ratio=2):
        super(Attention, self).__init__()
        self.channel = Channel_attention(channels, ratio)
        self.spatial = Spatial_attention()
        self.spectral = Spectial_attention()

    def forward(self, x):
        out = self.channel(x)
        out = self.spatial(out)
        out = self.spectral(out)

        return out

class Channel_attention(nn.Module):
    def __init__(self, channels, ratio):
        super(Channel_attention, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(channels, channels//ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channels//ratio, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        avg_x = self.avg_pooling(x).view(b, c)
        max_x = self.max_pooling(x).view(b, c)
        v = self.fc_layers(avg_x) + self.fc_layers(max_x)
        v = self.sigmoid(v).view(b, c, 1, 1)
        return x * v

class Spatial_attention(nn.Module):
    def __init__(self):
        super(Spatial_attention, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        self.Conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pooling(x, dim=1, keepdim=True)
        max_x, _ = self.max_pooling(x, dim=1, keepdim=True)
        v = self.Conv(torch.cat((max_x, avg_x), dim=1))
        v = self.sigmoid(v)
        return x * v

class Spectial_attention(nn.Module):
    def __init__(self):
        super(Spectial_attention, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        self.Conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pooling(x, dim=1, keepdim=True)
        max_x, _ = self.max_pooling(x, dim=1, keepdim=True)
        v = self.Conv(torch.cat((max_x, avg_x), dim=1))
        v = self.sigmoid(v)
        return x * v

class Dense_Block(nn.Module):
    def __init__(self):
        super(Dense_Block, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 16, 3, 1, 1),
            nn.ReLU(True),
            nn.PixelShuffle(2),
            nn.Conv2d(4, 16, 3, 1, 1),
            nn.ReLU(True),
        )
        self.layers = nn.ModuleDict({
            'DenseConv1': nn.Conv2d(16, 16, 3, 1, 1),
            'DenseConv2': nn.Conv2d(32, 16, 3, 1, 1),
            'DenseConv3': nn.Conv2d(48, 16, 3, 1, 1),
        })
        self.postConv = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(16, 4, 3, 1, 1),
        )

    def forward(self, pan, ms):
        input = torch.cat([ms, pan], 1)
        x = self.layer1(input)
        for i in range(len(self.layers)):
            out = self.layers['DenseConv'+str(i+1)](x)
            x = torch.cat([x, out], 1)
        out = self.postConv(x)
        return out

class edge_enhance_multi(nn.Module):
    def __init__(self, channels, num_of_layers):
        super(edge_enhance_multi, self).__init__()

        self.up1 = Dense_Block()
        self.up2 = Dense_Block()

        self.pan_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=3, stride=1, padding=1)
        )
        self.ms_layer = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=channels, kernel_size=3, stride=1, padding=1)
        )
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=channels*2, kernel_size=3, stride=1, padding=1)
        )
        self.recon_layer = nn.Sequential(
            nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # nn.Conv2d(channels, 4, 1, 1, padding=0),
            nn.Conv2d(channels, 4, 3, 1, padding=1),
        )
        self.blocklists = nn.ModuleList([Fusion_Block(channels=channels) for i in range(num_of_layers)])

    def forward(self, pan, lr):
        _, N, H, W = lr.shape
        pan_4 = updown(pan, (H, W))
        pan_2 = updown(pan, (H * 2, W * 2))
        ms_2 = updown(lr, (H*2, W*2))
        ms_4 = updown(lr, (H*4, W*4))
        pan_blur = kornia.filters.GaussianBlur2d((11, 11), (1, 1))(pan)
        pan_2_blur = kornia.filters.GaussianBlur2d((11, 11), (1, 1))(pan_2)
        pan_hp = pan - pan_blur
        pan_2_hp = pan_2 - pan_2_blur

        lr_2 = self.up1(pan_4, lr) + ms_2 + pan_2_hp

        lr_u = self.up2(pan_2, lr_2) + ms_4 + pan_hp

        pan_pre = self.pan_layer(pan)
        ms_pre = self.ms_layer(lr_u)
        fusion_pre = self.fusion_layer(torch.cat([lr_u, pan], dim=1))
        for i in range(len(self.blocklists)):
            pan_pre, ms_pre, fusion_pre = self.blocklists[i](pan_pre, ms_pre, fusion_pre)
        out = self.recon_layer(fusion_pre)
        out = torch.sigmoid(out)
        return out

    def test(self, device='cuda:0'):
        total_params = sum(p.numel() for p in self.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        input_ms = torch.rand(1, 4, 64, 64)
        input_pan = torch.rand(1, 1, 256, 256)

        import torchsummaryX
        torchsummaryX.summary(self, input_pan.to(device), input_ms.to(device))

if __name__ == "__main__":

    net = edge_enhance_multi(channels=32, num_of_layers=8).cuda()
    net.test()






