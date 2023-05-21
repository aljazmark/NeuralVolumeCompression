import torch.nn as nn
import torch
import numpy as np

class UBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, 3, 1, 1,),
            nn.ReLU(),
            nn.Conv3d(ch_out, ch_out, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self,base=4, in_channels=1, out_channels=1,kernel_size=2):
        super(Unet, self).__init__()
        self.first=True
        l1 = base
        l2 = l1*2
        l3 = l2*2
        l4 = l3*2
        self.f1 = True

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.dss = nn.ModuleList()
        self.uss = nn.ModuleList()

        self.dss.append(nn.Conv3d(l1, l1, kernel_size=2, stride=2))
        self.uss.append(nn.ConvTranspose3d(l4,l4, kernel_size=2, stride=2))

        self.dss.append(nn.Conv3d(l2, l2, kernel_size=2, stride=2))
        self.uss.append(nn.ConvTranspose3d(l3, l3, kernel_size=2, stride=2))

        self.dss.append(nn.Conv3d(l3, l3, kernel_size=2, stride=2))
        self.uss.append(nn.ConvTranspose3d(l2, l2, kernel_size=2, stride=2))

        self.dss.append(nn.Conv3d(l4, l4, kernel_size=2, stride=2))
        self.uss.append(nn.ConvTranspose3d(l1, l1, kernel_size=2, stride=2))


        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

        self.down.append(UBlock(in_channels, l1))
        self.down.append(UBlock(l1, l2))
        self.down.append(UBlock(l2, l3))
        self.down.append(UBlock(l3, l4))

        self.up.append(nn.ConvTranspose3d(l4 * 2, l4, kernel_size=2, stride=2))
        self.up.append(UBlock(l4 * 1, l4))
        self.up.append(nn.ConvTranspose3d(l3 * 2, l3, kernel_size=2, stride=2))
        self.up.append(UBlock(l3 * 1, l3))
        self.up.append(nn.ConvTranspose3d(l2 * 2, l2, kernel_size=2, stride=2))
        self.up.append(UBlock(l2 *2, l2))
        self.up.append(nn.ConvTranspose3d(l1 * 2, l1, kernel_size=2, stride=2))
        self.up.append(UBlock(l1*2, l1))

        self.bottom = UBlock(l4, l4 * 2)
        self.final = nn.Conv3d(l1, 1, kernel_size=1)

    def forward(self, x, mode="train", q_sk1=[],q_sk2=[]):
        if self.first and mode == "train":
            print("base shape")
            odim = x.shape
            print(x.shape)

        if mode == "train" or mode == "compress":
            skips = []
            # Downwards pass (encoder), save skip connections
            #"for down in self.down:
            for i in range(len(self.down)):
                x = self.down[i](x)
                if i <1:
                    skips.append(self.dss[i](x))
                    #skips.append(x)
                elif i <2:
                    skips.append(self.dss[i](x))
                else:
                    skips.append(x)
                x = self.pool(x)

        if mode == "compress":
            return torch.quantize_per_tensor(x, scale=0.1, zero_point=128, dtype=torch.quint8), \
                   torch.quantize_per_tensor(skips[0], scale=0.1, zero_point=128, dtype=torch.quint8), \
                   torch.quantize_per_tensor(skips[1], scale=0.1, zero_point=128, dtype=torch.quint8)

        if self.first and mode == "train":
            print("compression:")
            skdims = []
            skdims.append(x.shape)
            for j in skips:
                skdims.append(j.shape)
                print(j.shape)

            print(x.shape)
            print(calcCompression(skdims,odim))
            self.first = False

        # The bottom of the U

        if mode == "extract":
            x = torch.dequantize(x)
            skips = [torch.dequantize(q_sk1),torch.dequantize(q_sk2),torch.empty(0),torch.empty(0)]

        x = self.bottom(x)


        # Reverse skip conections
        skips = skips[::-1]

        # Upwards pass, two steps at a time
        for idx in range(0, len(self.up), 2):
            x = self.up[idx](x)
            id = idx // 2
            skip = skips[id]
            if id == 3:
                concat_skip = torch.cat((self.uss[id](skip), x), dim=1)
            elif id == 2:
                concat_skip = torch.cat((self.uss[id](skip), x), dim=1)
            else:
                concat_skip = x

            x = self.up[idx + 1](concat_skip)


        x = self.final(x)
        return x

def calcCompression(sdims,odim):
    original_size = np.prod(odim)
    compressed_size = 0
    print("using dims")
    for k in range(3):
        print(sdims[k])
        compressed_size += np.prod(sdims[k])
    return compressed_size/original_size
