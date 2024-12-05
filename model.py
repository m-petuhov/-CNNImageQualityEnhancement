import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = self.conv_block(32, 64)
        self.encoder3 = self.conv_block(64, 128)
        self.encoder4 = self.conv_block(128, 256)

        # Decoder
        self.decoder4 = self.upconv_block2(256, 128)
        self.decoder3 = self.upconv_block(128, 64)
        self.decoder2 = self.upconv_block(64, 32)
        self.decoder1 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv_block2(self, in_channels, out_channels, output_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, output_padding=output_padding),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels, output_padding=0):
        return nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, output_padding=output_padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, kernel_size=2))
        e3 = self.encoder3(F.max_pool2d(e2, kernel_size=2))
        e4 = self.encoder4(F.max_pool2d(e3, kernel_size=2))

        # Decoder
        d4 = self.decoder4(e4)
        d4 = torch.cat((d4, e3), dim=1)
        d3 = self.decoder3(d4)
        d3 = torch.cat((d3, e2), dim=1)
        d2 = self.decoder2(d3)
        d2 = torch.cat((d2, e1), dim=1)
        d1 = self.decoder1(d2)
        # Final Convolution
        out = x + d1
        return out