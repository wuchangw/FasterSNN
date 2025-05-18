import torch
import torch.nn as nn
import torch.nn.functional as F

class LIFNeuronLayer(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.membrane_potential = None
        self.spike_count = 0

    def reset(self):
        self.membrane_potential = None
        self.spike_count = 0

    def forward(self, x):
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros(x.size(0), x.size(1), device=x.device)

        self.membrane_potential = self.membrane_potential * self.decay + x.mean(dim=(2, 3, 4))
        spikes = (self.membrane_potential >= self.threshold).float()
        self.spike_count += spikes.sum().item()
        self.membrane_potential = self.membrane_potential * (1 - spikes)
        return spikes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

class FasterBlock3d(nn.Module):
    def __init__(self, in_channels, expansion=2, spatial_div=3):
        super().__init__()
        self.hidden_channels = in_channels * expansion
        self.spatial_div = spatial_div

        self.conv_center = nn.Conv3d(
            in_channels, self.hidden_channels,
            kernel_size=3, padding=1
        )

        self.conv_edge = nn.Sequential(
            nn.Conv3d(
                in_channels, in_channels,
                kernel_size=3, padding=1,
                groups=in_channels
            ),
            nn.Conv3d(
                in_channels, self.hidden_channels,
                kernel_size=1
            )
        )

        self.bn = nn.BatchNorm3d(self.hidden_channels)
        self.reduce = nn.Conv3d(self.hidden_channels, in_channels, 3, padding=1)
        self.lif = LIFNeuronLayer()

    def forward(self, x):
        B, C, H, W, D = x.shape
        split = self.spatial_div

        h_start = H // split
        h_end = H - h_start
        w_start = W // split
        w_end = W - w_start
        d_start = D // split
        d_end = D - d_start

        center_feat = self.conv_center(x)
        edge_feat = self.conv_edge(x)

        mask = torch.zeros_like(x)
        mask[:, :, h_start:h_end, w_start:w_end, d_start:d_end] = 1
        mask = mask.bool()
        mask = mask[:, :1]
        mask = mask.expand(-1, self.hidden_channels, -1, -1, -1)

        combined = torch.where(mask, center_feat, edge_feat)
        out = self.bn(combined)
        out = self.reduce(out)
        return x + self.lif(out)

class SWA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(in_channels, in_channels // 4, 1),
            LIFNeuronLayer(),
            nn.Conv3d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        self.spat_att = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 4, 1),
            LIFNeuronLayer(),
            nn.Conv3d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        c_att = self.channel_att(x)
        s_att = self.spat_att(x)
        att = self.alpha * c_att + self.beta * s_att
        return x * att

class FasterSNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, time_steps=4):
        super().__init__()
        self.time_steps = time_steps
        self.att_weights = nn.ParameterDict({
            'block1': nn.Parameter(torch.ones(1)),
            'block2': nn.Parameter(torch.ones(1)),
            'block3': nn.Parameter(torch.ones(1))
        })

        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, stride=2, padding=1),
            FasterBlock3d(64),
            nn.GELU(),
            SWA(64)
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            FasterBlock3d(128),
            nn.GELU(),
            SWA(128)
        )

        self.block3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            FasterBlock3d(256),
            nn.GELU(),
            SWA(256)
        )

        self.block4 = nn.Sequential(
            nn.Conv3d(256, 512, 3, stride=2, padding=1),
            FasterBlock3d(512),
            nn.GELU(),
        )

        self.add_layer4 = nn.Sequential(nn.Conv3d(512, 64, 1))
        self.add_layer3 = nn.Sequential(nn.Conv3d(256, 64, 1))
        self.add_layer2 = nn.Sequential(nn.Conv3d(128, 64, 1))
        self.add_layer1 = nn.Sequential(nn.Conv3d(64, 64, 1))
        self.pool = nn.MaxPool3d(2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        B, T = x.shape[0], x.shape[1]
        all_outputs = []

        for m in self.modules():
            if isinstance(m, LIFNeuronLayer):
                m.reset()

        for t in range(T):
            x_t = x[:, t].unsqueeze(1)
            b1 = self.block1(x_t)
            b2 = self.block2(b1)
            b3 = self.block3(b2)
            b4 = self.block4(b3)

            b1 = self.pool(self.add_layer1(b1)) * self.att_weights['block1']
            b2 = self.pool(self.add_layer2(b2) + b1) * self.att_weights['block2']
            b3 = self.pool(self.add_layer3(b3) + b2) * self.att_weights['block3']
            b4 = self.pool(self.add_layer4(b4) + b3)

            output = self.classifier(b4)
            all_outputs.append(output)

        return torch.stack(all_outputs, dim=1).mean(dim=1)