from torch import nn
import torch
import torch.nn.functional as F


class ciac_Model(nn.Module):
    """
    CIACNet (Composite Improved Attention Convolutional Network)
    from:
      Liao et al., Frontiers in Neuroscience (2025)
      DOI: 10.3389/fnins.2025.1543508

    Blocks:
      - CV block: dual-branch CNN (CV1 + CV2), multi-scale temporal kernels
      - IAT block: improved CBAM (avg + max + stochastic pooling)
      - TC block: TCN for high-level temporal features
      - Final: concat (IAT features + TC features + CV2 features) -> FC classifier
    """
    def __init__(self, n_ch=16, n_classes=2):
        super().__init__()

        # Paper hyperparams
        D = 2
        drop = 0.3

        # CV1: F1=16, KC1=32, F2=32
        self.cv1 = CVBranch(n_ch=n_ch, F_temp=16, K_temp=32, D=D, F_out=32, K_out=16, pool=8, drop=drop)

        # CV2: F3=32, KC3=64, F4=64
        self.cv2 = CVBranch(n_ch=n_ch, F_temp=32, K_temp=64, D=D, F_out=64, K_out=16, pool=8, drop=drop)

        # IAT operates on CV1 output
        self.iat = IATBlock(channels=32, reduction=8)

        # TC operates on IAT output sequence
        self.tc = TCBlock(c_in=32, c_out=32, k=4, L=2, drop=drop)

        # classifier after multi-level concat
        self.classifier = nn.Sequential(
            nn.LazyLinear(n_classes)
        )

    def forward(self, x):
        # x: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)

        f1 = self.cv1(x)            # (B, 32, 1, T')
        f2 = self.cv2(x)            # (B, 64, 1, T')

        f1a = self.iat(f1)          # (B, 32, 1, T')

        # TC wants (B, C, T)
        tcn_in = f1a.squeeze(2)     # (B, 32, T')
        ft = self.tc(tcn_in)        # (B, 32, T')

        # concat paper-style: [IAT, TC, CV2]
        v_iat = f1a.flatten(1)
        v_tc  = ft.flatten(1)
        v_cv2 = f2.flatten(1)

        feats = torch.cat([v_iat, v_tc, v_cv2], dim=1)
        return self.classifier(feats)



# ------- Temporal Convolution Network (TC block) -------
class TCBlock(nn.Module):
    """
    Paper: L=2 residual blocks, KT=4, filters=32, dropout=0.3
    """
    def __init__(self, c_in=32, c_out=32, k=4, L=2, drop=0.3):
        super().__init__()
        dilations = [1, 2][:L]  # L=2 in paper
        blocks = []
        c_prev = c_in
        for d in dilations:
            blocks.append(TCNResidualBlock(c_prev, c_out, k=k, dilation=d, drop=drop))
            c_prev = c_out
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        # x: (B, C, T)
        return self.net(x)
    
class TCNResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, k=4, dilation=1, drop=0.3):
        super().__init__()
        self.conv1 = CausalConv1d(c_in, c_out, k, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(c_out)

        self.conv2 = CausalConv1d(c_out, c_out, k, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(c_out)

        self.drop = nn.Dropout(drop)
        self.down = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False) if c_in != c_out else None

    def forward(self, x):
        y = self.drop(F.elu(self.bn1(self.conv1(x))))
        y = self.drop(F.elu(self.bn2(self.conv2(y))))
        res = x if self.down is None else self.down(x)
        return F.elu(y + res)
    
class CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, k, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=dilation, bias=False)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CVBranch(nn.Module):
    """
    One branch of CV block (CV1 or CV2) from CIACNet paper.
    Input:  (B, 1, C, T)
    Output: (B, F_out, 1, T')
    """
    def __init__(
        self,
        n_ch: int,
        F_temp: int,
        K_temp: int,
        D: int,
        F_out: int,
        K_out: int = 16,
        pool: int = 8,
        drop: float = 0.3
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F_temp, kernel_size=(1, K_temp), padding=(0, K_temp // 2), bias=False),
            nn.BatchNorm2d(F_temp),
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(F_temp, F_temp * D, kernel_size=(n_ch, 1), groups=F_temp, bias=False),
            nn.BatchNorm2d(F_temp * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool)),
            nn.Dropout(drop),
        )

        # Paper: separable conv replaced with 2D conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(F_temp * D, F_out, kernel_size=(1, K_out), padding=(0, K_out // 2), bias=False),
            nn.BatchNorm2d(F_out),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool)),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.depthwise(x)
        x = self.conv2(x)
        return x

class IATBlock(nn.Module):
    """
    CIACNet "Improved CBAM attention (IAT)" block:
    Channel attention + Spatial attention,
    using avg + max + stochastic pooling.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)

        # channel MLP implemented with 1x1 convs
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

        # spatial attention conv (3 pooled maps -> 1 map)
        self.spatial_conv = nn.Conv2d(3, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, Fmap):
        # ---------- Channel attention ----------
        avg = Fmap.mean(dim=(2, 3), keepdim=True)
        mx  = Fmap.amax(dim=(2, 3), keepdim=True)
        sto = stochastic_pool2d_global(Fmap)

        Mc = torch.sigmoid(self.mlp(avg) + self.mlp(mx) + self.mlp(sto))
        F1 = Fmap * Mc

        # ---------- Spatial attention ----------
        avg_s = F1.mean(dim=1, keepdim=True)
        mx_s  = F1.amax(dim=1, keepdim=True)
        sto_s = stochastic_pool_channel(F1)

        Ms = torch.sigmoid(self.spatial_conv(torch.cat([avg_s, mx_s, sto_s], dim=1)))
        F2 = F1 * Ms

        return F2
    

def stochastic_pool2d_global(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable-ish stochastic pooling approximation:
    weights = softmax over spatial positions
    returns expected value (B, C, 1, 1)
    """
    B, C, H, W = x.shape
    w = torch.softmax(x.view(B, C, -1), dim=-1).view(B, C, H, W)
    return (x * w).sum(dim=(2, 3), keepdim=True)

def stochastic_pool_channel(x: torch.Tensor) -> torch.Tensor:
    """
    Stochastic pooling across channel dim (for spatial attention):
    returns (B, 1, H, W)
    """
    w = torch.softmax(x, dim=1)
    return (x * w).sum(dim=1, keepdim=True)

if __name__ == "__main__":
    model = ciac_Model(n_ch=16, n_classes=2)
    x = torch.randn(8, 16, 1536)  # (B,C,T)
    out = model(x)
    print(out.shape)  # (8, 2)