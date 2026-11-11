import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import letterbox


class Encoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        def conv_block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.GroupNorm(num_groups=max(1, min(8, o // 8)), num_channels=o),
                nn.ReLU(inplace=True),
            )

        self.c1 = conv_block(3, 32)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = conv_block(32, 64)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = conv_block(64, 128)
        self.p3 = nn.MaxPool2d(2)
        self.c4 = conv_block(128, 192)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(192, embed_dim)

    def forward(self, x):
        x = self.p1(self.c1(x))
        x = self.p2(self.c2(x))
        x = self.p3(self.c3(x))
        feats = self.c4(x)
        emb = self.head(self.pool(feats).flatten(1))
        return F.normalize(emb, dim=1), feats


class SiameseNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.encoder = Encoder(embed_dim)

    def encode(self, x):
        return self.encoder(x)


def distance(e1, e2):
    return (e1 - e2).pow(2).sum(1).clamp(min=1e-7).sqrt()


def contrastive_loss(d, y, margin=1.0):
    return (y * d.pow(2) + (1 - y) * F.relu(margin - d).pow(2)).mean()


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict()}, path)


def load_model(path, device="cpu", embed_dim=128):
    model = SiameseNet(embed_dim).to(device)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


def to_tensor(bgr_img, size=256, device="cpu"):
    box = letterbox(bgr_img, size)
    rgb = cv2g(box)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return t.to(device)


def cv2g(bgr):
    return bgr[:, :, ::-1].copy()