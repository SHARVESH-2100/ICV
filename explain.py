import numpy as np
import torch
import torch.nn.functional as F
import cv2


def gradcam(model, ref_t, query_t):
    model.eval()
    query_t.requires_grad_(True)
    with torch.no_grad():
        emb_r, _ = model.encode(ref_t)
    emb_q, feats_q = model.encode(query_t)
    score = (emb_q - emb_r).pow(2).sum()
    grads = torch.autograd.grad(score, feats_q)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats_q).sum(dim=1, keepdim=True))
    cam = F.interpolate(
        cam, size=(ref_t.shape[2], ref_t.shape[3]), mode="bilinear", align_corners=False
    )
    cam = cam[0, 0].detach().cpu().numpy()
    cam = cam - cam.min()
    span = cam.max()
    if span > 1e-8:
        cam = cam / span
    return cam


def overlay_heatmap(bgr, heat, alpha=0.5):
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    h, w = bgr.shape[:2]
    if colored.shape[:2] != (h, w):
        colored = cv2.resize(colored, (w, h))
    return np.clip(bgr.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha, 0, 255).astype(np.uint8)