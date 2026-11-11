import numpy as np
import cv2

import config

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_products():
    d = config.GENUINE_DIR
    if not d.exists():
        return []
    products = []
    for p in sorted(d.iterdir()):
        if p.is_dir() and any(f.suffix.lower() in IMG_EXTS for f in p.iterdir()):
            products.append(p.name)
    return products


def product_images(product):
    d = config.GENUINE_DIR / product
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def product_ref(product, max_n=6):
    images = []
    for p in product_images(product)[:max_n]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    return images


def save_reference(product, name, data):
    d = config.GENUINE_DIR / product
    d.mkdir(parents=True, exist_ok=True)
    if not name.lower().endswith(IMG_EXTS):
        name = name + ".jpg"
    path = d / name
    path.write_bytes(data)
    return path


def _recolor(img, rng):
    out = cv2.convertScaleAbs(img, alpha=float(rng.uniform(0.7, 1.3)), beta=int(rng.uniform(-25, 25)))
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-15, 15)) % 180
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def _warp(img, rng):
    h, w = img.shape[:2]
    j = 0.10
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src.copy()
    for i in range(4):
        dst[i, 0] += rng.uniform(-j * w, j * w)
        dst[i, 1] += rng.uniform(-j * h, j * h)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _blur(img, rng):
    k = int(rng.choice([3, 5, 7, 9]))
    return cv2.GaussianBlur(img, (k, k), float(rng.uniform(0.5, 2.5)))


def _noise(img, rng):
    sigma = float(rng.uniform(4, 22))
    n = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def _jpeg(img, rng):
    q = int(rng.uniform(20, 65))
    ok, buff = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    return cv2.imdecode(buff, cv2.IMREAD_COLOR)


def _stain(img, rng):
    h, w = img.shape[:2]
    cx, cy = int(w * rng.uniform(0.35, 0.65)), int(h * rng.uniform(0.35, 0.65))
    rx, ry = int(w * rng.uniform(0.08, 0.18)), int(h * rng.uniform(0.08, 0.18))
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.4, 1.6), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.5, 1.5), 0, 255)
    variant = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    variant = cv2.GaussianBlur(variant, (7, 7), 2)
    return np.where(mask[..., None] > 0, variant, img).astype(np.uint8)


def _shift(img, rng):
    dx = int(rng.uniform(-0.04, 0.04) * img.shape[1])
    dy = int(rng.uniform(-0.04, 0.04) * img.shape[0])
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def _reprint(img, rng):
    h, w = img.shape[:2]
    scale = float(rng.uniform(0.35, 0.6))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    low = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    low = low.astype(np.float32) + rng.normal(0.0, 5.0, low.shape).astype(np.float32)
    sat = cv2.cvtColor(np.clip(low, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    sat[..., 1] = np.clip(sat[..., 1] * rng.uniform(0.5, 1.05), 0, 255)
    return cv2.cvtColor(sat.astype(np.uint8), cv2.COLOR_HSV2BGR)


_FAKE_OPS = (_warp, _recolor, _blur, _noise, _jpeg, _stain, _shift, _reprint)


def synth_fake(img, seed=None):
    rng = np.random.default_rng(seed)
    out = img.copy()
    ops = list(_FAKE_OPS)
    rng.shuffle(ops)
    n = int(rng.integers(1, 4))
    used = []
    for op in ops[:n]:
        try:
            out = op(out, rng)
            used.append(op.__name__)
        except Exception:
            pass
    return out, used


def counterfeit(img, seed=None):
    rng = np.random.default_rng(seed)
    out = _reprint(img, rng)
    out = _recolor(out, rng)
    out = _jpeg(out, rng)
    if rng.random() < 0.5:
        out = _warp(out, rng)
    return out


def light_aug(img, seed=None):
    rng = np.random.default_rng(seed)
    out = cv2.convertScaleAbs(img, alpha=float(rng.uniform(0.9, 1.1)), beta=int(rng.uniform(-12, 12)))
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out