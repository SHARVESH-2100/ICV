import numpy as np
import cv2

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def imread_bytes(data):
    raw = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def imread(path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def letterbox(img, size=256):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        canvas = np.zeros((size, size), dtype=np.uint8)
        canvas[:nh, :nw] = resized
    else:
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized
    return canvas


def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class Aligner:
    def __init__(self, nfeatures=2000, min_matches=10, ratio_threshold=0.75):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.min_matches = min_matches
        self.ratio_threshold = ratio_threshold

    def _find(self, ref, query):
        rg, qg = to_gray(ref), to_gray(query)
        kps_r, des_r = self.orb.detectAndCompute(rg, None)
        kps_q, des_q = self.orb.detectAndCompute(qg, None)
        info = {
            "keypoints_ref": 0 if des_r is None else len(kps_r),
            "keypoints_query": 0 if des_q is None else len(kps_q),
            "matches": 0,
            "homography": None,
        }
        if des_r is None or des_q is None or len(des_r) < 2 or len(des_q) < 2:
            return 0.0, info
        matches = self.matcher.knnMatch(des_q, des_r, k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_threshold * n.distance:
                good.append(m)
        good = sorted(good, key=lambda m: m.distance)[:150]
        info["matches"] = len(good)
        denom = max(1, min(len(kps_r), len(kps_q)))
        score = min(1.0, len(good) / denom)
        if len(good) >= self.min_matches:
            src = np.float32([kps_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kps_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is not None:
                info["homography"] = H
        return score, info

    def align(self, ref, query):
        rh, rw = ref.shape[:2]
        score, info = self._find(ref, query)
        if info["homography"] is not None:
            aligned = cv2.warpPerspective(query, info["homography"], (rw, rh))
        else:
            aligned = cv2.resize(query, (rw, rh), interpolation=cv2.INTER_AREA)
        return aligned, score, info