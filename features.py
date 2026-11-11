import numpy as np
import cv2

HIST_BINS = (50, 60)
HIST_RANGES = [0, 180, 0, 256]


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _same_size(ref, other):
    if ref.shape[:2] == other.shape[:2]:
        return other
    return cv2.resize(other, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)


def template_score(ref, aligned):
    r_g, q_g = _gray(ref), _gray(aligned)
    q_g = _same_size(r_g, q_g)
    val = cv2.matchTemplate(q_g, r_g, cv2.TM_CCOEFF_NORMED)[0][0]
    return float(max(0.0, val))


def orb_score(ref, aligned):
    r_g, q_g = _gray(ref), _gray(aligned)
    q_g = _same_size(r_g, q_g)
    orb = cv2.ORB_create(nfeatures=1000)
    kps_r, des_r = orb.detectAndCompute(r_g, None)
    kps_q, des_q = orb.detectAndCompute(q_g, None)
    if des_r is None or des_q is None or len(des_r) < 2 or len(des_q) < 2:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    good = bf.match(des_r, des_q)
    denom = max(1, min(len(kps_r), len(kps_q)))
    return min(1.0, len(good) / denom)


def edge_template_score(ref, aligned):
    r_g, q_g = _gray(ref), _gray(aligned)
    q_g = _same_size(r_g, q_g)
    r_e = cv2.Canny(r_g, 50, 160)
    q_e = cv2.Canny(q_g, 50, 160)
    val = cv2.matchTemplate(q_e, r_e, cv2.TM_CCOEFF_NORMED)[0][0]
    return float(max(0.0, val))


def color_hist_score(ref, aligned):
    r_hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    q_hsv = cv2.cvtColor(_same_size(ref, aligned), cv2.COLOR_BGR2HSV)
    hist_r = cv2.calcHist([r_hsv], [0, 1], None, HIST_BINS, HIST_RANGES)
    hist_o = cv2.calcHist([q_hsv], [0, 1], None, HIST_BINS, HIST_RANGES)
    cv2.normalize(hist_r, hist_r, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_o, hist_o, 0, 1, cv2.NORM_MINMAX)
    val = cv2.compareHist(hist_r, hist_o, cv2.HISTCMP_CORREL)
    return float(np.clip(val, 0.0, 1.0))


def _laplacian_var(img):
    g = _gray(img)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _ratio_penalty(ref_val, q_val, pivot=4.0, eps=1e-6):
    if max(ref_val, q_val) < 1e-4:
        return 1.0
    ref_val = max(ref_val, eps)
    q_val = max(q_val, eps)
    return float(np.clip(1.0 - abs(np.log(q_val / ref_val)) / np.log(pivot), 0.0, 1.0))


def sharpness_score(ref, aligned):
    return _ratio_penalty(_laplacian_var(ref), _laplacian_var(aligned))


def noise_score(ref, aligned):
    def noise_of(img):
        g = _gray(img).astype(np.float32)
        blur = cv2.medianBlur(_gray(img), 5).astype(np.float32)
        return float(np.std(g - blur))

    return _ratio_penalty(noise_of(ref), noise_of(_same_size(ref, aligned)))


def deviation_map(ref, aligned, grid=(8, 8)):
    gh, gw = ref.shape[:2]
    r_e = cv2.Canny(_gray(ref), 50, 160)
    q_e = cv2.Canny(_gray(_same_size(ref, aligned)), 50, 160)
    r_f = ref.astype(np.float32)
    q_f = _same_size(ref, aligned).astype(np.float32)
    gy = np.linspace(0, gh, grid[0] + 1).astype(int)
    gx = np.linspace(0, gw, grid[1] + 1).astype(int)
    cells = np.zeros((grid[0], grid[1]), dtype=np.float32)
    for i in range(grid[0]):
        for j in range(grid[1]):
            ec = cv2.countNonZero(r_e[gy[i]:gy[i + 1], gx[j]:gx[j + 1]])
            eq = cv2.countNonZero(q_e[gy[i]:gy[i + 1], gx[j]:gx[j + 1]])
            area = max(1, (gy[i + 1] - gy[i]) * (gx[j + 1] - gx[j]))
            edge_diff = abs(eq / area - ec / area) * 8.0
            color_diff = float(np.mean(np.abs(r_f - q_f), axis=2))
            roi = slice(gy[i], gy[i + 1]), slice(gx[j], gx[j + 1])
            color_diff_cell = float(np.mean(color_diff[roi])) / 255.0 * 3.0
            cells[i, j] = float(np.clip(edge_diff + color_diff_cell, 0.0, 1.0))
    cells -= cells.min()
    span = cells.max()
    if span > 1e-6:
        cells = cells / span
    heat = cv2.resize(cells, (gw, gh), interpolation=cv2.INTER_CUBIC)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=min(gw, gh) / 24.0)
    heat -= heat.min()
    span = heat.max()
    if span > 1e-6:
        heat = heat / span
    return heat