import cv2


def read_qr(img):
    if img is None or img.size == 0:
        return None, None
    det = cv2.QRCodeDetector()
    data, pts, _ = det.detectAndDecode(img)
    data = (data or "").strip()
    return (data or None), pts


def qr_score(ref_data, query_data):
    if ref_data is None and query_data is None:
        return None, "skipped"
    if ref_data is None and query_data is not None:
        return 0.0, "unexpected_qr"
    if ref_data is not None and query_data is None:
        return 0.0, "missing_qr"
    if ref_data == query_data:
        return 1.0, "qr_match"
    return 0.0, "qr_mismatch"