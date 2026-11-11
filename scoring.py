import numpy as np
import torch

import config
from preprocess import Aligner
from features import (
    template_score,
    orb_score,
    edge_template_score,
    color_hist_score,
    sharpness_score,
    noise_score,
    deviation_map,
)
from security import read_qr, qr_score
from models.siamese import load_model, to_tensor, distance
from explain import gradcam, overlay_heatmap

FAST_METRICS = ("template", "orb", "color", "edge_layout", "sharpness", "noise")


class Assessor:
    def __init__(self, model_path=config.MODEL_PATH, device=None):
        self.aligner = Aligner(
            nfeatures=config.ORB_FEATURES,
            ratio_threshold=config.ORB_RATIO_THRESHOLD,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        if model_path.exists():
            try:
                self.model = load_model(model_path, self.device, config.EMBED_DIM)
            except Exception:
                self.model = None

    def _metrics(self, ref, query, aligned):
        m = {
            "template": template_score(ref, aligned),
            "orb": orb_score(ref, aligned),
            "color": color_hist_score(ref, aligned),
            "edge_layout": edge_template_score(ref, aligned),
            "sharpness": sharpness_score(ref, aligned),
            "noise": noise_score(ref, aligned),
        }
        ref_qr, _ = read_qr(ref)
        query_qr, _ = read_qr(query)
        m["qr"], m["qr_status"] = qr_score(ref_qr, query_qr)
        m["qr_ref"] = ref_qr
        m["qr_query"] = query_qr
        m["siamese"] = None
        return m

    def _fuse(self, m):
        weights = config.WEIGHTS
        scores = {}
        for k in FAST_METRICS:
            scores[k] = m[k]
        if m["qr"] is not None:
            scores["qr"] = m["qr"]
        if m["siamese"] is not None:
            scores["siamese"] = m["siamese"]
        wsum = sum(weights[k] for k in scores)
        if wsum <= 0:
            return 0.0, {}
        raw = sum(weights[k] * scores[k] for k in scores) / wsum
        return float(np.clip(raw, 0.0, 1.0)), scores

    def assess(self, refs, query):
        if not refs:
            raise ValueError("No reference images for this product")
        candidates = []
        for ref in refs:
            aligned, align_score, info = self.aligner.align(ref, query)
            m = self._metrics(ref, query, aligned)
            m["alignment_score"] = float(align_score)
            m["alignment_matches"] = info["matches"]
            fused, _ = self._fuse(m)
            candidates.append((fused, ref, aligned, m, info))
        fused, ref, aligned, m, info = max(candidates, key=lambda c: c[0])

        if self.model is not None:
            ref_t = to_tensor(ref, config.SIAMESE_SIZE, self.device)
            query_t = to_tensor(query, config.SIAMESE_SIZE, self.device)
            with torch.no_grad():
                e_ref, _ = self.model.encode(ref_t)
                e_q, _ = self.model.encode(query_t)
                d = distance(e_ref, e_q).item()
            m["siamese"] = float(np.clip((config.SIAMESE_MARGIN - d) / config.SIAMESE_MARGIN, 0.0, 1.0))
            m["siamese_d"] = d
            aligned_t = to_tensor(aligned, config.SIAMESE_SIZE, self.device)
            heat = gradcam(self.model, ref_t, aligned_t)
        else:
            m["siamese"] = None
            heat = deviation_map(ref, aligned, grid=config.GRID)

        final, scores = self._fuse(m)
        score = final * 100.0
        verdict, color = self._verdict(score)
        binary = "Original" if score >= config.BINARY_THRESHOLD else "Fake"
        overlay = overlay_heatmap(aligned, heat)
        return {
            "score": score,
            "verdict": verdict,
            "binary": binary,
            "color": color,
            "scores": scores,
            "metrics": m,
            "heat": heat,
            "overlay": overlay,
            "ref": ref,
            "aligned": aligned,
            "query": query,
        }

    @staticmethod
    def _verdict(score):
        for threshold, name in config.VERDICTS:
            if score >= threshold:
                return name, {
                    "Authentic": "green",
                    "Likely Authentic": "blue",
                    "Suspicious": "orange",
                    "Counterfeit": "red",
                }[name]
        return "Counterfeit", "red"


def reasons(result):
    m = result["metrics"]
    texts = []
    rules = (
        ("orb", 0.40, "Logo/print feature alignment is weak"),
        ("template", 0.45, "Packaging layout deviates from the reference"),
        ("color", 0.40, "Colour distribution differs from the reference"),
        ("edge_layout", 0.40, "Print/edge structure is inconsistent"),
        ("sharpness", 0.35, "Print sharpness deviates (possible poor-quality reprint)"),
        ("noise", 0.35, "Print surface noise is unusual"),
    )
    for key, thr, msg in rules:
        if m[key] < thr:
            texts.append(msg)
    if result["scores"].get("qr") is not None:
        status_msg = {
            "qr_match": "Embedded QR code matches the reference",
            "missing_qr": "QR code present on reference is missing from the query",
            "unexpected_qr": "Query shows a QR code not present on the reference",
            "qr_mismatch": "QR code content differs from the reference",
        }
        texts.append(status_msg.get(m["qr_status"], ""))
    if m.get("siamese") is not None and m["siamese"] < 0.5:
        texts.append("Learned visual signature differs strongly (Siamese network)")
    return [t for t in texts if t]