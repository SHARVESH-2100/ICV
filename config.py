from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
GENUINE_DIR = DATA_DIR / "genuine"
WEIGHTS_DIR = ROOT / "weights"
MODEL_PATH = WEIGHTS_DIR / "siamese.pt"
TRAIN_LOG = ROOT / "train_log.txt"

IMAGE_SIZE = 256
SIAMESE_SIZE = 128
SIAMESE_MARGIN = 1.0
EMBED_DIM = 128
ORB_FEATURES = 2000
GRID = (8, 8)
ORB_RATIO_THRESHOLD = 0.75

WEIGHTS = {
    "template": 0.20,
    "orb": 0.20,
    "color": 0.12,
    "edge_layout": 0.12,
    "sharpness": 0.10,
    "noise": 0.06,
    "qr": 0.20,
    "siamese": 0.25,
}

VERDICTS = (
    (75.0, "Authentic"),
    (55.0, "Likely Authentic"),
    (35.0, "Suspicious"),
    (0.0, "Counterfeit"),
)

BINARY_THRESHOLD = 70.0
BINARY_LABELS = ("Original", "Fake")
