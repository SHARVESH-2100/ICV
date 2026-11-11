import sys
from pathlib import Path

import cv2

import config
from dataset import product_images, light_aug, counterfeit

OUT_GENUINE = config.DATA_DIR / "test" / "genuine"
OUT_FAKE = config.DATA_DIR / "test" / "counterfeit"


def main():
    OUT_GENUINE.mkdir(parents=True, exist_ok=True)
    OUT_FAKE.mkdir(parents=True, exist_ok=True)
    made = 0
    for product in (config.GENUINE_DIR.iterdir() if config.GENUINE_DIR.exists() else []):
        if not product.is_dir():
            continue
        images = product_images(product.name)
        for i, p in enumerate(images):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            fname = f"{product.name.replace(' ', '_')}_{Path(p).stem}"
            cv2.imwrite(str(OUT_GENUINE / f"{fname}_q.png"), light_aug(img, seed=100 + i))
            cv2.imwrite(str(OUT_FAKE / f"{fname}_f.png"), counterfeit(img, seed=200 + i))
            made += 1
    print(f"saved test images x{made} -> {OUT_GENUINE}, {OUT_FAKE}")


if __name__ == "__main__":
    main()