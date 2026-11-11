import sys
from pathlib import Path

import cv2

import config
from dataset import product_ref
from scoring import Assessor, reasons

test_dir = config.DATA_DIR / "test"


def main():
    products = [p.name for p in (config.GENUINE_DIR.iterdir() if config.GENUINE_DIR.exists() else []) if p.is_dir()]
    if not products:
        print("No reference products found.")
        return
    assessor = Assessor()
    print(f"{'TEST IMAGE':<50} {'SCORE':>7} {'EXPECTED':<12} {'RESULT':<10} REASONS")
    print("-" * 110)
    correct = 0
    total = 0
    for product in products:
        refs = product_ref(product)
        if not refs:
            continue
        gen_dir = test_dir / "genuine"
        fake_dir = test_dir / "counterfeit"
        pairs = [(p, "genuine") for p in sorted(gen_dir.glob("*.png"))] if gen_dir.exists() else []
        pairs += [(p, "counterfeit") for p in sorted(fake_dir.glob("*.png"))] if fake_dir.exists() else []
        for img_path, kind in pairs:
            query = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if query is None:
                continue
            result = assessor.assess(refs, query)
            r = [t for t in reasons(result) if t]
            expected = "ORIGINAL" if kind == "genuine" else "FAKE"
            ok = expected == result["binary"].upper()
            correct += int(ok)
            total += 1
            print(
                f"{img_path.name[:49]:<50} {result['score']:>6.1f}% {expected:<12} "
                f"{result['binary'].upper():<10} {'; '.join(r)[:32]}"
            )
    if total:
        print("-" * 110)
        print(f"Accuracy: {correct}/{total} ({100.0 * correct / total:.0f}%)")


if __name__ == "__main__":
    main()