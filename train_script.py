import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config
from dataset import list_products, product_images, synth_fake, light_aug
from models.siamese import SiameseNet, distance, contrastive_loss, save_model, load_model, to_tensor
from preprocess import letterbox


def build_pool():
    pool = []
    for product in list_products():
        pool.extend(product_images(product))
    if not pool:
        raise RuntimeError("No genuine reference images. Add some in data/genuine/<product>/ first.")
    return pool


class SiameseDataset(Dataset):
    def __init__(self, pool, size, image_size):
        self.pool = pool
        self.size = size
        self.image_size = image_size
        self.rng = random.Random()
        self.np_rng = np.random.default_rng()
        self.small = []
        for p in pool:
            a = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if a is None:
                a = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            self.small.append(letterbox(a, image_size))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        a = self.rng.choice(self.small)
        if self.rng.random() < 0.5:
            b = light_aug(a, seed=int(self.np_rng.integers(0, 2**31)))
            y = 1.0
        else:
            b, _ = synth_fake(a, seed=int(self.np_rng.integers(0, 2**31)))
            y = 0.0
        return (
            to_tensor(a, self.image_size)[0],
            to_tensor(b, self.image_size)[0],
            torch.tensor(y, dtype=torch.float32),
        )


def train_model(epochs=15, batch=32, image_size=256, lr=1e-3, out_path=config.MODEL_PATH, log_path=config.TRAIN_LOG, log_cb=None):
    pool = build_pool()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiameseNet(config.EMBED_DIM).to(device)
    start_ep = 1
    if out_path.exists():
        try:
            model.load_state_dict(torch.load(out_path, map_location="cpu", weights_only=True)["state"])
            print(f"Resuming from {out_path}")
        except Exception:
            print("Ignoring existing checkpoint (unable to load).")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = SiameseDataset(pool, size=256, image_size=image_size)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    history = []
    start_time = time.time()
    for ep in range(start_ep, epochs + 1):
        model.train()
        total = 0.0
        nb = 0
        for a, b, y in dl:
            a = a.to(device)
            b = b.to(device)
            y = y.to(device)
            ea, _ = model.encode(a)
            eb, _ = model.encode(b)
            d = distance(ea, eb)
            loss = contrastive_loss(d, y, margin=config.SIAMESE_MARGIN)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        avg = total / max(1, nb)
        history.append(float(avg))
        if nb > 0 and avg == avg:
            save_model(model, out_path)
        if log_cb:
            log_cb(ep, avg)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"epoch {ep:>2}/{epochs:<2}  loss={avg:.5f}\n")
    model.eval()
    save_model(model, out_path)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"run complete: epochs={epochs} batch={batch} image_size={image_size} "
                f"lr={lr} device={device} pool={len(pool)} took={time.time() - start_time:.1f}s "
                f"-> {out_path}\n"
            )
    return history, out_path


if __name__ == "__main__":
    import argparse
    import datetime
    import datetime

    parser = argparse.ArgumentParser(description="Train the Siamese authenticity encoder")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default=config.MODEL_PATH)
    parser.add_argument("--log", default=config.TRAIN_LOG)
    args = parser.parse_args()

    with open(args.log, "a", encoding="utf-8") as f:
        f.write(f"\n== Training run at {datetime.datetime.now().isoformat()} ==\n")

    history, out = train_model(
        epochs=args.epochs,
        batch=args.batch,
        image_size=args.image_size,
        lr=args.lr,
        out_path=args.out,
        log_path=args.log,
        log_cb=lambda ep, loss: print(f"epoch {ep}/{args.epochs} loss={loss:.5f}"),
    )
    print("loss history:", [round(x, 3) for x in history])
    print("saved to", out)