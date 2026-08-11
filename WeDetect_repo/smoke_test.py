"""
Smoke test: OralCLIP-v8 ConvNeXt backbone -> FasterRCNN -> dental detection.

Verifies the full pipeline end-to-end on a real COCO-format dental dataset:
  1. build the ConvNeXt-FPN detector and load OralCLIP-v8 vision weights,
  2. run a few train iterations (forward + backward + optimizer step) and show
     the detection losses going down,
  3. run one eval/inference forward and report predicted boxes.

Run:
  CUDA_VISIBLE_DEVICES=3 python WeDetect_repro/smoke_test.py \
      --data-root /nfs_share/public/OralGPT/Xray_4diseases_det \
      --oralclip OralCLIP/ckpts/clip_checkpoints_v8/best.pt --iters 10
"""
import argparse
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convnext_fpn import build_detector
from coco_dataset import CocoDetDataset, collate_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/nfs_share/public/OralGPT/Xray_4diseases_det")
    ap.add_argument("--oralclip", default="OralCLIP/ckpts/clip_checkpoints_v8/best.pt")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="skip loading OralCLIP weights (random backbone)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\n[1/4] Dataset: {args.data_root}")
    ds = CocoDetDataset(args.data_root, "train2017")
    print(f"  images={len(ds)}  classes({ds.num_classes - 1})={ds.classes}")
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn)

    print("\n[2/4] Build detector (ConvNeXt-FPN + FasterRCNN)")
    ckpt = None if args.no_pretrained else args.oralclip
    model = build_detector(num_classes=ds.num_classes, oralclip_ckpt=ckpt).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"  trainable params: {n_train / 1e6:.1f}M")
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    print(f"\n[3/4] Train smoke loop: {args.iters} iters (bs={args.batch_size})")
    model.train()
    it = iter(dl)
    first_loss = last_loss = None
    for i in range(args.iters):
        try:
            images, targets = next(it)
        except StopIteration:
            it = iter(dl); images, targets = next(it)
        images = [img.to(device) for img in images]
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        opt.zero_grad(); loss.backward(); opt.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
        parts = " ".join(f"{k.split('_',1)[-1][:4]}={v.item():.3f}"
                         for k, v in loss_dict.items())
        print(f"  iter {i + 1:>2}/{args.iters}  loss={loss.item():.4f}  [{parts}]")

    print(f"\n[4/4] Eval / inference forward")
    model.eval()
    with torch.no_grad():
        img, target = ds[0]
        out = model([img.to(device)])[0]
        keep = out["scores"] > 0.05
        print(f"  image_id={target['image_id']}  raw_dets={len(out['boxes'])}  "
              f"score>0.05={int(keep.sum())}  "
              f"max_score={out['scores'].max().item():.3f}" if len(out['scores']) else
              f"  image_id={target['image_id']}  raw_dets=0")

    ok = (first_loss is not None and last_loss is not None
          and torch.isfinite(torch.tensor(last_loss)))
    print("\n" + ("=" * 60))
    print(f"loss: {first_loss:.4f} -> {last_loss:.4f}  "
          f"(Δ={last_loss - first_loss:+.4f})")
    print("SMOKE TEST PASSED ✅" if ok else "SMOKE TEST FAILED ❌")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
