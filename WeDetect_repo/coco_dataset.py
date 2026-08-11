"""
COCO-format dental detection dataset -> torchvision detection target format.

The OralGPT detection datasets live as standard COCO:
    <root>/coco_format/annotations/instances_train2017.json
    <root>/coco_format/train2017/<file_name>

torchvision's FasterRCNN expects, per image:
    image : FloatTensor (3,H,W) in [0,1]
    target: {"boxes": (N,4) xyxy float, "labels": (N,) int64}
COCO category ids are remapped to a contiguous 1..K (0 reserved for background).
"""
import os
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from pycocotools.coco import COCO


class CocoDetDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train2017"):
        self.img_dir = os.path.join(root, "coco_format", split)
        ann = os.path.join(root, "coco_format", "annotations", f"instances_{split}.json")
        self.coco = COCO(ann)
        self.ids = sorted(self.coco.imgs.keys())
        # contiguous label map: coco cat_id -> 1..K
        cats = sorted(self.coco.cats.keys())
        self.cat2label = {c: i + 1 for i, c in enumerate(cats)}
        self.classes = [self.coco.cats[c]["name"] for c in cats]
        self.num_classes = len(cats) + 1  # + background

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(self.img_dir, info["file_name"])).convert("RGB")
        W, H = img.size

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat2label[a["category_id"]])

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:  # keep negatives valid for torchvision
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        img = F.to_image(img)
        img = F.to_dtype(img, torch.float32, scale=True)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(H, W)),
            "labels": labels,
            "image_id": img_id,
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
