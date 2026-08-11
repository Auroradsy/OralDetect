"""
ConvNeXt-Base + FPN detection backbone, initialised from the OralCLIP-v8 vision
encoder.

WeDetect (the original) is a YOLO-World / MMYOLO open-vocabulary detector whose
*image* backbone is ConvNeXt-Base (`backbone.image_model.model.*` in the
checkpoint). Here we reproduce the *detection pipeline around that same backbone*
with torchvision (no mmcv/mmdet/mmyolo dependency), so we can verify end-to-end
that the OralCLIP-v8-trained ConvNeXt features are usable for dental detection.

OralCLIP's vision encoder is `timm.create_model("convnext_base", num_classes=0)`
(its weights live under `vision.backbone.*`, 342 tensors, all matching the base
model). We load those, then expose the 4 stage feature maps via
`forward_intermediates` and attach a torchvision FeaturePyramidNetwork.
"""
import torch
import torch.nn as nn
import timm
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


class ConvNeXtFPN(nn.Module):
    """ConvNeXt-Base body (timm) + FPN, torchvision-detection compatible.

    forward(x) -> OrderedDict{'0','1','2','3','pool'} of (N, out_channels, H, W),
    mirroring torchvision's resnet_fpn_backbone output so it drops straight into
    FasterRCNN / RetinaNet.
    """

    def __init__(self, oralclip_ckpt: str = None, out_channels: int = 256,
                 freeze_backbone: bool = False):
        super().__init__()
        self.body = timm.create_model("convnext_base", pretrained=False, num_classes=0)
        self.stage_channels = [128, 256, 512, 1024]  # convnext_base stage dims
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.stage_channels,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels
        if oralclip_ckpt:
            self.load_oralclip(oralclip_ckpt)
        if freeze_backbone:
            for p in self.body.parameters():
                p.requires_grad = False

    def load_oralclip(self, path: str):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = ck.get("model", ck.get("model_state_dict", ck))
        bb = {k[len("vision.backbone."):]: v
              for k, v in state.items() if k.startswith("vision.backbone.")}
        if not bb:
            raise RuntimeError(f"no 'vision.backbone.*' keys in {path}")
        missing, unexpected = self.body.load_state_dict(bb, strict=False)
        loaded = len(bb) - len(unexpected)
        print(f"  [ConvNeXtFPN] loaded OralCLIP backbone: {loaded}/{len(bb)} tensors "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    def forward(self, x):
        feats = self.body.forward_intermediates(
            x, indices=(0, 1, 2, 3), intermediates_only=True)
        feat_dict = {str(i): f for i, f in enumerate(feats)}
        return self.fpn(feat_dict)


def build_detector(num_classes: int, oralclip_ckpt: str = None,
                   freeze_backbone: bool = False, min_size: int = 640,
                   max_size: int = 960):
    """FasterRCNN with the ConvNeXt-FPN backbone. num_classes INCLUDES background."""
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator

    backbone = ConvNeXtFPN(oralclip_ckpt=oralclip_ckpt, freeze_backbone=freeze_backbone)
    # 5 feature maps (4 FPN levels + pool) -> 5 anchor sizes
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    model = FasterRCNN(
        backbone, num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        min_size=min_size, max_size=max_size,
    )
    return model
