"""Inject the OralCLIP-v8 ConvNeXt vision backbone into the WeDetect-base
checkpoint, so WeDetect fine-tunes from OUR dental-CLIP-adapted backbone.

OralCLIP uses new-timm ConvNeXt naming (stem / stages.n.blocks.m.conv_dw /
mlp.fc1 ...); WeDetect uses old-timm naming (downsample_layers / stages.n.m.dwconv
/ pwconv1 ...). Same ConvNeXt-Base architecture & lineage (OralCLIP was itself
WeDetect-pretrained), so we remap keys and replace only backbone.image_model.*;
neck / bbox_head / text_model are kept from wedetect_base. Unmapped/mismatched
keys keep their base weights (logged)."""
import re, torch

WD = "/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/pretrained_models/wedetect_base.pth"
CL = "/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralCLIP/ckpts/clip_checkpoints_v8/best.pt"
OUT = "/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/pretrained_models/wedetect_oralclipv8_base.pth"
PFX = "backbone.image_model.model."

def wd_to_clip(k):
    """old-timm convnext key (no PFX) -> new-timm (OralCLIP vision.backbone) key."""
    if k.startswith("downsample_layers.0."):
        return "stem." + k[len("downsample_layers.0."):]
    m = re.match(r"downsample_layers\.([123])\.([01])\.(.+)", k)
    if m:
        n, idx, rest = m.groups()
        return f"stages.{n}.downsample.{idx}.{rest}"
    m = re.match(r"stages\.(\d)\.(\d+)\.(dwconv|norm|pwconv1|pwconv2|gamma)(.*)", k)
    if m:
        n, mm, part, rest = m.groups()
        pm = {"dwconv": "conv_dw", "norm": "norm", "pwconv1": "mlp.fc1",
              "pwconv2": "mlp.fc2", "gamma": "gamma"}
        return f"stages.{n}.blocks.{mm}.{pm[part]}{rest}"
    if k.startswith("norm."):
        return "head.norm." + k[len("norm."):]
    return None

ckpt = torch.load(WD, map_location="cpu")
sd = ckpt["state_dict"]
cl = torch.load(CL, map_location="cpu")
csd = cl.get("model", cl.get("state_dict", cl))
clk = {k[len("vision.backbone."):]: v for k, v in csd.items() if k.startswith("vision.backbone.")}

replaced = kept = mismatch = nomap = 0
miss_examples = []
for k in list(sd.keys()):
    if not k.startswith(PFX):
        continue
    bare = k[len(PFX):]
    ck = wd_to_clip(bare)
    if ck is None or ck not in clk:
        nomap += 1; kept += 1
        if len(miss_examples) < 8: miss_examples.append((bare, ck))
        continue
    if tuple(clk[ck].shape) != tuple(sd[k].shape):
        mismatch += 1; kept += 1
        if len(miss_examples) < 8: miss_examples.append((bare, ck, "SHAPE", tuple(sd[k].shape), tuple(clk[ck].shape)))
        continue
    sd[k] = clk[ck].clone()
    replaced += 1

n_img = sum(1 for k in sd if k.startswith(PFX))
print(f"backbone.image_model keys: {n_img}")
print(f"  replaced from OralCLIP-v8: {replaced}")
print(f"  kept (no map / mismatch):  {kept}  (nomap={nomap}, shape-mismatch={mismatch})")
print(f"  examples kept: {miss_examples}")
# tag meta so it's traceable
if isinstance(ckpt.get("meta"), dict):
    ckpt["meta"]["backbone_injected_from"] = "OralCLIP-v8 vision.backbone"
torch.save(ckpt, OUT)
print(f"saved -> {OUT}")
