"""Build a WeDetect init checkpoint with OralCLIP-v8's dental-CLIP-trained ConvNeXt
vision backbone injected, keeping WeDetect's XLM-RoBERTa text tower + neck + head.

OralCLIP stores the backbone as a timm convnext_base (`vision.backbone.stem/stages.i.
blocks.j.{conv_dw,mlp.fc1,mlp.fc2,...}`); WeDetect stores it as the Facebook/mmpretrain
ConvNeXt (`backbone.image_model.model.downsample_layers/stages.i.j.{dwconv,pwconv1,
pwconv2,...}`). Same architecture/weights, different key names -> remap by name, copy
only where shapes match, leave any unmatched WeDetect tensors as the base init."""
import re, torch

WD = "/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/pretrained_models/wedetect_base.pth"
OC = "/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralCLIP/ckpts/clip_checkpoints_v8/best.pt"
OUT = "/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/pretrained_models/wedetect_oralclip_init.pth"
WD_PREF = "backbone.image_model.model."


def timm_to_fb(k):
    """timm convnext key (no prefix) -> Facebook convnext key (no prefix)."""
    if k.startswith("stem."):                       # stem.0=conv stem.1=norm
        return "downsample_layers.0." + k[len("stem."):]
    m = re.match(r"stages\.(\d+)\.downsample\.(.*)", k)
    if m:                                            # stage i downsample -> downsample_layers.i
        return f"downsample_layers.{m.group(1)}.{m.group(2)}"
    m = re.match(r"stages\.(\d+)\.blocks\.(\d+)\.(.*)", k)
    if m:
        i, j, rest = m.groups()
        rest = (rest.replace("conv_dw", "dwconv")
                    .replace("mlp.fc1", "pwconv1")
                    .replace("mlp.fc2", "pwconv2"))
        return f"stages.{i}.{j}.{rest}"
    if k.startswith("head.norm."):                   # timm final norm -> FB final norm
        return "norm." + k[len("head.norm."):]
    return None


wd = torch.load(WD, map_location="cpu", weights_only=False)
sd = wd["state_dict"]
oc = torch.load(OC, map_location="cpu", weights_only=False)["model"]
oc_vb = {k[len("vision.backbone."):]: v for k, v in oc.items() if k.startswith("vision.backbone.")}

wd_img = {k[len(WD_PREF):] for k in sd if k.startswith(WD_PREF)}
copied, shape_mismatch, no_target, unmapped = 0, [], [], []
for k, v in oc_vb.items():
    fb = timm_to_fb(k)
    if fb is None:
        unmapped.append(k); continue
    if fb not in wd_img:
        no_target.append((k, fb)); continue
    tgt = WD_PREF + fb
    if tuple(sd[tgt].shape) != tuple(v.shape):
        shape_mismatch.append((k, fb, tuple(v.shape), tuple(sd[tgt].shape))); continue
    sd[tgt] = v.clone()
    copied += 1

covered = {WD_PREF + timm_to_fb(k) for k in oc_vb if timm_to_fb(k)}
wd_uncovered = [k for k in sd if k.startswith(WD_PREF) and k not in covered]

print(f"OralCLIP vision.backbone tensors : {len(oc_vb)}")
print(f"WeDetect image_model tensors     : {len(wd_img)}")
print(f"COPIED (OralCLIP -> WeDetect)    : {copied}")
print(f"unmapped OralCLIP keys           : {len(unmapped)}  {unmapped[:6]}")
print(f"mapped but no WeDetect target    : {len(no_target)}  {no_target[:6]}")
print(f"shape mismatch                   : {len(shape_mismatch)}  {shape_mismatch[:6]}")
print(f"WeDetect image_model NOT replaced: {len(wd_uncovered)}  {[k[len(WD_PREF):] for k in wd_uncovered][:8]}")

torch.save({"meta": wd.get("meta", {}), "state_dict": sd}, OUT)
print(f"\nsaved -> {OUT}")
