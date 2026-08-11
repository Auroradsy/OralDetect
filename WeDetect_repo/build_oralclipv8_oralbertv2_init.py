"""Build oraldetect-v3 init: OralCLIP-v8 vision + OralBERT-v2 (XLM-R) text + base neck/head.

Starts from wedetect_oralclipv8_base.pth (already has OralCLIP-v8 ConvNeXt + orig XLM-R +
neck/head) and overwrites the text encoder weights (backbone.text_model.model.{embeddings,
encoder}.*) with OralBERT-v2's finetuned roberta.* weights. pooler + the 768->768 head are
kept from base (pooler unused; head is WeDetect's, trained during detection).
"""
import torch
from safetensors.torch import load_file

WD  = "pretrained_models/wedetect_oralclipv8_base.pth"
OB  = "ckpts/oralbert/xlm-roberta-base_oral_mlm/final/model.safetensors"
OUT = "pretrained_models/wedetect_oralclipv8_oralbertv2_init.pth"

ck = torch.load(WD, map_location="cpu", weights_only=False)
if   "state_dict" in ck: sd = ck["state_dict"]
elif "model"      in ck: sd = ck["model"]
else:                    sd = ck

ob = load_file(OB)
mapped, missing, mism = 0, [], []
for k, v in ob.items():
    if not k.startswith("roberta."):
        continue
    wk = "backbone.text_model.model." + k[len("roberta."):]
    if wk not in sd:
        missing.append(wk); continue
    if tuple(sd[wk].shape) != tuple(v.shape):
        mism.append((wk, tuple(sd[wk].shape), tuple(v.shape))); continue
    sd[wk] = v.clone()
    mapped += 1

print(f"overwrote {mapped} text-encoder tensors with OralBERT-v2")
print(f"missing in WD ckpt: {len(missing)}  shape-mismatch: {len(mism)}")
if missing: print("  missing examples:", missing[:5])
if mism:    print("  mismatch examples:", mism[:5])
assert mapped == 197 and not missing and not mism, "merge sanity check failed"

torch.save(ck, OUT)
import os
print(f"saved {OUT} ({os.path.getsize(OUT)/1e9:.2f} GB)")
