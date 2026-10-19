#!/usr/bin/env python3
"""
Exports SUPIR (Yu et al., "Scaling Up to Excellence", 2024) with its SDXL base to the ONNX model folder
ZawUpscale runs: the UNet with SUPIR's control network (float16), the VAE encoders and decoder, the two
CLIP text encoders (float32) and the CLIP tokenizer, plus a supir.json manifest.

Run it inside SUPIR's own Python environment (see README.md next to this script):

    python export_supir.py --supir-repo ../SUPIR --sdxl sd_xl_base_1.0_0.9vae.safetensors --supir SUPIR-v0Q.ckpt

Every network takes and returns float32; the float16 casts are inside the graph. The tensor names and
dynamic dimensions below are the contract with src/ZawUpscale/Supir/SupirModel.cs.
"""

import argparse
import inspect
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

FORMAT = "zawupscale-supir"

QUALITY_PROMPT = ("Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - "
                  "realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, "
                  "hyper sharpness, perfect without deformations.")
NEGATIVE_PROMPT = ("painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, "
                   "unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, "
                   "jpeg artifacts, deformed, lowres, over-smooth")

# SUPIR's Gradio presets: v0Q goes with "Quality", v0F with "Fidelity".
PRESETS = {
    "Q": {"name": "SUPIR v0Q (quality)",
          "description": "SUPIR v0Q on SDXL: rebuilds lifelike detail in blurry, noisy or damaged photos. "
                         "Slow (minutes per image); needs a GPU with about 12 GB.",
          "defaults": {"guidance": 7.5, "guidanceStart": 4.0}},
    "F": {"name": "SUPIR v0F (fidelity)",
          "description": "SUPIR v0F on SDXL: restoration that stays closer to the original, for light degradation. "
                         "Slow (minutes per image); needs a GPU with about 12 GB.",
          "defaults": {"guidance": 4.0, "guidanceStart": 1.0}},
}


def default_models_dir():
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "ZawUpscale" / "models"
    return Path.home() / ".config" / "ZawUpscale" / "models"


def parse_args():
    ap = argparse.ArgumentParser(description="Export SUPIR to a ZawUpscale model folder.")
    ap.add_argument("--supir-repo", required=True, type=Path, help="a clone of https://github.com/Fanghua-Yu/SUPIR")
    ap.add_argument("--sdxl", required=True, type=Path, help="SDXL base checkpoint, e.g. sd_xl_base_1.0_0.9vae.safetensors")
    ap.add_argument("--supir", required=True, type=Path, help="SUPIR checkpoint: SUPIR-v0Q.ckpt or SUPIR-v0F.ckpt")
    ap.add_argument("--variant", choices=["Q", "F"], help="Q or F (default: from the checkpoint's name)")
    ap.add_argument("--out", type=Path, help="output folder (default: ZawUpscale's models folder / supir-v0q or supir-v0f)")
    ap.add_argument("--device", help="cuda or cpu for tracing (default: cuda if available)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-verify", action="store_true", help="don't compare ONNX Runtime's outputs with PyTorch's")
    return ap.parse_args()


def without_xformers(node):
    """SUPIR's config asks for xformers attention, which can't be exported; plain softmax attention computes the same."""
    from omegaconf import DictConfig, ListConfig
    if isinstance(node, DictConfig):
        for k in list(node.keys()):
            v = node[k]
            if isinstance(v, str) and "xformers" in v:
                node[k] = v.replace("softmax-xformers", "softmax").replace("vanilla-xformers", "vanilla")
            else:
                without_xformers(v)
    elif isinstance(node, ListConfig):
        for v in node:
            without_xformers(v)


def load_supir(args, variant):
    from omegaconf import OmegaConf
    repo = args.supir_repo.resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)  # SUPIR's config and CKPT_PTH.py use paths relative to the repository
    from SUPIR.util import create_SUPIR_model

    config = OmegaConf.load(repo / "options" / "SUPIR_v0.yaml")
    config.SDXL_CKPT = str(args.sdxl.resolve())
    config.SUPIR_CKPT = None
    config.SUPIR_CKPT_Q = str(args.supir.resolve()) if variant == "Q" else None
    config.SUPIR_CKPT_F = str(args.supir.resolve()) if variant == "F" else None
    without_xformers(config)
    tmp = Path(tempfile.mkdtemp()) / "SUPIR_export.yaml"
    OmegaConf.save(config, tmp)
    print("Loading SUPIR (this takes a while and about 20 GB of RAM)…")
    model = create_SUPIR_model(str(tmp), SUPIR_sign=variant)
    model.eval()
    return model


def export(module, inputs, path, input_names, output_names, dynamic_axes, opset):
    import torch
    kwargs = dict(input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
                  opset_version=opset, do_constant_folding=True)
    # Newer PyTorch defaults to the dynamo exporter, which doesn't handle SUPIR's code; use the TorchScript one.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(module, inputs, str(path), **kwargs)


def save_single(src, dst):
    """Re-saves an exported model with all its weights in one external file next to it (PyTorch writes one per tensor)."""
    import onnx
    model = onnx.load(str(src))
    data = dst.name + ".data"
    for f in (dst, dst.with_name(data)):
        if f.exists():
            f.unlink()
    big = sum(t.ByteSize() for t in model.graph.initializer) > 1_800_000_000
    if big:
        onnx.save_model(model, str(dst), save_as_external_data=True, all_tensors_to_one_file=True, location=data, size_threshold=1024)
    else:
        onnx.save_model(model, str(dst))
    print(f"  wrote {dst.name}" + (f" + {data}" if big else ""))


def main():
    args = parse_args()
    import torch
    from torch import nn

    variant = args.variant or ("F" if "v0F" in args.supir.name or "-F" in args.supir.name else "Q")
    out = (args.out or default_models_dir() / f"supir-v0{variant.lower()}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_supir(args, variant)
    work = Path(tempfile.mkdtemp(prefix="supir-onnx-"))
    print(f"Exporting to {out} (temporary files in {work})")

    # ---------------------------------------------------------------- UNet + control network (float16)

    class CastInput(nn.Module):
        """Feeds a layer in its own dtype: SDXL computes the timestep embedding in float32 and relies on autocast."""
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x.to(next(self.inner.parameters()).dtype))

    class Unet(nn.Module):
        def __init__(self, wrapper):
            super().__init__()
            self.unet = wrapper.diffusion_model
            self.control = wrapper.control_model
            for m in (self.unet, self.control):
                m.time_embed = CastInput(m.time_embed)

        def forward(self, sample, timestep, context, y, control, control_scale):
            dt = torch.float16
            sample, context, y, control = sample.to(dt), context.to(dt), y.to(dt), control.to(dt)
            hint = self.control(x=control, timesteps=timestep, xt=sample, context=context, y=y)
            eps = self.unet(sample, timesteps=timestep, context=context, y=y, control=hint,
                            control_scale=control_scale.to(dt).reshape(()))
            return eps.float()

    wrapper = model.model
    if getattr(wrapper, "control_model", None) is None:
        sys.exit("The loaded model has no control network; is --supir a SUPIR checkpoint?")
    unet = Unet(wrapper).half().to(device).eval()
    b, lh, lw = 2, 64, 64
    unet_inputs = (torch.randn(b, 4, lh, lw, device=device), torch.full((b,), 500.0, device=device),
                   torch.randn(b, 77, 2048, device=device), torch.randn(b, 2816, device=device),
                   torch.randn(b, 4, lh, lw, device=device), torch.ones(1, device=device))
    print("Exporting the UNet with the control network (float16)…")
    lat = {0: "batch", 2: "lat_h", 3: "lat_w"}
    export(unet, unet_inputs, work / "unet.onnx",
           ["sample", "timestep", "context", "y", "control", "control_scale"], ["eps"],
           {"sample": lat, "timestep": {0: "batch"}, "context": {0: "batch"}, "y": {0: "batch"}, "control": lat, "eps": lat},
           args.opset)
    with torch.no_grad():
        unet_ref = unet(*unet_inputs).cpu().numpy()
    unet_inputs = [t.cpu().numpy() for t in unet_inputs]
    save_single(work / "unet.onnx", out / "unet.onnx")
    del unet
    model.model.cpu()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- VAE (float32)

    fs = model.first_stage_model.float().to(device).eval()
    scale = float(model.scale_factor)
    if not hasattr(fs, "denoise_encoder"):
        sys.exit("The VAE has no denoise_encoder; is --supir a SUPIR checkpoint and --supir-repo SUPIR's code?")

    class Encoder(nn.Module):
        """Image in [-1, 1] → the posterior's mean, times SDXL's latent scale."""
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.quant_conv = fs.quant_conv

        def forward(self, image):
            moments = self.quant_conv(self.encoder(image))
            return moments[:, :4] * scale

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.post_quant_conv = fs.post_quant_conv
            self.decoder = fs.decoder

        def forward(self, latent):
            return self.decoder(self.post_quant_conv(latent / scale)).float()

    img = {0: "batch", 2: "img_h", 3: "img_w"}
    image = torch.rand(1, 3, 256, 256, device=device) * 2 - 1
    latent = torch.randn(1, 4, 32, 32, device=device)
    vae = {}
    for name, module, x, ins, outs, axes in [
        ("denoise_encoder", Encoder(fs.denoise_encoder), image, ["image"], ["latent"], {"image": img, "latent": lat}),
        ("vae_encoder", Encoder(fs.encoder), image, ["image"], ["latent"], {"image": img, "latent": lat}),
        ("vae_decoder", Decoder(), latent, ["latent"], ["image"], {"latent": lat, "image": img}),
    ]:
        print(f"Exporting {name} (float32)…")
        module = module.eval()
        export(module, (x,), work / f"{name}.onnx", ins, outs, axes, args.opset)
        with torch.no_grad():
            vae[name] = (x.cpu().numpy(), module(x).cpu().numpy())
        save_single(work / f"{name}.onnx", out / f"{name}.onnx")
    fs.cpu()

    # ---------------------------------------------------------------- text encoders (float32, run on the CPU)

    embedders = {type(e).__name__: e for e in model.conditioner.embedders}
    clip_l = embedders.get("FrozenCLIPEmbedder")
    clip_g = embedders.get("FrozenOpenCLIPEmbedder2")
    if clip_l is None or clip_g is None:
        sys.exit(f"Unexpected text encoders: {list(embedders)}")

    class ClipL(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.transformer = emb.transformer
            self.layer_idx = emb.layer_idx if emb.layer == "hidden" else None

        def forward(self, input_ids):
            o = self.transformer(input_ids=input_ids, output_hidden_states=True)
            return o.hidden_states[self.layer_idx] if self.layer_idx is not None else o.last_hidden_state

    class ClipG(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.emb = emb

        def forward(self, input_ids):
            z = self.emb.encode_with_transformer(input_ids)
            return z[self.emb.layer], z["pooled"]

    tokens = clip_l.tokenizer("a photo of a cat", truncation=True, max_length=77, padding="max_length", return_tensors="pt")["input_ids"]
    tokens_g = tokens.clone()
    tokens_g[tokens_g == 49407] = 0
    tokens_g[0, int((tokens[0] == 49407).nonzero()[0])] = 49407  # OpenCLIP pads with 0 after the end token
    text = {}
    for name, module, ids, outs in [("text_encoder_l", ClipL(clip_l.float().cpu()), tokens, ["hidden"]),
                                    ("text_encoder_g", ClipG(clip_g.float().cpu()), tokens_g, ["hidden", "pooled"])]:
        print(f"Exporting {name} (float32)…")
        module = module.eval()
        export(module, (ids,), work / f"{name}.onnx", ["input_ids"], outs, None, args.opset)
        with torch.no_grad():
            r = module(ids)
        text[name] = (ids.numpy(), [t.numpy() for t in (r if isinstance(r, tuple) else (r,))])
        save_single(work / f"{name}.onnx", out / f"{name}.onnx")

    tok = work / "tokenizer"
    clip_l.tokenizer.save_pretrained(str(tok))
    (out / "tokenizer").mkdir(exist_ok=True)
    for f in ("vocab.json", "merges.txt"):
        shutil.copy(tok / f, out / "tokenizer" / f)

    preset = PRESETS[variant]
    manifest = {
        "format": FORMAT, "version": 1, "name": preset["name"], "description": preset["description"], "minSize": 1024,
        "defaults": {"qualityPrompt": QUALITY_PROMPT, "negativePrompt": NEGATIVE_PROMPT, "steps": 50, "control": 1.0,
                     "controlStart": 1.0, "restoration": 0, "churn": 5, "noise": 1.003, "colorFix": "wavelet",
                     **preset["defaults"]},
        "source": {"supir": args.supir.name, "sdxl": args.sdxl.name},
    }
    (out / "supir.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)

    if not args.skip_verify:
        verify(out, unet_inputs, unet_ref, vae, text)
    print(f"\nDone: {out}\nRestart ZawUpscale (or reopen Models) and choose {preset['name']}.")


def verify(out, unet_inputs, unet_ref, vae, text):
    """Runs each exported network in ONNX Runtime on the inputs PyTorch saw and compares the outputs."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("\nonnxruntime isn't installed; skipping the check (pip install onnxruntime-directml or onnxruntime-gpu).")
        return
    available = ort.get_available_providers()
    gpu = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in available]
    print("\nChecking the exported networks with ONNX Runtime…")

    def check(name, providers, inputs, refs, tol):
        sess = ort.InferenceSession(str(out / f"{name}.onnx"), providers=providers)
        feeds = {i.name: v for i, v in zip(sess.get_inputs(), inputs)}
        for o, ref in zip(sess.run(None, feeds), refs):
            err = float(np.abs(o - ref).max() / (np.abs(ref).max() + 1e-6))
            print(f"  {name}: relative error {err:.2e} {'ok' if err < tol else 'TOO LARGE'}")

    for name, (x, ref) in vae.items():
        check(name, gpu + ["CPUExecutionProvider"], [x], [ref], 1e-3)
    for name, (ids, refs) in text.items():
        check(name, ["CPUExecutionProvider"], [ids], refs, 1e-3)
    if gpu:
        check("unet", gpu, unet_inputs, [unet_ref], 5e-2)
    else:
        print("  unet: skipped (float16 needs a GPU provider: onnxruntime-directml or onnxruntime-gpu)")


if __name__ == "__main__":
    main()
