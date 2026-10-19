# SUPIR for ZawUpscale

[SUPIR](https://github.com/Fanghua-Yu/SUPIR) (Scaling-UP Image Restoration, Yu et al., CVPR 2024) rebuilds real detail in blurry, noisy, compressed or otherwise damaged photos. It's a diffusion model built on Stable Diffusion XL, so it's far heavier than Real-ESRGAN:

- **A GPU with about 12 GB** of memory (16 GB is comfortable). ZawUpscale runs it through DirectML, so on Windows only. It doesn't run on the CPU.
- **Minutes per image**: each 1024 × 1024 tile goes through the UNet twice for each of 50 steps.
- **About 11 GB of disk** for the exported model.

There's no ready-made ONNX version of SUPIR to download, and its weights may only be used non-commercially (see SUPIR's license), so you make the model folder yourself, once, from the original checkpoints, using `export_supir.py`.

## Exporting

You need Python 3.10, about 32 GB of RAM, and preferably an NVIDIA GPU with CUDA for the export. The export only runs on that machine; ZawUpscale itself works with any DirectX 12 GPU.

1. Get SUPIR's code and its Python environment:

   ```sh
   git clone https://github.com/Fanghua-Yu/SUPIR
   cd SUPIR
   python -m venv .venv && .venv\Scripts\activate      # Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt
   pip install onnx onnxruntime-directml                # or onnxruntime-gpu on Linux
   ```

2. Download the checkpoints:
   - an SDXL base model, e.g. `sd_xl_base_1.0_0.9vae.safetensors` from [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
   - `SUPIR-v0Q.ckpt` (general, better looking results) or `SUPIR-v0F.ckpt` (stays closer to lightly degraded originals), linked from [SUPIR's README](https://github.com/Fanghua-Yu/SUPIR#-dependencies-and-installation)

   The CLIP text encoders download automatically the first time (about 11 GB, into the Hugging Face cache).

3. Export, from this folder:

   ```sh
   python export_supir.py --supir-repo path/to/SUPIR --sdxl path/to/sd_xl_base_1.0_0.9vae.safetensors --supir path/to/SUPIR-v0Q.ckpt
   ```

   By default this writes the folder `supir-v0q` into ZawUpscale's models folder (`%APPDATA%\ZawUpscale\models` on Windows), so it appears in **Models** straight away. Use `--out <folder>` to put it somewhere else, then move it into the models folder. Once it's done, the script runs each network in ONNX Runtime and compares the result with PyTorch's. Export both checkpoints if you want: they're separate models.

## The model folder

| File | Contents |
| --- | --- |
| `supir.json` | Name, description and default settings (`"format": "zawupscale-supir"`) |
| `unet.onnx` + `unet.onnx.data` | The SDXL UNet with SUPIR's control network, float16 |
| `denoise_encoder.onnx` | SUPIR's degradation-robust VAE encoder |
| `vae_encoder.onnx`, `vae_decoder.onnx` | The SDXL VAE |
| `text_encoder_l.onnx`, `text_encoder_g.onnx` (+ `.data`) | CLIP ViT-L and OpenCLIP ViT-bigG, float32, run on the CPU |
| `tokenizer/vocab.json`, `tokenizer/merges.txt` | The CLIP tokenizer |

You can edit `defaults` in `supir.json`. The keys are the same as the options in the app: `steps`, `guidance`, `guidanceStart`, `control`, `controlStart`, `restoration`, `churn`, `noise`, `colorFix`, `qualityPrompt`, `negativePrompt`. You can also change `minSize`.

## How ZawUpscale runs it

It follows SUPIR's reference code (`batchify_sample` with the tiled restore EDM sampler):

1. The image is enlarged to the target size (a high-quality resize). If its shorter side would be under 1024 px, it's processed larger and reduced at the end. Then it's padded to a multiple of 64.
2. The denoising encoder turns it into the control latent. That latent is decoded and encoded again to give the restoration target and the colors for the color fix.
3. The EDM sampler (Euler steps with churn noise, linear guidance schedule) starts from pure noise. It works in 1024 px tiles overlapping by half, blended with Gaussian weights after every step. Each tile runs the conditional and unconditional prompts as one batch of two.
4. The result is decoded in overlapping tiles and given the input's colors (wavelet color fix).

If the GPU runs out of memory, the tiles get smaller (768, then 576 and 512 px) and the image starts again. The seed of each image is logged, so you can reproduce a result by entering it.

Results depend on the random noise and use their own random number generator, so they won't match SUPIR's Python demo pixel for pixel, even with the same seed.
