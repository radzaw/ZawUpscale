# <img src="assets/icon/png/zawupscale-64.png" width="40" height="40" alt="" align="top"> ZawUpscale

An AI image upscaler for single images and whole folders, in the spirit of Upscayl, written in C#. A sibling of ZawSQL and ZawHTTP, sharing their architecture and look.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/screenshots/zawupscale-dark.png">
  <img src="assets/screenshots/zawupscale-light.png" alt="ZawUpscale: settings on the left; on the right a 500 × 333 photo of a fox kit upscaled 4× with Real-ESRGAN x4plus, with a slider wiping between the blurry original and the sharp result">
</picture>

The backend is an ASP.NET Core app that runs the AI models with [ONNX Runtime](https://onnxruntime.ai) and reads and writes the images in place. The UI is plain HTML/CSS/JS embedded in the executable, shown in a **native window** through [Photino](https://www.tryphotino.io) with the operating system's own web view: WebView2 on Windows, WKWebView on macOS and WebKitGTK on Linux. It has native open dialogs, follows the light or dark theme in its title bar, and closing it quits the app. Where the web view isn't available (WebKitGTK not installed, or no display), ZawUpscale falls back to an app-mode window of an installed Chromium browser (Chrome, Edge, Chromium or Brave).

## Features

- **One image or a whole folder**: open images (Ctrl+O), a folder (Ctrl+Shift+O, optionally with subfolders), drop files or folders onto the window, or paste an image or a copied path with Ctrl+V.
- **Formats**: PNG, JPEG, WebP, BMP, GIF and ICO, and through [ImageMagick](https://github.com/dlemstra/Magick.NET) AVIF, HEIC/HEIF (iPhone photos; read only), TIFF and JPEG XL. Wide-gamut photos (Display P3, Adobe RGB) are converted to sRGB. **Animated GIFs and WebPs** are upscaled frame by frame and stay animated, with their timing.
- **AI models**, downloaded on first use and checked against their SHA-256:
  - **General (fast)**: Real-ESRGAN general v3 (4.6 MB). A good default for photos, screenshots and everyday images.
  - **Real-ESRGAN x4plus (high quality)**: the full network (64 MB), with the most detail for real-world photos. It's about ten times slower.
  - **Anime & illustration**: Real-ESRGAN x4plus anime 6B (17 MB), for clean lines and flat colors.
  - In **Models…**, more to download: **Real-ESRGAN x2plus** (64 MB), trained for 2× itself, so 2× and 3× keep finer detail than a 4× model reduced afterwards; and **RealPLKSR** 4× and 2× (28 MB each, MIT, from darktable), sharp and faithful without invented texture. Downloaded ones join the three above in the panel.
  - **Half precision**: with *Half precision (float16) on the GPU* in Preferences, the Real-ESRGAN models are downloaded and run in their float16 versions: half the size and GPU memory, 10–20% faster, practically the same result. Both versions can be kept; the CPU always uses float32, which is faster there.
  - **SUPIR** (optional): diffusion-based restoration on Stable Diffusion XL that rebuilds real detail in blurry, noisy or damaged photos. It needs a GPU with about 12 GB and takes minutes per image. Its weights are for non-commercial use only and there's no ONNX version to download, so you export it once from the original checkpoints with [tools/supir](tools/supir/README.md). With SUPIR selected, the panel shows an optional description of the image, steps, guidance and seed. More options (restoration guidance, control strength, color fix, prompts) are folded away.
  - **Your own models**: any ESRGAN-style super-resolution network exported to ONNX (an RGB input `[1, 3, H, W]` in 0–1, float32 or float16). It can take any size or a fixed tile size, and its scale (1×, 2×, 4×…) is detected automatically.
- **Face restoration** (optional): after upscaling, faces are found ([YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)), cut out aligned, rebuilt by a face model and blended back over the result, with a strength setting. It brings back eyes, teeth and skin in small or blurry faces that upscalers smear. Choose **GFPGAN 1.4** (325 MB, Apache-2.0) or **CodeFormer** (360 MB, non-commercial only, keeps closer to the person's likeness); the detector (230 KB) comes with either. Results go in a `…_x4_faces` folder. It's off for SUPIR, which rebuilds faces itself.
- **Or a target size**: a width, a height, a long edge, or a box to fit inside (it turns to match portrait and landscape images, so 3840 × 2160 fits a portrait photo into 2160 × 3840), with presets for screens (Full HD, 4K, 5K, 8K) and prints at 300 dpi (10 × 15 cm to A3). Images already larger than the target are reduced to it and enhanced at that size.
- **Look**: how much of the AI's detail to keep (blending in a plain enlargement looks more natural on some photos), sharpening or softening, and *Keep the original colors*, which gives the result the original's colors and tones if the model shifted them.
- **Any scale from 1× to 32×**: a 4× model runs several times for 8× or 16×. For in-between scales (2×, 3×, 6×) the result is resized with a high-quality filter, and the image is shrunk *before* the last pass instead of overshooting (6× takes one 4× pass, a reduction to 1.5×, then a 4× pass, not 16× followed by throwing most of it away).
- **GPU acceleration**, with a choice of GPU, and the CPU when there's none or you choose it:
  - **Windows**: DirectML, for any DirectX 12 graphics card (NVIDIA, AMD or Intel).
  - **macOS**: Core ML, on the GPU and Neural Engine of Apple silicon. SUPIR doesn't run on Core ML.
  - **Linux**: CUDA for NVIDIA GPUs, in the separate `linux-x64-cuda` build. It's much larger (about 260 MB) and needs the NVIDIA driver, CUDA 12 and cuDNN 9 (in Docker, NVIDIA's `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` image has them; run it with `--gpus all`). Without them it says which library is missing and uses the CPU. On GeForce RTX 50 cards the very first run takes about a minute longer while the driver compiles the GPU code; it's cached after that (in `~/.nv`, so in Docker mount that to keep it). The plain `linux-x64` and `linux-arm64` builds use the CPU.
- **Images of any size**: they're processed in overlapping tiles (16 px of context on each side, of which only the centre is kept), so there are no seams. If the GPU runs out of memory, the tile size is halved and the image started again automatically.
- **Transparency** is kept: the alpha channel is resized alongside the result. **EXIF orientation** is applied, so phone photos come out the right way up.
- **Compare**: a before/after slider, side by side, or either image alone, with zoom (mouse wheel, 1:1, fit) and pan. When zoomed in to 200% or more, pixels are shown sharply.
- **Batches**: a grid of thumbnails with live progress, status and timing per image, and an estimate of the time left. Click a finished image to compare it. One failing image doesn't stop the rest. **Pause** stops at the next tile and **Resume** goes on from there. On Windows the taskbar button shows the progress (yellow while paused, red if images failed), and when a batch finishes while you're in another window, a notification says so and the button flashes.
- **Output**: the same format as the original (HEIC becomes JPEG), or PNG, JPEG, WebP, AVIF, TIFF or JPEG XL (with a quality setting, where WebP and JPEG XL at 100 are lossless; PNG and TIFF always are), saved in a subfolder of the opened folder named after the model and size (`realesr-general-x4v3_x4`, `…_3840x2160`), or in a folder of your choice (keeping the subfolder structure either way). Results keep the original's name by default; a pattern can change it, with `{name}`, `{scale}` and `{model}` filled in (`{name}_x{scale}` gives `photo_x4.png`). Existing files can be skipped (so an interrupted batch can simply be run again), overwritten or kept with a number added. Earlier results (those subfolders, or `photo_x4.png` next to `photo.jpg`) aren't picked up as new inputs.
- **Command line**: `ZawUpscale upscale <files or folders>` for scripts (see below).
- **Update notice**: once a day ZawUpscale asks GitHub for its latest release; if it's newer, the status bar says so, and a click shows what's new with a download for your system. Nothing is installed by itself. Help → Check for updates asks right away; Preferences turns it off, and *Skip this version* hides one release.
- **Light and dark themes** (or follow the system), and keyboard shortcuts.

### Keyboard shortcuts

| Keys | Action |
| --- | --- |
| Ctrl+O / Ctrl+Shift+O | Open images / a folder |
| Ctrl+V | Paste an image, or a copied file or folder path |
| Ctrl+Enter | Upscale |
| Esc | Cancel |
| Ctrl+M | Models |
| Ctrl+Shift+E | Show the result in its folder |
| C / S / B / A | Viewer: compare slider, side by side, original, result |
| F or 0 / 1 | Viewer: fit / 100% |
| + / − | Viewer: zoom in / out |
| ← / → | Viewer: move the slider |

## Command line

```sh
ZawUpscale upscale photo.jpg                         # realesr-general-x4v3_x4/photo.jpg
ZawUpscale upscale ~/Pictures/scans -r -s 2 -f png -o ~/Pictures/scans-2x
ZawUpscale upscale art.png -m realesrgan-x4plus-anime -n "{name}-{model}"
ZawUpscale models                                    # list the models (* = installed)
ZawUpscale download realesrgan-x4plus                # download one (--half: its float16 version)
ZawUpscale check-update                              # is a newer version out? (exit code 2 if so)
ZawUpscale upscale ~/Pictures/phone --fit 3840x2160 -f avif   # HEIC photos to 4K AVIFs
ZawUpscale upscale family.jpg --faces gfpgan-v1.4         # upscale and restore the faces
ZawUpscale upscale old-scan.jpg -m supir-v0q -s 2 --prompt "a family on a beach, 1970s photo"
```

Options for `upscale`: `-m <model>`, `-s <scale>`, `-o <dir>`, `--width <px>`, `--height <px>`, `--fit <W>x<H>`, `--long-edge <px>` (instead of `-s`), `-f auto|png|jpg|webp|avif|tif|jxl`, `-q <quality>`, `-n <pattern>`, `-r` (subfolders), `--overwrite`, `--tile <px>`, `--device auto|cpu|gpu:N`, `--faces gfpgan-v1.4|codeformer` and `--face-strength <0-1>` (default 0.8), `--detail <0-1>`, `--sharpen <-1-1>`, `--keep-colors` and `--half` (the float16 version on the GPU; `download <id> --half` gets it). SUPIR models also take `--prompt <text>`, `--negative <text>`, `--steps <n>`, `--guidance <n>`, `--restoration <n>`, `--control <n>`, `--color-fix wavelet|adain|none` and `--seed <n>`. The seed used is printed. It shares the models folder with the app (or `--config <dir>`), and exits with code 1 if an image failed.

On Windows the published `ZawUpscale.exe` is a windowed app, so shells don't wait for it by default. In `cmd` and batch files use `start /wait ZawUpscale.exe upscale …`; in PowerShell pipe it (`ZawUpscale.exe upscale … | Out-Host`) to see the output and get the exit code. `dotnet run --project src/ZawUpscale -- upscale …` behaves like a normal console program.

## Where things are stored

Everything is kept in the configuration directory (`%APPDATA%\ZawUpscale` on Windows, `~/.config/ZawUpscale` on Linux and macOS, or `--config <dir>`):

| File | Contents |
| --- | --- |
| `models/` | The downloaded models, your own `.onnx` files, and SUPIR model folders |
| `state.json` | Upscale settings, preferences, last folders |
| `update.json` | The last update check (the newest release on GitHub, and when it was checked) |
| `inbox/` | Copies of images dropped on or pasted into the window (cleared after a week) |
| `webview/` | The native window's web view data and icon |
| `browser/` | The browser profile, when the fallback browser window is used |

The originals are never changed. Results of dropped and pasted images are saved in `Pictures/ZawUpscale` unless you choose an output folder: the web view doesn't tell the app where a dropped file came from. Opening images with Ctrl+O or the Open buttons uses the original files and puts the results next to them.

## Security

- The backend listens on **127.0.0.1 only**, on a random port.
- Every API call must carry a random per-process token, which is passed to the window in the URL fragment. Other web pages and local programs can't use ZawUpscale to read your files.
- Only image files are served to the window.
- Apart from model downloads you start, ZawUpscale only goes online for the daily update check, which asks GitHub's releases API for the latest version and sends nothing but `ZawUpscale/<version>` as its user agent. It can be turned off in Preferences. Release links are opened only if they point to this project on GitHub.

## Requirements

- To run a published build: nothing on Windows 10 and 11 (self-contained; the WebView2 runtime comes with Windows) or macOS. On Linux the native window needs WebKitGTK 4.1 and libnotify (`sudo apt install libwebkit2gtk-4.1-0 libnotify4` on Debian and Ubuntu); without them a Chromium-based browser is used for the window, or the default browser.
- For the GPU: Windows 10 or 11 with a DirectX 12 graphics card. Otherwise the CPU is used, which is much slower. For a 1700 × 1075 image at 4×, the General model took 3 s on an RTX 5060 Ti and 17 s on a 16-thread CPU; x4plus took about 4½ minutes on that CPU.
- To build: the [.NET 10 SDK](https://dotnet.microsoft.com/download).

## Run from source

```sh
dotnet run --project src/ZawUpscale
```

On Windows, `run.bat` builds a Release build and starts it; any arguments are passed on (`run.bat --no-browser`).

Options (`ZawUpscale --help`): `--port <n>`, `--no-browser` (print the URL instead of opening a window), `--use-browser` (a browser app window instead of the native window), `--browser <path>`, `--keep-alive` (with a browser window), `--config <dir>`, `--token <value>`.

## Build standalone executables

```powershell
./publish.ps1                          # all platforms into dist/<rid>/
./publish.ps1 -Runtimes win-x64        # just Windows
```

```sh
./publish.sh linux-x64 osx-arm64
```

The result is a single self-contained file per platform (`ZawUpscale.exe` / `ZawUpscale`), with ONNX Runtime and Skia inside. Windows builds use the DirectML build of ONNX Runtime, macOS the standard build with Core ML, and Linux the CPU build; `./publish.sh linux-x64-cuda` (or `-p:ZawCuda=true` on any `dotnet` command, on Linux) makes the CUDA build instead. There is no Intel Mac build, because ONNX Runtime no longer ships one. Pushing a `vX.Y.Z` tag makes CI build all platforms and publish a GitHub release with `SHA256SUMS`.

### App icon

`assets/icon/zawupscale.svg` is the master drawing. `assets/icon/build-icons.ps1` draws the same icon with GDI+ and writes the PNGs, `src/ZawUpscale/zawupscale.ico` and `wwwroot/icon-256.png`:

```powershell
powershell -ExecutionPolicy Bypass -File assets/icon/build-icons.ps1
```

## Tests

```sh
dotnet test ZawUpscale.slnx
```

Unit tests cover tiling (compared pixel for pixel with an exact reference, for many image and tile sizes), multi-pass planning, alpha, cancellation, EXIF orientation, the encoders, output naming and folder scans. SUPIR's pipeline (the sampling schedule, latent tiling and color fix) is checked end to end with stand-in networks written in C#, along with the CLIP tokenizer and the model folder. Integration tests run the real ONNX Runtime on tiny models that the tests build themselves (a nearest-neighbour `Resize`, float32 and float16, dynamic and fixed size), then the whole backend: jobs, skipping and renaming, per-image errors, uploads, thumbnails and the API token. No network access or downloaded models are needed.

## Project layout

```
src/ZawUpscale/
  Program.cs, AppHost.cs, AppOptions.cs   host, token check, command line options
  AppWindow.cs                            the native window (Photino) and its open dialogs
  BrowserLauncher.cs, Heartbeat.cs        fallback browser window; quit when the last one closes
  Api.cs                                  /api endpoints
  Cli.cs                                  the upscale / models / download / check-update commands
  Upscaler.cs                             tiling, multi-pass planning, alpha
  Engine.cs                               ONNX Runtime sessions, devices, out-of-memory retry
  Gpu.cs                                  DirectX adapter list (DXGI)
  ModelStore.cs                           model catalog, downloads, custom models, SUPIR folders
  Supir/                                  SUPIR: model folder and settings, CLIP tokenizer,
                                          sampler and latent tiling, ONNX Runtime sessions
  Jobs.cs                                 the job queue, output names, existing files
  Files.cs                                file browser, folder scans, drops, reveal in folder
  Updates.cs                              the update check (GitHub releases)
  ImageIO.cs                              decoding, encoding, resizing, thumbnails (SkiaSharp)
  wwwroot/                                the UI (embedded into the executable)
    js/app.js                             layout, menus, settings panel, batches, drop and paste
    js/viewer.js                          before / after viewer
    js/filebrowser.js                     open dialogs (native, or a built-in file browser)
    js/tools.js                           models, preferences, about
tests/ZawUpscale.Tests/                   xUnit tests
tools/supir/                              export_supir.py: makes a SUPIR model folder from the original weights
```

## Credits

The built-in models are [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al. (BSD-3-Clause), in ONNX conversions published on Hugging Face by [jonathanst29](https://huggingface.co/jonathanst29/tinier-upscale-models) and [SkillSafe](https://huggingface.co/skillsafe-ai/realesrgan-x4plus-anime-6b).

[SUPIR](https://github.com/Fanghua-Yu/SUPIR) is by Fanghua Yu et al. The sampler and its settings follow their reference implementation. ZawUpscale contains none of its weights.

The native window uses [Photino](https://github.com/tryphotino/photino.NET) (Apache-2.0).

The fox photo in the screenshot is [Red Fox Mom and Kit](https://commons.wikimedia.org/wiki/File:Red_Fox_Mom_and_Kit_(53611640425).jpg) by Erin Strand, U.S. Fish and Wildlife Service (public domain), reduced to 500 × 333 before upscaling.
