# mlx-nafnet-swift

The MLXEngine **`imageRestore`** package over [NAFNet](https://github.com/xocialize/nafnet-mlx-swift) — image restoration (denoise / deblock / encode-artifact removal) on Apple Silicon, the first **transform** capability of the visual optimization tier.

Restores an `Image` at constant resolution. The Layer-3 pipeline planner pairs it with
`imageQualityScore` for the gated-restoration pattern ("restore only where it pays").

## Variants

| Variant | Checkpoint | Source |
|---|---|---|
| `.signage` (default) | width-24 signage-trained (4.9 MB fp16, val PSNR 41.515 dB) | **bundled in the core — no download** |
| `.siddWidth64` | denoise | `mlx-community/NAFNet-SIDD-width64` |
| `.goproWidth64` | deblur | `mlx-community/NAFNet-GoPro-width64` |
| `.redsWidth64` | REDS | `mlx-community/NAFNet-REDS-width64` |

## Usage

```swift
import MLXServeCore
import MLXNAFNet

let engine = MLXServeEngine()
try await engine.register(NAFNetRestorePackage.registration, configuration: NAFNetConfiguration())

let resp = try await engine.run(ImageRestoreRequest(image: noisyImage)) as! ImageRestoreResponse
// resp.image — restored, same dimensions, .png
```

Requirements: macOS 26+ (Apple Silicon, Metal GPU). MIT throughout (port, architecture, checkpoints).
