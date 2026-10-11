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

Weights (since 0.5.0, AB-T-0222): the configuration is **hybrid**. Signage declares its bundled checkpoint
(`BundledWeightSourcing`); a width-64 variant declares its repo (`WeightSourcing`, role `weights`), which MLXEngine
downloads into its model store before `load()`, with byte progress and resume. `load()` reads the store's flat or
snapshot layout, and also `<root>/models/<org>/<name>/`, where 0.4.x downloaded through swift-transformers' `HubApi`,
so existing installs aren't fetched again. Outside the engine it still downloads through `HubApi`.

## Usage

```swift
import MLXServeCore
import MLXNAFNet

let engine = MLXServeEngine()
try await engine.register(NAFNetRestorePackage.registration, configuration: NAFNetConfiguration())

let resp = try await engine.run(ImageRestoreRequest(image: noisyImage)) as! ImageRestoreResponse
// resp.image — restored, same dimensions, .png
```

## Consuming it

Public + version-tagged on github.com/xocialize. Add by tagged URL:
`.package(url: "https://github.com/xocialize/mlx-nafnet-swift", from: "0.1.0")`, then import `MLXNAFNet` (the conformant `imageRestore` package). Builds standalone — its engine contract (`MLXToolKit`) and model-core dependencies are tagged-URL net deps, no local checkouts.

Requirements: macOS 26+ (Apple Silicon, Metal GPU). MIT throughout (port, architecture, checkpoints).
