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

## 16 bits end to end (0.6.0)

The surface declares `ToolDescriptor.rawImageFormats = [.rawBGRA8, .rawRGBA16]` (MLXToolKit contract 1.52.0,
AB-T-0230). A `.rawRGBA16` image comes back `.rawRGBA16`; a PNG or TIFF deeper than 8 bits comes back as a 16-bit PNG.
- The input is read at 16 bits into the same float32 NHWC tensor, and the output is rounded to 16 bits.
- The forward is float32 even for the fp16 signage weights (a float32 input promotes them), so the output carries the
  model's full precision: 46–65 k levels per channel on the G-6 photos, 94–99.6% of values off the 8-bit grid.
- 8-bit inputs take the 8-bit path unchanged. Gate (2026-10-12): release builds on the GPU, this version against
  v0.5.0, 7 images × png and rawBGRA8 (and jpeg for the two JPEGs): every output byte-identical.
- G-6 direct gate (`mlxengine-image/WIP/g6-16bit-gate`, `restore`): the 16-bit output's smooth-area flat-neighbour
  share is 0.016–0.022 with clipped values left out, against 0.91–0.95 for the 8-bit output. The signage model pushes
  near-white smooth areas up by a step or so, so 2% of Sony's smooth values clip at 16 bits, where Desktop's 8-bit
  route used to put the source back.

## Consuming it

Public + version-tagged on github.com/xocialize. Add by tagged URL:
`.package(url: "https://github.com/xocialize/mlx-nafnet-swift", from: "0.1.0")`, then import `MLXNAFNet` (the conformant `imageRestore` package). Builds standalone — its engine contract (`MLXToolKit`) and model-core dependencies are tagged-URL net deps, no local checkouts.

Requirements: macOS 26+ (Apple Silicon, Metal GPU). MIT throughout (port, architecture, checkpoints).
