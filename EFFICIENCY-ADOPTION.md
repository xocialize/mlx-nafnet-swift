# Efficiency Adoption Brief — `mlx-nafnet-swift` (NAFNet, `imageRestore`)

> **For a session-specific agent.** Self-contained: audit + tasks to adopt the MLXEngine
> library-efficiency contract (engine 1.14.0 / 0.15.0). Load the `mlx-swift-integration` skill and read
> `references/package-efficiency.md` (incl. "Gotchas & measurement") + `references/memory-harness.md`
> first. Brief shape follows the BiRefNet/LTX template. Audited + executed 2026-06-30.

## Why this one matters
NAFNet is the **restore** link of the ForgeOptimizer chain (IQA → restore → upscale → colorize,
alongside BiRefNet matting). It's single-component but runs the **full frame** (no internal tiling), so
its footprint is overwhelmingly activation. With the 1.14 split + serialized transient reserve it
co-resides with the rest of the chain on weights while sharing ONE activation reserve — the direct
co-residency win for the optimizer.

## Package at a glance
- **Wrapper:** `NAFNetRestorePackage` (`Sources/MLXNAFNet/`), core `NAFNet` (`NAFNetMLXCore`). Capability `imageRestore`. **Single-component**, one forward over the whole image.
- **Two declared footprints by variant/quant:** `signage` (width-24, **fp16**, bundled 4.9 MB weights, default, no download) · width64 publics (`siddWidth64`/`goproWidth64`/`redsWidth64`, **fp32**, mlx-community downloads).
- **Home:** `mlxengine-image/PROD/`.

## Engine dependency status
- `Package.swift` pinned `mlx-engine-swift` **`from: "0.10.0"`**, resolved 0.10.0. **P0 = `swift package update`** → 0.15.0 (the pin admits it; no manifest edit). Done.

## Audit vs. the four levers

| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡→🟢 | resolved 0.10.0; re-resolved to 0.15.0 | **P0 (done)** |
| 1. Split footprint | ❌→🟢 | was flat `QuantFootprint(.fp16, 0.6 GB)` + `(.fp32, 2.0 GB)` — and the flat fp16 0.6 GB **under-declared** the measured 2.0 GB peak. Now split per quant | **P1 (done)** |
| QuantConfigured | ❌→🟢 | config had `variant`, no `quant`; added derived `quant` (signage→fp16, width64→fp32) + conformance | **P1 (done)** |
| 2. mmap/lazy load | 🟢 | `MLX.loadArrays` → `update` → `eval(parameters())`; no full eager cast of a second dict | note only |
| 3. Per-stage evict | ➖ | single-component — no multi-stage pipeline to stage | n/a |
| 4. BudgetAware | ➖ | fp16/fp32 are the validated runtimes per variant; no in-variant memory/quality dtype lever | defer |

## P0 — engine update (done)
`swift package update mlx-engine-swift` → resolved **0.10.0 → 0.15.0**. Builds clean against the 1.14 contract.

## QuantConfigured (done)
`NAFNetConfiguration` conforms via an extension exposing `variant.quant` (signage→fp16, width64→fp32),
so the `MemoryGovernor` charges the matching declared `QuantFootprint` per variant instead of the
largest-that-fits heuristic. The footprint key here *is* the quant.

## P1 — split declared (done)
NAFNet processes the **full frame**, so the working set is overwhelmingly activation. Re-measured via the
new `nafnet-smoke` target through the real `MLXServeEngine` (register → run) at a **1024×1024** envelope:

| Variant | quant | floor (resident) | peak | activation (peak−floor) | run |
|---|---|---|---|---|---|
| signage (w24) | fp16 | 5 MB | 2034 MB | ~2.0 GB | 1.1 s |
| siddWidth64 | fp32 | 443 MB | 3271 MB | ~2.8 GB | 32 s |

Declared (with overhead/headroom):
```swift
QuantFootprint(.fp16, residentBytes:  64_000_000, peakActivationBytes: 2_000_000_000)  // signage
QuantFootprint(.fp32, residentBytes: 512_000_000, peakActivationBytes: 2_900_000_000)  // width64
```
**Engine charge:** the flat fp16 0.6 GB (which *under-declared* the real 2.0 GB activation) and fp32 2.0 GB
become resident floors of **64 MB / 512 MB** plus a **shared** transient reserve (≈2.0–2.9 GB, one across
all residents). Output validated non-uniform (luma 0…0.99). Regression tests added
(`splitFootprintDeclared`, `quantConfigured`).

## Already good (don't regress)
- Native HF downloader (width64) → engine model store (`modelsRootDirectory`) + `WeightDownloadProgress` forwarding.
- Bundled signage weights (no download), `ModelStorable`, cancellation, rawBGRA8 in/out.
- The legacy `NAFNetVariant.residentBytes` computed prop is now **unused** (superseded by the manifest split); kept as a labeled reference (the manifest is source of truth). Don't reintroduce it into the manifest.

## Deferred — P2 (n/a single-component), P3 (n/a), P4 (BudgetAware: per-variant quant is the runtime, no in-variant dtype lever).

## Definition of done
- [x] `swift package update` → engine 0.15.0.
- [x] Config conforms to `QuantConfigured` (derived from variant).
- [x] Split declared per quant, re-measured via `nafnet-smoke` (1024² envelope), provenance comments kept.
- [x] Smoke + offline tests green; valid non-uniform output captured.
- [x] BudgetAware deferred (note).
- [x] Update the NAFNet row in `mlx-engine-swift/docs/model-registry.md`: Eff ✅, Eng 0.15.0.

## Outcome (executed 2026-06-30)
P0+QuantConfigured+P1 done as above. Validated via `nafnet-smoke` through `MLXServeEngine` at 1024².
Notable finding: the **old flat fp16 0.6 GB under-declared** the activation peak (2.0 GB) — the split both
right-sizes the charge and frees it into the shared reserve. width64 download path unchanged.
