//
//  NAFNet.swift
//  ForgeOptimizer / Restoration
//
//  Role: MLX-Swift port of NAFNet (Nonlinear Activation Free Network for Image
//        Restoration). Single-pass image restoration covering Gaussian noise +
//        HEVC/AV1/MPEG-2 compression artifacts. Replaces the v0.3 chain of
//        DnCNN-color + DnCNN-gray + ARCNN with one model.
//
//  Plan ref: Forge-CodingPlan-v1.0.md §B.1 / Task #10 (Phase B.1)
//  Upstream: https://github.com/megvii-research/NAFNet (MIT)
//
//  Conventions:
//    - NHWC tensor layout (MLX-Swift default; matches CLAUDE.md)
//    - `@unchecked Sendable` for classes that hold MLX state (existing convention)
//    - Weight loading uses the standard MLX.loadArrays → ModuleParameters.unflattened
//      → Module.update(verify: .noUnusedKeys) pipeline (see LiteFlowNet.swift)
//
//  Numerical correctness against the PyTorch reference is verified in Phase B.4
//  (Task #13) once a real weight conversion script and trained weights exist.
//  The tests in this phase verify architecture only (shapes, padding, param count).
//

import Foundation
import MLX
import MLXNN

// MARK: - Primitives

/// 2D channel-wise layer normalization.
///
/// Normalizes each spatial position across the channel dimension (equivalent to
/// PyTorch `LayerNorm` over `dim=1` in NCHW). For NHWC tensors the channel axis
/// is the last axis, so `MLXNN.LayerNorm(dimensions:)` already implements the
/// correct math. This is a thin wrapper purely so the module key in the
/// safetensors dump can be the same as the PyTorch model's `norm1` / `norm2`.
final class LayerNorm2d: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo var norm: LayerNorm

    init(channels: Int, eps: Float = 1e-6) {
        self._norm.wrappedValue = LayerNorm(dimensions: channels, eps: eps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm(x)
    }
}

/// SimpleGate: splits the channel dim in half and element-wise-multiplies the halves.
///
/// This is the "activation free" element of NAFNet — replaces ReLU/GELU with a
/// pure shape op (no learnable params, no nonlinearity beyond the multiply).
final class SimpleGate: Module, UnaryLayer, @unchecked Sendable {
    override init() {}

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (a, b) = x.split(axis: -1)
        return a * b
    }
}

/// Simplified Channel Attention: global average pool → 1×1 conv → multiply with input.
///
/// Equivalent to the upstream `nn.Sequential(AdaptiveAvgPool2d(1), Conv2d(c, c, 1))`
/// followed by an element-wise multiply with the gated tensor. We expose `conv`
/// as a `@ModuleInfo` so the safetensors key matches the upstream
/// `sca.1.weight` (the conv is index 1 inside the Sequential; AdaptiveAvgPool
/// has no params).
final class SCA: Module, @unchecked Sendable {
    @ModuleInfo var conv: Conv2d

    init(channels: Int) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Global average pool to [B, 1, 1, C], 1×1 conv, broadcast-multiply with input.
        // Pool in fp32: the spatial mean sums H×W values, which overflows fp16's
        // ~65504 ceiling in the intermediate accumulation at video resolutions
        // (NaN output at ≥540×960, fine at 128² — only caught by the real 4K
        // optimizer benchmark, #40). Reduce in fp32, then return to x's dtype.
        let pooled = x.asType(.float32).mean(axes: [1, 2], keepDims: true).asType(x.dtype)
        let weights = conv(pooled)
        return x * weights
    }
}

// MARK: - NAFBlock

/// The core repeating unit of NAFNet.
///
/// Two residual sub-blocks per NAFBlock. Channel expansions follow the upstream
/// defaults (`DW_Expand = 2`, `FFN_Expand = 2`).
///
/// Sub-block 1 (spatial + channel attention path):
/// ```
///   x → LN → conv1 (1×1, c → 2c)
///         → conv2 (3×3 depthwise, 2c → 2c, groups=2c)
///         → SimpleGate (2c → c)
///         → SCA (c → c)
///         → conv3 (1×1, c → c)
///         → +x · beta
/// ```
///
/// Sub-block 2 (feed-forward path):
/// ```
///   y → LN → conv4 (1×1, c → 2c)
///         → SimpleGate (2c → c)
///         → conv5 (1×1, c → c)
///         → +y · gamma
/// ```
///
/// `beta` and `gamma` are learnable per-channel residual scales, both
/// zero-initialized per the paper (the block starts as the identity).
final class NAFBlock: Module, UnaryLayer, @unchecked Sendable {

    @ModuleInfo var norm1: LayerNorm2d
    @ModuleInfo var norm2: LayerNorm2d

    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo var conv3: Conv2d
    @ModuleInfo var conv4: Conv2d
    @ModuleInfo var conv5: Conv2d

    @ModuleInfo var sca: SCA
    @ModuleInfo var sg: SimpleGate

    /// Per-channel residual scale on sub-block 1, shape [1, 1, 1, c]. Zero-init.
    @ParameterInfo var beta: MLXArray
    /// Per-channel residual scale on sub-block 2, shape [1, 1, 1, c]. Zero-init.
    @ParameterInfo var gamma: MLXArray

    init(channels c: Int, dwExpand: Int = 2, ffnExpand: Int = 2) {
        let dw = c * dwExpand
        let ffn = c * ffnExpand

        self._norm1.wrappedValue = LayerNorm2d(channels: c)
        self._norm2.wrappedValue = LayerNorm2d(channels: c)

        // 1×1 expand
        self._conv1.wrappedValue = Conv2d(
            inputChannels: c, outputChannels: dw,
            kernelSize: 1, stride: 1, padding: 0, bias: true
        )
        // 3×3 depthwise (groups = dw)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: dw, outputChannels: dw,
            kernelSize: 3, stride: 1, padding: 1,
            groups: dw, bias: true
        )
        // After SimpleGate (dw → dw/2), 1×1 project back to c
        self._conv3.wrappedValue = Conv2d(
            inputChannels: dw / 2, outputChannels: c,
            kernelSize: 1, stride: 1, padding: 0, bias: true
        )

        // FFN 1×1 expand
        self._conv4.wrappedValue = Conv2d(
            inputChannels: c, outputChannels: ffn,
            kernelSize: 1, stride: 1, padding: 0, bias: true
        )
        // After SimpleGate (ffn → ffn/2), 1×1 project back to c
        self._conv5.wrappedValue = Conv2d(
            inputChannels: ffn / 2, outputChannels: c,
            kernelSize: 1, stride: 1, padding: 0, bias: true
        )

        self._sca.wrappedValue = SCA(channels: dw / 2)
        self._sg.wrappedValue = SimpleGate()

        // NHWC broadcast shape [1, 1, 1, C]. Upstream uses NCHW [1, C, 1, 1];
        // weight conversion in B.4 reshapes accordingly.
        self._beta.wrappedValue = MLXArray.zeros([1, 1, 1, c])
        self._gamma.wrappedValue = MLXArray.zeros([1, 1, 1, c])
    }

    func callAsFunction(_ inp: MLXArray) -> MLXArray {
        // Sub-block 1: spatial + channel attention
        var x = norm1(inp)
        x = conv1(x)
        x = conv2(x)
        x = sg(x)
        x = sca(x)
        x = conv3(x)
        let y = inp + x * beta

        // Sub-block 2: pointwise FFN
        var z = norm2(y)
        z = conv4(z)
        z = sg(z)
        z = conv5(z)
        return y + z * gamma
    }
}

// MARK: - PixelShuffle (NHWC)

/// Upsample by `r` via channel-to-space reshuffling.
/// Input  : `[N, H, W, C * r * r]`
/// Output : `[N, H * r, W * r, C]`
///
/// PyTorch's `nn.PixelShuffle` operates on NCHW. The NHWC equivalent is a
/// reshape + transpose + reshape with no learnable params.
func pixelShuffleNHWC(_ x: MLXArray, upscaleFactor r: Int) -> MLXArray {
    let s = x.shape
    let N = s[0]
    let H = s[1]
    let W = s[2]
    let Cin = s[3]
    precondition(Cin % (r * r) == 0,
                 "pixelShuffleNHWC: input channels (\(Cin)) must be divisible by r*r (\(r * r))")
    let C = Cin / (r * r)

    // Channel-split uses (C, r, r) ordering to match PyTorch's nn.PixelShuffle,
    // which reads input channel index `c*r*r + i*r + j` for output channel `c`
    // at sub-pixel (i, j). Phase C.3 parity testing on EfRLFN caught the
    // previous (r, r, C) variant — it produced max_abs ≈ 2.27 in the
    // upsampler vs ≈ 1e-6 with this layout. NAFNet weights aren't trained
    // yet (Phase B.3) but the same bug would have surfaced at B.4 weight
    // conversion / B.5 integration. Fix applied here proactively; matches
    // EfRLFN.swift::pixelShuffleNHWC.
    //
    // [N, H, W, C, r_i, r_j]
    let reshaped = x.reshaped([N, H, W, C, r, r])
    // Permute so r_i sits next to H, r_j next to W, C trails:
    // [N, H, r_i, W, r_j, C]
    let transposed = reshaped.transposed(0, 1, 4, 2, 5, 3)
    // [N, H*r, W*r, C]
    return transposed.reshaped([N, H * r, W * r, C])
}

// MARK: - Encoder / Decoder stages

/// One encoder stage: N NAFBlocks at the current channel count, followed by a
/// 2×2 stride-2 conv that doubles the channels and halves the spatial dims.
final class NAFNetEncoderStage: Module, @unchecked Sendable {
    @ModuleInfo var blocks: Sequential
    @ModuleInfo var down: Conv2d

    init(channels: Int, numBlocks: Int) {
        let layers: [UnaryLayer] = (0 ..< numBlocks).map { _ in NAFBlock(channels: channels) }
        self._blocks.wrappedValue = Sequential(layers: layers)
        self._down.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels * 2,
            kernelSize: 2, stride: 2, padding: 0, bias: true
        )
    }
}

/// One decoder stage: an upsample (1×1 conv to 4× channels then PixelShuffle 2×),
/// the encoder-side skip is added in NAFNet.callAsFunction (not here), then
/// N NAFBlocks at the halved channel count.
final class NAFNetDecoderStage: Module, @unchecked Sendable {
    @ModuleInfo var upConv: Conv2d
    @ModuleInfo var blocks: Sequential

    init(channels: Int, numBlocks: Int) {
        // Upstream uses Conv2d(channels → channels * 2, 1×1, bias=False) → PixelShuffle(2),
        // which preserves spatial doubling while halving the channel count.
        // Specifically: c → 2c (1×1), then PixelShuffle(2) yields 2c / (2*2) = c/2 channels.
        self._upConv.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels * 2,
            kernelSize: 1, stride: 1, padding: 0, bias: false
        )
        let outCh = channels / 2
        let layers: [UnaryLayer] = (0 ..< numBlocks).map { _ in NAFBlock(channels: outCh) }
        self._blocks.wrappedValue = Sequential(layers: layers)
    }
}

// MARK: - NAFNet

/// NAFNet — U-Net-style image restoration network.
///
/// Forward pass:
/// 1. Pad H, W up to a multiple of `2^len(encoderBlockNums)` (16 for the default
///    [2, 2, 2, 2] config).
/// 2. Intro 3×3 conv lifts 3 channels → `width`.
/// 3. For each encoder stage: N NAFBlocks at current channels, save the activation
///    for the matching decoder skip, then 2×2 stride-2 downsample (channels × 2).
/// 4. Middle: `middleBlockNum` NAFBlocks at the bottleneck channel count.
/// 5. For each decoder stage (reversed): 1×1 conv (c → 2c) + PixelShuffle(2) →
///    add encoder skip → N NAFBlocks at the new (halved) channel count.
/// 6. Ending 3×3 conv brings `width` → 3.
/// 7. Add the (padded) input as a global residual.
/// 8. Crop back to the original H, W.
public final class NAFNet: Module, @unchecked Sendable {

    @ModuleInfo var intro: Conv2d
    @ModuleInfo var ending: Conv2d

    @ModuleInfo var encoders: [NAFNetEncoderStage]
    @ModuleInfo var middle_blks: Sequential
    @ModuleInfo var decoders: [NAFNetDecoderStage]

    /// Padding granularity = 2 ^ (number of encoder stages).
    /// Cached so callAsFunction can pad inputs to a multiple of this value.
    let padderSize: Int

    /// Default config rescoped per ADR-0003: width=24, [1,1,1,1] encoder /
    /// decoder block counts produce ~1.4M params (~2.8 MB FP16), keeping the
    /// ForgeOptimizer bundle under the §4 `bundle_size_max` gate of 12 MB
    /// when replacing the legacy DnCNN×2 + ARCNN chain. The full upstream
    /// width=32 / [2,2,2,2] config remains supported via explicit args; pass
    /// it to Phase B.3 training if larger capacity is wanted.
    public init(
        imgChannels: Int = 3,
        width: Int = 24,
        middleBlockNum: Int = 1,
        encoderBlockNums: [Int] = [1, 1, 1, 1],
        decoderBlockNums: [Int] = [1, 1, 1, 1]
    ) {
        precondition(encoderBlockNums.count == decoderBlockNums.count,
                     "encoderBlockNums and decoderBlockNums must have the same length")

        self.padderSize = 1 << encoderBlockNums.count  // 2 ** len(encoders)

        // Intro: 3 → width, 3×3, pad 1
        self._intro.wrappedValue = Conv2d(
            inputChannels: imgChannels, outputChannels: width,
            kernelSize: 3, stride: 1, padding: 1, bias: true
        )

        // Encoder stages
        var encs: [NAFNetEncoderStage] = []
        var chan = width
        for n in encoderBlockNums {
            encs.append(NAFNetEncoderStage(channels: chan, numBlocks: n))
            chan *= 2
        }
        self._encoders.wrappedValue = encs

        // Middle blocks at the bottleneck channel count
        let middleLayers: [UnaryLayer] = (0 ..< middleBlockNum).map { _ in NAFBlock(channels: chan) }
        self._middle_blks.wrappedValue = Sequential(layers: middleLayers)

        // Decoder stages: walk decoderBlockNums in upstream order; channel count
        // halves at each stage.
        var decs: [NAFNetDecoderStage] = []
        for n in decoderBlockNums {
            decs.append(NAFNetDecoderStage(channels: chan, numBlocks: n))
            chan /= 2
        }
        self._decoders.wrappedValue = decs

        // Ending: width → imgChannels, 3×3, pad 1
        self._ending.wrappedValue = Conv2d(
            inputChannels: width, outputChannels: imgChannels,
            kernelSize: 3, stride: 1, padding: 1, bias: true
        )
    }

    /// Forward pass.
    /// - Parameter x: `[N, H, W, 3]` NHWC image tensor, value range matching the
    ///   training distribution (typically `[0, 1]` floats — formal preprocessing
    ///   is set by Phase B.2 training).
    /// - Returns: `[N, H, W, 3]` restored image, same dtype as input.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        let origH = s[1]
        let origW = s[2]

        // Pad H and W up to multiples of padderSize.
        let padded = padToPadderSize(x)

        // Intro
        var h = intro(padded)

        // Encoder: collect skips
        var skips: [MLXArray] = []
        for enc in encoders {
            h = enc.blocks(h)
            skips.append(h)
            h = enc.down(h)
        }

        // Middle
        h = middle_blks(h)

        // Decoder: upsample, add skip, then blocks. Iterate decoders in order;
        // skips are consumed last-in-first-out (encoder stage N matches decoder
        // stage 0, etc).
        for (i, dec) in decoders.enumerated() {
            let skip = skips[skips.count - 1 - i]
            h = dec.upConv(h)
            h = pixelShuffleNHWC(h, upscaleFactor: 2)
            h = h + skip
            h = dec.blocks(h)
        }

        // Ending
        h = ending(h)

        // Global residual against the padded input (upstream NAFNet does this
        // before the spatial crop).
        h = h + padded

        // Crop back to original H × W.
        return h[0..., 0 ..< origH, 0 ..< origW, 0...]
    }

    /// Zero-pad H and W on the right/bottom to the next multiple of `padderSize`.
    /// No-op when both dims are already aligned.
    private func padToPadderSize(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        let H = s[1]
        let W = s[2]
        let padH = (padderSize - H % padderSize) % padderSize
        let padW = (padderSize - W % padderSize) % padderSize
        if padH == 0 && padW == 0 {
            return x
        }
        return MLX.padded(
            x,
            widths: [
                IntOrPair((0, 0)),
                IntOrPair((0, padH)),
                IntOrPair((0, padW)),
                IntOrPair((0, 0)),
            ]
        )
    }
}

// MARK: - Weight loading

/// Errors raised by NAFNet's weight-loading helpers.
public enum NAFNetError: Error, CustomStringConvertible {
    case weightsNotFound(String)
    case loadFailed(String)

    public var description: String {
        switch self {
        case .weightsNotFound(let path):
            return "NAFNet weights file not found: \(path)"
        case .loadFailed(let detail):
            return "NAFNet weight load failed: \(detail)"
        }
    }
}

public extension NAFNet {
    /// Load weights from a safetensors file produced by Phase B.4's
    /// `convert_nafnet_to_mlx.py`.
    ///
    /// The converter is expected to:
    /// - Transpose conv weights from PyTorch `[out, in, kH, kW]` to MLX
    ///   `[out, kH, kW, in]`.
    /// - Reshape `beta` and `gamma` from `[1, C, 1, 1]` (NCHW) to
    ///   `[1, 1, 1, C]` (NHWC).
    /// - Emit keys that match this Swift module hierarchy
    ///   (`encoders.0.blocks.layers.0.conv1.weight`, etc).
    ///
    /// Uses the standard MLX-Swift pattern from LiteFlowNet:
    /// `MLX.loadArrays` → `ModuleParameters.unflattened` → `update(verify:)`.
    /// `.noUnusedKeys` catches converter drift (extra keys in the file the
    /// model doesn't consume).
    public func loadWeights(from url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NAFNetError.weightsNotFound(url.path)
        }

        var arrays: [String: MLXArray]
        do {
            arrays = try MLX.loadArrays(url: url)
        } catch {
            throw NAFNetError.loadFailed(String(describing: error))
        }
        // mlx-community checkpoints (SIDD/GoPro/REDS width64) ship the UPSTREAM key layout; remap to
        // this port's module tree when detected (the bundled signage weights are already port-layout).
        arrays = Self.remapUpstreamKeysIfNeeded(arrays)

        let loaded = ModuleParameters.unflattened(arrays)
        do {
            try update(parameters: loaded, verify: .noUnusedKeys)
        } catch {
            throw NAFNetError.loadFailed(String(describing: error))
        }

        MLX.eval(parameters())
    }

    // MARK: - Upstream key remap

    /// The mlx-community NAFNet checkpoints (`NAFNet-{SIDD,GoPro,REDS}-width64`) keep the **upstream**
    /// PyTorch key layout — separate `ups`/`downs` ModuleLists, flat `encoders.i.j` / `decoders.i.j` /
    /// `middle_blks.j` block arrays, and flat block-internal `norm1` / `norm2` / `sca` names — whereas
    /// this port nests them (`encoders.i.blocks.layers.j`, `decoders.i.upConv`, `norm1.norm`,
    /// `sca.conv`). The bundled signage weights are pre-sanitized to the port layout, so only the
    /// mlx-community checkpoints need remapping. Tensors are already MLX-NHWC in both — no transpose.
    /// Detected by the presence of top-level `ups.` / `downs.` keys (absent in the port layout).
    static func remapUpstreamKeysIfNeeded(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        guard isUpstreamLayout(Array(arrays.keys)) else { return arrays }
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(arrays.count)
        for (key, value) in arrays { out[remapUpstreamKey(key)] = value }
        return out
    }

    /// True when `keys` are the upstream layout (separate `ups`/`downs` ModuleLists) rather than this
    /// port's nested layout. The port layout never has top-level `ups.`/`downs.` keys.
    static func isUpstreamLayout(_ keys: [String]) -> Bool {
        keys.contains { $0.hasPrefix("ups.") || $0.hasPrefix("downs.") }
    }

    /// Upstream key → this port's module-tree key. Pure string transform.
    static func remapUpstreamKey(_ key: String) -> String {
        let p = key.split(separator: ".").map(String.init)
        guard let head = p.first else { return key }
        switch head {
        case "intro", "ending":
            return key
        case "downs":  // downs.I.{weight,bias} → encoders.I.down.{…}
            guard p.count >= 3 else { return key }
            return (["encoders", p[1], "down"] + p[2...]).joined(separator: ".")
        case "ups":    // ups.I.weight → decoders.I.upConv.weight
            guard p.count >= 3 else { return key }
            return (["decoders", p[1], "upConv"] + p[2...]).joined(separator: ".")
        case "middle_blks":  // middle_blks.J.REST → middle_blks.layers.J.<block(REST)>
            guard p.count >= 3 else { return key }
            return (["middle_blks", "layers", p[1]] + remapBlockSuffix(Array(p[2...]))).joined(separator: ".")
        case "encoders", "decoders":  // STAGE.I.J.REST → STAGE.I.blocks.layers.J.<block(REST)>
            guard p.count >= 4 else { return key }
            return ([head, p[1], "blocks", "layers", p[2]] + remapBlockSuffix(Array(p[3...]))).joined(separator: ".")
        default:
            return key
        }
    }

    /// Within a NAFBlock the upstream flat names map to this port's wrapped submodules:
    /// `norm1.*` → `norm1.norm.*`, `norm2.*` → `norm2.norm.*`, `sca.*` → `sca.conv.*`; all else 1:1.
    private static func remapBlockSuffix(_ rest: [String]) -> [String] {
        guard let first = rest.first else { return rest }
        switch first {
        case "norm1", "norm2": return [first, "norm"] + rest[1...]
        case "sca":            return ["sca", "conv"] + rest[1...]
        default:               return rest
        }
    }
}
