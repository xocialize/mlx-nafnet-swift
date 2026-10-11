import Foundation
import MLXToolKit
import NAFNetMLXCore

/// A NAFNet checkpoint this package can load.
public enum NAFNetVariant: String, Codable, Sendable, CaseIterable {
    /// Signage-trained width-24 checkpoint **bundled in the core package** (4.9 MB fp16;
    /// encode-artifact restoration, val PSNR 41.515 dB). No download. Default.
    case signage
    /// Public denoise checkpoint (width 64) — `mlx-community/NAFNet-SIDD-width64`.
    case siddWidth64
    /// Public deblur checkpoint (width 64) — `mlx-community/NAFNet-GoPro-width64`.
    case goproWidth64
    /// Public REDS checkpoint (width 64) — `mlx-community/NAFNet-REDS-width64`.
    case redsWidth64

    /// HF repo for downloadable variants; `nil` for the bundled one.
    public var repo: String? {
        switch self {
        case .signage: return nil
        case .siddWidth64: return "mlx-community/NAFNet-SIDD-width64"
        case .goproWidth64: return "mlx-community/NAFNet-GoPro-width64"
        case .redsWidth64: return "mlx-community/NAFNet-REDS-width64"
        }
    }

    /// Architecture parameters (width64 repos: width=64, middle=12, enc [2,2,4,8], dec [2,2,2,2]
    /// per their config.json; signage: the core's defaults, ADR-0003).
    var architecture: (width: Int, middle: Int, enc: [Int], dec: [Int]) {
        switch self {
        case .signage: return (24, 1, [1, 1, 1, 1], [1, 1, 1, 1])
        case .siddWidth64, .goproWidth64, .redsWidth64: return (64, 12, [2, 2, 4, 8], [2, 2, 2, 2])
        }
    }

    /// Legacy flat estimate (weights + full-frame activations as one number). Superseded by the
    /// **split** `QuantFootprint` declared on the manifest (engine 1.14): the resident weights floor and
    /// the transient activation peak are now declared separately (measured via `nafnet-smoke` — see
    /// EFFICIENCY-ADOPTION.md). Kept for reference only; the manifest is the source of truth.
    var residentBytes: UInt64 {
        switch self {
        case .signage: return 600_000_000
        case .siddWidth64, .goproWidth64, .redsWidth64: return 2_000_000_000
        }
    }

    /// The declared-footprint quant this variant maps to. The two NAFNet footprints are different
    /// quants (signage = fp16, width64 publics = fp32), so the footprint key IS the quant — exposing
    /// it via `QuantConfigured` lets the governor charge the matching `QuantFootprint` per variant.
    var quant: Quant {
        switch self {
        case .signage: return .fp16
        case .siddWidth64, .goproWidth64, .redsWidth64: return .fp32
        }
    }
}

/// Init-time configuration for `NAFNetRestorePackage` (C9): which checkpoint to load.
public struct NAFNetConfiguration: PackageConfiguration, ModelStorable {
    public var variant: NAFNetVariant
    /// The engine's model store, where it materializes `weightSources` before `load()`. Set by the
    /// engine; `nil` → `load()` downloads into the default swift-transformers cache. Excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(variant: NAFNetVariant = .signage, modelsRootDirectory: URL? = nil) {
        self.variant = variant
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case variant
    }
}

/// `QuantConfigured` (engine 1.14): expose the selected variant's quant so the `MemoryGovernor`
/// charges the matching declared `QuantFootprint` (signage→fp16, width64→fp32) instead of the
/// largest-that-fits heuristic. Derived from `variant` (the footprint key here is the quant).
extension NAFNetConfiguration: QuantConfigured {
    public var quant: Quant { variant.quant }
}

/// What the engine sees of this configuration's weights (contract 1.24): the signage default is bundled in the binary
/// and needs nothing; a width-64 variant is one network source, which the engine downloads into its model store before
/// `load()` (sized, previewed and resumable) instead of the package fetching it on first use.
extension NAFNetConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        variant.repo.map { [WeightSource(role: "weights", repo: $0, matching: ["model.safetensors"])] } ?? []
    }

    /// The store's layouts, including where 0.4.x downloaded them.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        defaultMissingWeightSources(storeRoot: storeRoot).filter {
            !StoreWeights.installedByHubAPI("model.safetensors", repo: $0.repo, root: storeRoot)
        }
    }
}

extension NAFNetConfiguration: BundledWeightSourcing {
    public var bundledWeightSources: [BundledWeightSource] {
        variant.repo == nil ? [BundledWeightSource(role: "checkpoint", url: bundledSignageWeightsURL)] : []
    }
}
