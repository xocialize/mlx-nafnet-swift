import Foundation
import MLXToolKit

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

    /// Conservative resident estimate (weights + full-frame activations).
    var residentBytes: UInt64 {
        switch self {
        case .signage: return 600_000_000
        case .siddWidth64, .goproWidth64, .redsWidth64: return 2_000_000_000
        }
    }
}

/// Init-time configuration for `NAFNetRestorePackage` (C9): which checkpoint to load.
public struct NAFNetConfiguration: PackageConfiguration, ModelStorable {
    public var variant: NAFNetVariant
    /// Where downloadable weights are materialized. Set by the engine from its `ModelStore`;
    /// `nil` → the default swift-transformers cache. Excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(variant: NAFNetVariant = .signage, modelsRootDirectory: URL? = nil) {
        self.variant = variant
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case variant
    }
}
