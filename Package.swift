// swift-tools-version: 6.2
import PackageDescription

// mlx-nafnet-swift — NAFNet image restoration for MLXEngine. ONE repo, TWO products:
//   • NAFNetMLX — engine-agnostic Swift/MLX core (no MLXToolKit dep; usable standalone)
//   • MLXNAFNet — the MLXEngine `imageRestore` ModelPackage over that core
// Consolidated 2026-06-18: the former standalone `nafnet-mlx-swift` core was folded in here (and
// archived) so a model is one repo with no cross-repo version skew. Python ref: xocialize/nafnet-mlx.
let package = Package(
    name: "mlx-nafnet-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "NAFNetMLX", targets: ["NAFNetMLX"]),
        .library(name: "MLXNAFNet", targets: ["MLXNAFNet"]),
    ],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.8.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        // Engine-agnostic core (folded in from nafnet-mlx-swift) — NO MLXToolKit dep, usable standalone.
        .target(
            name: "NAFNetMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            // Per-file .copy so the bundle layout is flat (forge ADR-0011).
            resources: [
                .copy("Resources/nafnet.safetensors"),
            ]
        ),
        // MLXEngine `imageRestore` wrapper over the local core.
        .target(
            name: "MLXNAFNet",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                "NAFNetMLX",
                .product(name: "Hub", package: "swift-transformers"),
            ],
            // NAFNet (MLX) isn't Sendable-audited; the engine serializes lifecycle on
            // InferenceActor, so v5 mode keeps region-isolation a warning — same as siblings.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "NAFNetMLXTests",
            dependencies: [
                "NAFNetMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXNAFNetTests",
            dependencies: [
                "MLXNAFNet",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
