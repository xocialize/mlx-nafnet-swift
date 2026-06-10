// swift-tools-version: 6.2
import PackageDescription

// mlx-nafnet-swift — the MLXEngine `imageRestore` package over NAFNet.
// The first transform capability of the visual optimization tier: denoise/deblock an Image at
// constant resolution, typically gated by imageQualityScore ("restore only where it pays").
// A thin conformance layer over the standalone nafnet-mlx-swift core. Module/product is `MLXNAFNet`.
let package = Package(
    name: "mlx-nafnet-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXNAFNet", targets: ["MLXNAFNet"]),
    ],
    dependencies: [
        .package(path: "../mlx-engine-swift"),
        .package(url: "https://github.com/xocialize/nafnet-mlx-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "MLXNAFNet",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "NAFNetMLX", package: "nafnet-mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            // NAFNet (MLX) isn't Sendable-audited; the engine serializes lifecycle on
            // InferenceActor, so v5 mode keeps region-isolation a warning — same as siblings.
            swiftSettings: [.swiftLanguageMode(.v5)]
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
