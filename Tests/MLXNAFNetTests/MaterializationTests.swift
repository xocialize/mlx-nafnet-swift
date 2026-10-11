import Foundation
import MLXServeConformance
import MLXToolKit
import NAFNetMLXCore
import Testing
@testable import MLXNAFNet

/// The engine materializes a width-64 variant's weights before load(); the signage default is bundled (contract 1.24,
/// AB-T-0222). Offline: no network, no weights, no MLX kernels.
struct MaterializationTests {
    let root = FileManager.default.temporaryDirectory.appending(path: "nafnet-store-\(UUID().uuidString)")

    @discardableResult
    func place(_ directory: URL) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appending(path: "model.safetensors")
        try Data([0]).write(to: file)
        return file
    }

    /// MAT-5 (optional) checks a satisfied configuration without a store, which only explicit paths can satisfy; a
    /// width-64 variant has none, so its satisfied posture is the store test below.
    @Test func matGatePassesEveryVariant() {
        for variant in NAFNetVariant.allCases {
            let config = NAFNetConfiguration(variant: variant)
            let report = MaterializationConformance.check(freshConfiguration: config,
                                                          satisfiedConfiguration: variant.repo == nil ? config : nil)
            #expect(report.passed, "\(variant): \(report.summary)")
        }
    }

    @Test func signageIsBundledAndWidth64IsANetworkSource() {
        let signage = NAFNetConfiguration()
        #expect(signage.weightSources.isEmpty)
        #expect(signage.bundledWeightSources.map(\.role) == ["checkpoint"])
        #expect(signage.bundledWeightSources.first?.url == bundledSignageWeightsURL)
        #expect(signage.missingWeightSources(storeRoot: nil).isEmpty)

        let sidd = NAFNetConfiguration(variant: .siddWidth64)
        #expect(sidd.bundledWeightSources.isEmpty)
        #expect(sidd.weightSources == [WeightSource(role: "weights", repo: "mlx-community/NAFNet-SIDD-width64",
                                                    matching: ["model.safetensors"])])
    }

    @Test func engineFlatLayoutSatisfiesAndLoads() throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let config = NAFNetConfiguration(variant: .goproWidth64, modelsRootDirectory: root)
        #expect(config.missingWeightSources(storeRoot: root).count == 1)
        let file = try place(ModelStore(root: root).directory(for: "mlx-community/NAFNet-GoPro-width64")!)
        #expect(config.missingWeightSources(storeRoot: root).isEmpty)
        #expect(StoreWeights.file("model.safetensors", repo: "mlx-community/NAFNet-GoPro-width64", root: root)?.path == file.path)
    }

    /// 0.4.x downloaded through `HubApi(downloadBase:)` into `<root>/models/<org>/<name>/`: not downloaded again.
    @Test func pre050InstallIsNotDownloadedAgain() throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let config = NAFNetConfiguration(variant: .redsWidth64, modelsRootDirectory: root)
        let file = try place(root.appending(path: "models/mlx-community/NAFNet-REDS-width64"))
        #expect(config.missingWeightSources(storeRoot: root).isEmpty)
        #expect(StoreWeights.file("model.safetensors", repo: "mlx-community/NAFNet-REDS-width64", root: root)?.path == file.path)
    }
}
