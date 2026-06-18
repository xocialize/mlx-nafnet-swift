//
//  NAFNetTests.swift
//  ForgeOptimizerTests
//
//  Architecture tests for the MLX-Swift NAFNet port (Phase B.1 / Task #10).
//
//  Verifies forward-pass shape correctness, padding behaviour for non-multiple
//  spatial inputs, encoder/decoder stage-count symmetry, and that the total
//  trainable-parameter count is in the expected band for the width=32,
//  [2,2,2,2] config.
//
//  Numerical correctness vs the PyTorch reference is deferred to Phase B.4
//  (Task #13), which provides the weight converter and trained weights.
//

import Testing
import MLX
import MLXNN
@testable import NAFNetMLX

/// Run a closure with the MLX default device pinned to CPU.
///
/// MLX-Swift's default device is GPU/Metal. From the SwiftPM CLI test runner
/// the Metal bundle (`mlx-swift_Cmlx.bundle/.../default.metallib`) is not
/// staged into the .xctest bundle, so the very first GPU op fails with
/// "Failed to load the default metallib". This wrapper routes all MLX ops in
/// the closure (including Conv2d's MLXRandom parameter init) to CPU.
///
/// Xcode picks up the metallib via the resource bundle and these tests would
/// also pass on GPU; we only check shapes / param counts, which are
/// device-independent.
private func withCPU<R>(_ body: () throws -> R) rethrows -> R {
    try Device.withDefaultDevice(Device(.cpu), body)
}

@Suite("NAFNet")
struct NAFNetTests {


    // MARK: - Forward pass shape

    @Test("Forward pass on 256×256 input preserves shape")
    func forwardShape256() {
        withCPU {
            let model = NAFNet()
            let x = MLXArray.zeros([1, 256, 256, 3])
            let y = model(x)
            MLX.eval(y)
            #expect(y.shape == [1, 256, 256, 3])
        }
    }

    // MARK: - Padding behaviour

    @Test("Forward pass on non-multiple-of-padder input is padded internally and unpadded on output")
    func paddingRoundTrip() {
        // padderSize is 2^4 = 16 for the default config. 257 is not a multiple
        // of 16 — the model must pad internally to 272 and crop back to 257.
        withCPU {
            let model = NAFNet()
            let x = MLXArray.zeros([1, 257, 257, 3])
            let y = model(x)
            MLX.eval(y)
            #expect(y.shape == [1, 257, 257, 3])
        }
    }

    @Test("Padder size matches 2^(encoder stages)")
    func padderSize() {
        withCPU {
            let m4 = NAFNet(encoderBlockNums: [2, 2, 2, 2], decoderBlockNums: [2, 2, 2, 2])
            #expect(m4.padderSize == 16)

            let m3 = NAFNet(encoderBlockNums: [1, 1, 1], decoderBlockNums: [1, 1, 1])
            #expect(m3.padderSize == 8)
        }
    }

    // MARK: - Skip-connection symmetry

    @Test("Encoder and decoder stage counts match")
    func encoderDecoderSymmetry() {
        withCPU {
            let model = NAFNet()
            #expect(model.encoders.count == model.decoders.count)
            #expect(model.encoders.count == 4)
        }
    }

    @Test("Custom config respects encoder/decoder symmetry requirement")
    func customSymmetricConfig() {
        // Three stages each — should construct without precondition trap.
        withCPU {
            let model = NAFNet(
                width: 16,
                middleBlockNum: 2,
                encoderBlockNums: [1, 1, 1],
                decoderBlockNums: [1, 1, 1]
            )
            #expect(model.encoders.count == 3)
            #expect(model.decoders.count == 3)

            let x = MLXArray.zeros([1, 64, 64, 3])
            let y = model(x)
            MLX.eval(y)
            #expect(y.shape == [1, 64, 64, 3])
        }
    }

    // MARK: - Parameter count

    @Test("Trainable parameter count is in the expected band for width=24 / [1,1,1,1] default config")
    func parameterCount() {
        withCPU {
            let model = NAFNet()
            let total = totalParameterCount(model)

            // Default config per ADR-0003 (Phase B.1 sizing rescope):
            // width=24, encoderBlockNums=[1,1,1,1], decoderBlockNums=[1,1,1,1],
            // middleBlockNum=1. Targets ~1.4M params (~2.8 MB FP16) to fit the
            // §4 bundle_size_max gate (≤12 MB total) when replacing the legacy
            // DnCNN×2 + ARCNN chain.
            //
            // The original §B.1 plan "~1.1 GMACs target" is a compute budget,
            // not a param-count budget. Wide sanity band (0.5M – 3M) catches
            // gross mistakes without locking the exact count.
            #expect(total >= 500_000, "Param count \(total) below 0.5M lower bound")
            #expect(total <= 3_000_000, "Param count \(total) above 3M upper bound")
        }
    }

    // MARK: - Helpers

    /// Sum of all trainable param sizes in a Module tree.
    private func totalParameterCount(_ module: Module) -> Int {
        var total = 0
        for (_, value) in module.parameters().flattened() {
            total += value.size
        }
        return total
    }
}
