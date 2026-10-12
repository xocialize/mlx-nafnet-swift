//
//  PadHelperTests.swift
//
//  The ending conv (width 24 → 3, 3×3, stride 1) is the shape mlx 0.32.2 serves with
//  `pad_in_channels_conv_2D_gpu`, which frees its zero fill value while the GPU still reads it
//  (AB-T-0214). It is a `RoutedConv2d`, so ExactConv pads the channels in the graph; on the GPU it
//  must still match a plain conv on the CPU. XCTest, not swift-testing: XCTest's bundle sees mlx's
//  metallib, so this one runs on the GPU.
//

import MLX
import MLXExactConv
import MLXNN
import XCTest
@testable import NAFNetMLXCore

final class PadHelperTests: XCTestCase {
    func testEndingConvIsRoutedOffThePadHelper() {
        let net = NAFNet()
        guard let conv = net.ending as? RoutedConv2d else {
            return XCTFail("the ending conv is not routed")
        }
        MLXRandom.seed(4)
        conv.update(parameters: conv.parameters().mapValues { MLXRandom.normal($0.shape) * 0.05 })
        let x = MLXRandom.normal([1, 64, 48, 24])
        XCTAssertTrue(
            ExactConv.takesPadHelper2D(input: x, weight: conv.weight, padding: (1, 1)),
            "a plain Conv2d at this shape would take mlx's pad helper")

        let gpu = Device.withDefaultDevice(Device(.gpu)) { conv(x) }
        let cpu = Device.withDefaultDevice(Device(.cpu)) { conv2d(x, conv.weight, padding: 1) + conv.bias! }
        eval(gpu, cpu)
        XCTAssertEqual(gpu.shape, [1, 64, 48, 3])
        XCTAssertLessThan(abs(gpu - cpu).max().item(Float.self), 1e-4)
    }
}
