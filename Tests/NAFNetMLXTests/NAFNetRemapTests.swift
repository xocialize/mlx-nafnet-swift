import XCTest
@testable import NAFNetMLX

/// Locks the upstream → port key remap that lets the mlx-community `NAFNet-*-width64` checkpoints
/// (official PyTorch key layout) load into this port's nested module tree. Proven exhaustively
/// offline (all 664 SIDD-width64 keys → exactly the port's width64 key set); these cases guard the
/// transform from regressing. Pure string logic — no MLX, no weights.
final class NAFNetRemapTests: XCTestCase {
    func testStageAndBlockRemap() {
        let cases: [(String, String)] = [
            // separate up/down ModuleLists → nested in the decoder/encoder stage
            ("ups.0.weight",            "decoders.0.upConv.weight"),
            ("downs.1.weight",          "encoders.1.down.weight"),
            ("downs.1.bias",            "encoders.1.down.bias"),
            // flat block arrays → blocks.layers.j
            ("encoders.3.7.conv2.weight", "encoders.3.blocks.layers.7.conv2.weight"),
            ("decoders.0.1.conv5.bias",   "decoders.0.blocks.layers.1.conv5.bias"),
            ("middle_blks.11.conv3.weight", "middle_blks.layers.11.conv3.weight"),
            // block-internal wrapped submodules
            ("decoders.0.0.norm1.weight", "decoders.0.blocks.layers.0.norm1.norm.weight"),
            ("encoders.0.0.norm2.bias",   "encoders.0.blocks.layers.0.norm2.norm.bias"),
            ("decoders.0.1.sca.bias",     "decoders.0.blocks.layers.1.sca.conv.bias"),
            ("middle_blks.0.sca.weight",  "middle_blks.layers.0.sca.conv.weight"),
            // 1:1 passthroughs
            ("encoders.0.0.beta",         "encoders.0.blocks.layers.0.beta"),
            ("intro.weight",              "intro.weight"),
            ("ending.bias",               "ending.bias"),
        ]
        for (input, expected) in cases {
            XCTAssertEqual(NAFNet.remapUpstreamKey(input), expected, "remap of \(input)")
        }
    }

    /// Layout detection: the bundled signage (port-layout) keys must NOT be detected as upstream (so
    /// they skip the remap entirely), while the mlx-community checkpoint keys must be.
    func testLayoutDetection() {
        let portKeys = [
            "decoders.0.upConv.weight",
            "decoders.0.blocks.layers.0.norm1.norm.weight",
            "encoders.0.down.weight",
            "intro.weight",
        ]
        XCTAssertFalse(NAFNet.isUpstreamLayout(portKeys), "port layout must not be flagged upstream")

        let upstreamKeys = ["intro.weight", "downs.0.weight", "ups.0.weight", "decoders.0.0.norm1.weight"]
        XCTAssertTrue(NAFNet.isUpstreamLayout(upstreamKeys), "mlx-community layout must be flagged upstream")
    }
}
