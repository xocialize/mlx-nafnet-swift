// CancellationTests.swift — NAFNet through the engine's CAN gate (offline, no MLX kernels).
// CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before notLoaded
// validation or image decode); CAN-3 is the document of record for the checkpoint cadence.
// NAFNet is a CAN-3 judgment case: run() is a SINGLE full-frame forward — one monolithic lazy
// MLX graph realized in the NHWC→pixel-buffer materialization, no tiling, no iterative loop —
// yet peakActivationBytes (5.79 GB fp16 / 8.0 GB fp32) ≥ 2 GB makes the manifest long-run
// implied. The declared cadence therefore names the REAL seams run() has, once per restored
// frame each:
//   • entry checkpoint (first act of run(), NAFNetRestorePackage.run)
//   • pre-forward, after input decode/NHWC conversion (last seam before the monolithic eval)
//   • post-forward / pre-encode (after the restored pixel buffer materializes)
// No per-step checkpoints are fabricated — there is no loop to hang them on.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXNAFNet

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = NAFNetRestorePackage(configuration: NAFNetConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ImageRestoreRequest(image: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // peakActivationBytes ≥ 2 GB on both declared footprints ⇒ long-run implied — the
        // sub-second exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: NAFNetRestorePackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: NAFNetRestorePackage.manifest,
            posture: .cadence([
                // Pre-forward seam: once per frame, after decode/NHWC conversion, before
                // committing to the monolithic full-frame eval.
                .init(phase: .encode, unit: .frame),
                // Post-forward seam: once per frame, between materialization and output encode.
                .init(phase: .postprocess, unit: .frame),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
