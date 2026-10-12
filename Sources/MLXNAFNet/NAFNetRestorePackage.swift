import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import MLXProfiling
import Hub
import NAFNetMLXCore

/// Errors at the NAFNet package boundary.
public enum NAFNetPackageError: Error, Equatable {
    case imageDecodeFailed(String)
    case imageEncodeFailed
    case bundledWeightsMissing
}

/// An MLXEngine `imageRestore` package over **NAFNet** — denoise / deblock / encode-artifact
/// removal at constant resolution. The first transform capability of the visual optimization
/// tier; the Layer-3 planner pairs it with `imageQualityScore` ("restore only where it pays").
///
/// A thin conformance wrapper over the standalone `NAFNetMLXCore` core (nafnet-mlx-swift); all model
/// logic (NAFNet blocks, fp32 global-pool convention, NHWC) lives there. The default variant is
/// the bundled signage-trained width-24 checkpoint (no download); public width-64 mlx-community
/// checkpoints are selectable via configuration.
///
/// **16 bits end to end** (contract 1.52.0, AB-T-0230): the surface declares `.rawRGBA16`. A `.rawRGBA16` image, or a
/// PNG/TIFF deeper than 8 bits, is read at 16 bits, runs the same full-frame forward (float32 activations: the input is
/// float32, so the fp16 signage weights promote) and comes back at 16 bits: `.rawRGBA16` for raw, a 16-bit PNG otherwise.
/// 8-bit inputs take the 8-bit path byte for byte.
@InferenceActor
public final class NAFNetRestorePackage: ModelPackage {
    public typealias Configuration = NAFNetConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // NAFNet (megvii-research) is MIT; bundled + mlx-community checkpoints MIT; port MIT.
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "mlx-community/NAFNet-SIDD-width64",
                                   revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (engine 1.14). NAFNet runs the **full frame** (no internal tiling),
                // so the working set is overwhelmingly activation — weights are tiny. Re-measured via
                // RE-BASELINED to in-app phys_footprint (MLXEngineImage, isolate+clearCache, engine 0.17.0
                // post-load floor). The `nafnet-smoke` MLX-peak under-read the admission-relevant phys by
                // ~2.9× (MLX cache + process overhead). Floor pass-1 (post-load) confirmed the true weights
                // resident; the ~5.4 GB the smoke/old-harness mis-read as "floor" was post-run retention —
                // a BOUNDED working buffer (flat across 3 resident passes, freed on evict), so it belongs in
                // the transient, not residency:
                //   • signage (w24, fp16): floor 0.07 GB · peak 5.86 GB → resident 70 MB / activation 5.79 GB (MEASURED)
                //   • width64 (publics, fp32): NOT yet measured in-app; smoke 0.5/2.9 GB scaled by the ~2.9×
                //     phys gap → activation ~8.0 GB FLAGGED-pending an in-app width64 run (conservative —
                //     under-declaring would falsely admit on tight Macs, the BiRefNet-best failure mode).
                // residentBytes = post-load weights floor; peakActivationBytes = phys peak − floor. The engine
                // reserves ONE shared transient across residents — the co-residency win for the optimizer
                // chain. `QuantConfigured` (NAFNetConfiguration) charges the per-variant quant.
                footprints: [
                    QuantFootprint(quant: .fp16, residentBytes:    70_000_000, peakActivationBytes: 5_790_000_000),  // signage w24 (in-app phys)
                    QuantFootprint(quant: .fp32, residentBytes:   512_000_000, peakActivationBytes: 8_000_000_000),  // width64 publics (FLAGGED: smoke-scaled, in-app pending)
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                ImageRestoreContract.descriptor(
                    name: "nafnet-restore",
                    summary: "NAFNet image restoration: denoise / deblock / encode-artifact removal at constant resolution."
                )
                // Mirrors rawBGRA8 (since 0.1.0) and keeps 16 bits for rawRGBA16 and deep PNG (0.6.0).
                .declaring(rawImageFormats: [.rawBGRA8, .rawRGBA16])
            ]
        )
    }

    private let configuration: Configuration
    private var model: NAFNet?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard model == nil else { return }
        let arch = configuration.variant.architecture
        let net = NAFNet(width: arch.width,
                         middleBlockNum: arch.middle,
                         encoderBlockNums: arch.enc,
                         decoderBlockNums: arch.dec)

        let weightsURL: URL
        if let repo = configuration.variant.repo,
           let stored = StoreWeights.file("model.safetensors", repo: repo, root: configuration.modelsRootDirectory) {
            // Materialized by the engine before load() (or by 0.4.x, in the HubApi layout).
            weightsURL = stored
        } else if let repo = configuration.variant.repo {
            // Backstop for a caller outside the engine: download as before, into the HubApi layout.
            let hub = configuration.modelsRootDirectory.map { HubApi(downloadBase: $0) } ?? HubApi()
            // Forward download progress to the engine's ambient sink so prepare() surfaces a real
            // `.downloading(fraction:…)` phase (MLXEngineUI ModelStateView). No-op when unbound.
            let dir = try await hub.snapshot(from: Hub.Repo(id: repo),
                                             matching: ["model.safetensors"]) { progress, speed in
                WeightDownloadProgress.report(fraction: progress.fractionCompleted, bytesPerSecond: speed)
            }
            weightsURL = dir.appendingPathComponent("model.safetensors")
        } else {
            guard let bundled = bundledSignageWeightsURL else {
                throw NAFNetPackageError.bundledWeightsMissing
            }
            weightsURL = bundled
        }
        try net.loadWeights(from: weightsURL)
        model = net
    }

    public func unload() async {
        model = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation
        // (engine ≥ 0.27.0). Mid-run: the full-frame forward is ONE monolithic MLX eval (no
        // tiling, no iterative loop); the real seams are pre-forward (post-decode) and
        // post-forward/pre-encode below — see CancellationTests for the cadence of record.
        try Task.checkCancellation()
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .imageRestore,
              let req = request as? ImageRestoreRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        // Image (.png/.jpeg/.rawBGRA8/.rawRGBA16) → BGRA or RGBA16 pixel buffer → NHWC [0,1] → NAFNet → pixel buffer
        // of the same depth → out. Output mirrors the input format: rawBGRA8 in ⇒ rawBGRA8 out, rawRGBA16 in ⇒
        // rawRGBA16 out (no re-encode); else .png — 16-bit when the input was deep.
        let pb = try Self.decodeToPixelBuffer(req.image)
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let deep = isDeep(pb)
        guard let x = deep ? rgbNHWC(fromRGBA16: pb, width: w, height: h)
                           : rgbNHWC(from: ensureBGRA(pb), width: w, height: h) else {
            throw NAFNetPackageError.imageDecodeFailed("NHWC conversion (\(w)x\(h))")
        }
        // The single full-frame forward is profiled (MLX_PROFILE=1). The graph is lazy end-to-end;
        // cost realizes in the NHWC→pixel-buffer `asArray` materialization, so ONE coarse region
        // wraps forward + conversion and times the whole graph via the pipeline's own eval
        // (rife-style — no added evals). Weights load in `load()` (engine prepare), so beginRun
        // at run entry can't be skewed by a first-run download.
        // Pre-forward checkpoint: last seam before committing to the monolithic full-frame eval.
        try Task.checkCancellation()
        let prof = MLXProfiler.shared
        prof.beginRun("nafnet imageRestore \(configuration.variant) \(w)x\(h)")
        let restoredPB = prof.region("restore", "forward") {
            deep ? pixelBufferRGBA16(fromRGBNHWC: model(x), width: w, height: h)
                 : pixelBuffer(fromRGBNHWC: model(x), width: w, height: h)
        }
        prof.endRun(denominators: ["image": 1])
        guard let outPB = restoredPB else {
            throw NAFNetPackageError.imageEncodeFailed
        }
        // Post-forward checkpoint: between the forward's materialization and output encode.
        try Task.checkCancellation()
        let outImage: Image
        if req.image.format == .rawBGRA8 {
            guard let raw = Self.encodeRawBGRA8(outPB) else { throw NAFNetPackageError.imageEncodeFailed }
            outImage = raw
        } else if req.image.format == .rawRGBA16 {
            guard let raw = Self.encodeRawRGBA16(outPB) else { throw NAFNetPackageError.imageEncodeFailed }
            outImage = raw
        } else {
            guard let png = Self.encodePNG(outPB) else { throw NAFNetPackageError.imageEncodeFailed }
            outImage = Image(format: .png, data: png, width: w, height: h)
        }
        return ImageRestoreResponse(image: outImage)
    }

    // MARK: - Image codec

    /// Decode a canonical `Image` (.png/.jpeg/.rawBGRA8/.rawRGBA16) to a `CVPixelBuffer`: BGRA, or 16-bit RGBA
    /// (`deepPixelFormat`) for `.rawRGBA16` and for an encoded image deeper than 8 bits per component.
    nonisolated static func decodeToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        if image.format == .rawBGRA8 { return try rawBGRA8ToPixelBuffer(image) }
        if image.format == .rawRGBA16 { return try rawRGBA16ToPixelBuffer(image) }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw NAFNetPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        if cg.bitsPerComponent > 8 { return try deepPixelBuffer(drawing: cg) }
        let w = cg.width, h = cg.height
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw NAFNetPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw NAFNetPackageError.imageDecodeFailed("CGContext for BGRA draw")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Encode a BGRA `CVPixelBuffer` as PNG bytes (a 16-bit PNG for a `deepPixelFormat` buffer).
    nonisolated static func encodePNG(_ pb: CVPixelBuffer) -> Data? {
        if isDeep(pb) {
            guard let cg = deepCGImage(pb) else { return nil }
            let out = NSMutableData()
            guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
            CGImageDestinationAddImage(dest, cg, nil)
            return CGImageDestinationFinalize(dest) ? out as Data : nil
        }
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    /// Wrap raw interleaved BGRA8 bytes straight into a 32BGRA `CVPixelBuffer` — no `CGImageSource`,
    /// no decode. `width`/`height` are required for `.rawBGRA8`; `bytesPerRow` is the source row stride
    /// (defaults to tightly packed `width * 4`).
    nonisolated static func rawBGRA8ToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        guard let w = image.width, let h = image.height, w > 0, h > 0 else {
            throw NAFNetPackageError.imageDecodeFailed("rawBGRA8 requires width/height")
        }
        let srcStride = image.bytesPerRow ?? (w * 4)
        guard srcStride >= w * 4, image.data.count >= srcStride * h else {
            throw NAFNetPackageError.imageDecodeFailed(
                "rawBGRA8 data too small (\(image.data.count) < \(srcStride * h))")
        }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw NAFNetPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw NAFNetPackageError.imageDecodeFailed("pixel buffer base address")
        }
        let dstStride = CVPixelBufferGetBytesPerRow(buffer)
        let rowBytes = min(srcStride, dstStride)
        image.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let srcBase = src.baseAddress else { return }
            for row in 0..<h {
                memcpy(base.advanced(by: row * dstStride), srcBase.advanced(by: row * srcStride), rowBytes)
            }
        }
        return buffer
    }

    /// Emit a 32BGRA `CVPixelBuffer` as tightly-packed raw BGRA8 `Image` bytes (no compression/clamp).
    nonisolated static func encodeRawBGRA8(_ pb: CVPixelBuffer) -> Image? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard w > 0, h > 0 else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let srcStride = CVPixelBufferGetBytesPerRow(pb)
        let dstStride = w * 4
        var out = Data(count: dstStride * h)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let dstBase = dst.baseAddress else { return }
            for row in 0..<h {
                memcpy(dstBase.advanced(by: row * dstStride), base.advanced(by: row * srcStride), dstStride)
            }
        }
        return Image.rawBGRA8(data: out, width: w, height: h)
    }

    // MARK: - 16 bits (contract 1.52.0, AB-T-0230; the mlx-realplksr-swift 0.2.0 codec)

    /// RGBA, 16 bits per component, little-endian, premultiplied — `.rawRGBA16`'s layout as CoreGraphics draws it.
    nonisolated static let rgba16BitmapInfo =
        CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder16Little.rawValue

    nonisolated static func makeDeepPixelBuffer(_ w: Int, _ h: Int) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: deepPixelFormat,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, deepPixelFormat, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw NAFNetPackageError.imageDecodeFailed("16-bit pixel buffer allocation (\(w)x\(h))")
        }
        return buffer
    }

    /// Draw `cg` into a new `deepPixelFormat` buffer, in sRGB at 16 bits.
    nonisolated static func deepPixelBuffer(drawing cg: CGImage) throws -> CVPixelBuffer {
        let w = cg.width, h = cg.height
        let buffer = try makeDeepPixelBuffer(w, h)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 16,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: rgba16BitmapInfo) else {
            throw NAFNetPackageError.imageDecodeFailed("CGContext for RGBA16 draw")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Wrap raw interleaved RGBA16 bytes straight into a `deepPixelFormat` buffer — no decode.
    nonisolated static func rawRGBA16ToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        guard let w = image.width, let h = image.height, w > 0, h > 0 else {
            throw NAFNetPackageError.imageDecodeFailed("rawRGBA16 requires width/height")
        }
        let srcStride = image.bytesPerRow ?? (w * 8)
        guard srcStride >= w * 8, image.data.count >= srcStride * h else {
            throw NAFNetPackageError.imageDecodeFailed(
                "rawRGBA16 data too small (\(image.data.count) < \(srcStride * h))")
        }
        let buffer = try makeDeepPixelBuffer(w, h)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw NAFNetPackageError.imageDecodeFailed("pixel buffer base address")
        }
        let dstStride = CVPixelBufferGetBytesPerRow(buffer)
        image.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let srcBase = src.baseAddress else { return }
            for row in 0..<h {
                memcpy(base.advanced(by: row * dstStride), srcBase.advanced(by: row * srcStride), w * 8)
            }
        }
        return buffer
    }

    /// A `deepPixelFormat` buffer as a 16-bit sRGB `CGImage` over a copy of its pixels.
    nonisolated static func deepCGImage(_ pb: CVPixelBuffer) -> CGImage? {
        guard let raw = encodeRawRGBA16(pb), let provider = CGDataProvider(data: raw.data as CFData) else { return nil }
        return CGImage(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb), bitsPerComponent: 16,
                       bitsPerPixel: 64, bytesPerRow: CVPixelBufferGetWidth(pb) * 8,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: rgba16BitmapInfo),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    /// Emit a `deepPixelFormat` buffer as tightly-packed `.rawRGBA16` bytes.
    nonisolated static func encodeRawRGBA16(_ pb: CVPixelBuffer) -> Image? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard w > 0, h > 0, isDeep(pb) else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let srcStride = CVPixelBufferGetBytesPerRow(pb)
        let dstStride = w * 8
        var out = Data(count: dstStride * h)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let dstBase = dst.baseAddress else { return }
            for row in 0..<h {
                memcpy(dstBase.advanced(by: row * dstStride), base.advanced(by: row * srcStride), dstStride)
            }
        }
        return Image.rawRGBA16(data: out, width: w, height: h)
    }
}

extension NAFNetRestorePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(NAFNetRestorePackage.self)
    }
}
