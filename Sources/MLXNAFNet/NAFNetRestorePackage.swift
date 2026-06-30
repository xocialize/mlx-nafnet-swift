import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLXToolKit
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
                // `nafnet-smoke` through the real MLXServeEngine at a 1024×1024 envelope (see
                // EFFICIENCY-ADOPTION.md):
                //   • signage (w24, fp16): floor 5 MB · peak 2034 MB → resident 64 MB / activation 2.0 GB
                //   • width64 (publics, fp32): floor 443 MB · peak 3271 MB → resident 512 MB / activation 2.9 GB
                // residentBytes = weights floor (+ overhead); peakActivationBytes = peak − floor. The
                // engine reserves ONE shared transient across residents, so the activation no longer
                // bakes into residency — the co-residency win for the optimizer chain. `QuantConfigured`
                // (NAFNetConfiguration) charges the per-variant quant. Re-measure if the envelope > 1024².
                footprints: [
                    QuantFootprint(quant: .fp16, residentBytes:    64_000_000, peakActivationBytes: 2_000_000_000),  // signage w24
                    QuantFootprint(quant: .fp32, residentBytes:   512_000_000, peakActivationBytes: 2_900_000_000),  // width64 publics
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
        if let repo = configuration.variant.repo {
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
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .imageRestore,
              let req = request as? ImageRestoreRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        // Image (.png/.jpeg/.rawBGRA8) → BGRA pixel buffer → NHWC [0,1] → NAFNet → pixel buffer → out.
        // Output mirrors the input format: rawBGRA8 in ⇒ rawBGRA8 out (no re-encode); else .png.
        let pb = try Self.decodeToPixelBuffer(req.image)
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard let x = rgbNHWC(from: ensureBGRA(pb), width: w, height: h) else {
            throw NAFNetPackageError.imageDecodeFailed("NHWC conversion (\(w)x\(h))")
        }
        let restored = model(x)
        guard let outPB = pixelBuffer(fromRGBNHWC: restored, width: w, height: h) else {
            throw NAFNetPackageError.imageEncodeFailed
        }
        let outImage: Image
        if req.image.format == .rawBGRA8 {
            guard let raw = Self.encodeRawBGRA8(outPB) else { throw NAFNetPackageError.imageEncodeFailed }
            outImage = raw
        } else {
            guard let png = Self.encodePNG(outPB) else { throw NAFNetPackageError.imageEncodeFailed }
            outImage = Image(format: .png, data: png, width: w, height: h)
        }
        return ImageRestoreResponse(image: outImage)
    }

    // MARK: - Image codec

    /// Decode a canonical `Image` (.png/.jpeg/.rawBGRA8) to a BGRA `CVPixelBuffer`.
    nonisolated static func decodeToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        if image.format == .rawBGRA8 { return try rawBGRA8ToPixelBuffer(image) }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw NAFNetPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
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

    /// Encode a BGRA `CVPixelBuffer` as PNG bytes.
    nonisolated static func encodePNG(_ pb: CVPixelBuffer) -> Data? {
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
}

extension NAFNetRestorePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(NAFNetRestorePackage.self)
    }
}
