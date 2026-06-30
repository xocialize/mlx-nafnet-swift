import Testing
import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLXToolKit
import NAFNetMLXCore
@testable import MLXNAFNet

/// Offline conformance checks — no Metal evaluation. Live restoration is proven in the
/// `MLXEngine Testing` app (the model evaluates MLX ops on the GPU).
struct NAFNetRestoreTests {

    @Test func manifestIsImageRestoreAndPermissive() {
        let m = NAFNetRestorePackage.manifest
        #expect(m.capabilities == [.imageRestore])
        #expect(m.license.weightLicense == .mit)
        #expect(m.license.portCodeLicense == .mit)
        #expect(LicensePolicy.permissiveOnly.evaluate(m.license) == .admitted)
    }

    @Test func manifestRequirements() {
        let r = NAFNetRestorePackage.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.os.minMacOS == SemanticVersion(major: 26, minor: 0, patch: 0))
        #expect(r.footprints.contains { $0.quant == .fp16 })
    }

    /// Efficiency adoption (engine 1.14): both footprints declare the split (resident weights floor +
    /// transient activation peak), so the engine reserves one shared transient across residents.
    @Test func splitFootprintDeclared() {
        let fps = NAFNetRestorePackage.manifest.requirements.footprints
        let fp16 = fps.first { $0.quant == .fp16 }
        let fp32 = fps.first { $0.quant == .fp32 }
        #expect(fp16?.peakActivationBytes ?? 0 > 0)        // activation declared, not 0
        #expect(fp32?.peakActivationBytes ?? 0 > 0)
        // NAFNet is activation-dominated: the transient peak dwarfs the resident weights floor.
        #expect((fp16?.peakActivationBytes ?? 0) > (fp16?.residentBytes ?? .max))
        #expect((fp32?.peakActivationBytes ?? 0) > (fp32?.residentBytes ?? .max))
    }

    /// `QuantConfigured` so the governor charges the per-variant declared `QuantFootprint`.
    @Test func quantConfigured() {
        let cfg: any PackageConfiguration = NAFNetConfiguration(variant: .signage)
        #expect((cfg as? QuantConfigured)?.quant == .fp16)
        let cfg64: any PackageConfiguration = NAFNetConfiguration(variant: .siddWidth64)
        #expect((cfg64 as? QuantConfigured)?.quant == .fp32)
    }

    @Test func surfaceIsTheCanonicalRestoreDescriptor() {
        let s = NAFNetRestorePackage.manifest.surfaces.first
        #expect(s?.capability == .imageRestore)
        #expect(s?.parameters.first?.kind == .image)
    }

    @Test func registrationConstructs() throws {
        let reg = NAFNetRestorePackage.registration
        #expect(reg.manifest.capabilities == [.imageRestore])
        let pkg = try reg.makePackage(NAFNetConfiguration())
        #expect(pkg is NAFNetRestorePackage)
    }

    @Test func variantsMapToCheckpoints() {
        #expect(NAFNetConfiguration().variant == .signage)
        #expect(NAFNetVariant.signage.repo == nil)                     // bundled, no download
        #expect(NAFNetVariant.siddWidth64.repo == "mlx-community/NAFNet-SIDD-width64")
        #expect(NAFNetVariant.signage.architecture.width == 24)
        #expect(NAFNetVariant.siddWidth64.architecture.width == 64)
        #expect(NAFNetVariant.siddWidth64.architecture.enc == [2, 2, 4, 8])
    }

    @Test func bundledSignageWeightsAreReachable() {
        #expect(bundledSignageWeightsURL != nil)
    }

    @Test func configurationCodableExcludesEnvironmentRoot() throws {
        var c = NAFNetConfiguration(variant: .goproWidth64)
        c.modelsRootDirectory = URL(fileURLWithPath: "/tmp/should-not-persist")
        let back = try JSONDecoder().decode(NAFNetConfiguration.self, from: JSONEncoder().encode(c))
        #expect(back.variant == .goproWidth64)
        #expect(back.modelsRootDirectory == nil)
    }

    @Test func pngRoundTripsThroughPixelBuffer() throws {
        // Decode + encode are pure CoreGraphics/CoreImage (no Metal): PNG → BGRA buffer → PNG.
        let png = try #require(Self.makePNG(width: 24, height: 24))
        let image = Image(format: .png, data: png, width: 24, height: 24)
        let pb = try NAFNetRestorePackage.decodeToPixelBuffer(image)
        #expect(CVPixelBufferGetWidth(pb) == 24)
        let back = try #require(NAFNetRestorePackage.encodePNG(pb))
        #expect(back.prefix(4) == Data([0x89, 0x50, 0x4E, 0x47]))
    }

    // MARK: - rawBGRA8 (contract 1.9.0)

    @Test func rawBGRA8RoundTripsBitIdentical() throws {
        // 8x4 tightly-packed BGRA8 ramp → buffer → rawBGRA8, bytes must survive exactly.
        let w = 8, h = 4
        let bytes = Data((0..<(w * h * 4)).map { UInt8($0 % 256) })
        let image = Image.rawBGRA8(data: bytes, width: w, height: h)
        let pb = try NAFNetRestorePackage.decodeToPixelBuffer(image)
        #expect(CVPixelBufferGetWidth(pb) == w)
        #expect(CVPixelBufferGetHeight(pb) == h)
        let back = try #require(NAFNetRestorePackage.encodeRawBGRA8(pb))
        #expect(back.format == .rawBGRA8)
        #expect(back.width == w && back.height == h)
        #expect(back.bytesPerRow == nil)        // tightly packed on emit
        #expect(back.data == bytes)             // bit-identical
    }

    @Test func rawBGRA8HonorsSourceStride() throws {
        // Padded source rows (stride = w*4 + 16) must decode to the correct pixels, not garbage.
        let w = 5, h = 3
        let stride = w * 4 + 16
        var src = Data(count: stride * h)
        for row in 0..<h {
            for col in 0..<(w * 4) { src[row * stride + col] = UInt8((row * 40 + col) % 256) }
        }
        let image = Image.rawBGRA8(data: src, width: w, height: h, bytesPerRow: stride)
        let pb = try NAFNetRestorePackage.decodeToPixelBuffer(image)
        let back = try #require(NAFNetRestorePackage.encodeRawBGRA8(pb))   // tightly packed
        #expect(back.data.count == w * 4 * h)
        for row in 0..<h {
            let expected = src.subdata(in: (row * stride)..<(row * stride + w * 4))
            let got = back.data.subdata(in: (row * w * 4)..<((row + 1) * w * 4))
            #expect(got == expected)
        }
    }

    @Test func rawBGRA8MissingDimensionsThrows() {
        let image = Image(format: .rawBGRA8, data: Data(count: 16))   // no width/height
        #expect(throws: NAFNetPackageError.self) {
            _ = try NAFNetRestorePackage.decodeToPixelBuffer(image)
        }
    }

    @Test func pngAndRawBGRA8DecodeToSamePixels() throws {
        // PNG path and rawBGRA8 path must yield identical pixel buffers for the same content.
        let png = try #require(Self.makePNG(width: 16, height: 16))
        let viaPNG = try NAFNetRestorePackage.decodeToPixelBuffer(Image(format: .png, data: png, width: 16, height: 16))
        let raw = try #require(NAFNetRestorePackage.encodeRawBGRA8(viaPNG))
        let viaRaw = try NAFNetRestorePackage.decodeToPixelBuffer(raw)
        let reRaw = try #require(NAFNetRestorePackage.encodeRawBGRA8(viaRaw))
        #expect(reRaw.data == raw.data)   // PNG → raw → buffer → raw is stable
    }

    static func makePNG(width: Int, height: Int) -> Data? {
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.setFillColor(CGColor(gray: 0.3, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let cg = ctx.makeImage() else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }
}
