import Testing
import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLXToolKit
import NAFNetMLX
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
