import Testing
import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import NAFNetMLXCore
@testable import MLXNAFNet

/// The 16-bit boundary (contract 1.52.0, AB-T-0230): `.rawRGBA16` and deep PNG decode into a 16-bit RGBA buffer, run
/// the forward at float32 and encode back at 16 bits. The codec tests are CoreGraphics and CoreVideo only; the forward
/// test pins MLX to the CPU (no metallib under `swift test`).
struct DeepBoundaryTests {

    /// A 16-bit gradient: ~0.1 of an 8-bit step per pixel in red, so an 8-bit hop would collapse it.
    static func deepWords(_ w: Int, _ h: Int) -> [UInt16] {
        var words = [UInt16](repeating: 0, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 4
                words[i] = UInt16(30_000 + x * 25); words[i + 1] = UInt16(10_000 + y * 7)
                words[i + 2] = UInt16((x * 977 + y * 131) % 65_536); words[i + 3] = UInt16.max
            }
        }
        return words
    }

    static func deepPNG(_ w: Int, _ h: Int) throws -> Data {
        var words = deepWords(w, h)
        let cg = try #require(words.withUnsafeMutableBytes { raw in
            CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 16, bytesPerRow: w * 8,
                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                      bitmapInfo: NAFNetRestorePackage.rgba16BitmapInfo)?.makeImage()
        })
        let out = NSMutableData()
        let dest = try #require(CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(dest, cg, nil)
        #expect(CGImageDestinationFinalize(dest))
        return out as Data
    }

    @Test func surfaceDeclaresTheSixteenBitBoundary() {
        let surface = NAFNetRestorePackage.manifest.surfaces[0]
        #expect(surface.rawImageFormats == [.rawBGRA8, .rawRGBA16])
        #expect(surface.acceptsRawRGBA16)
    }

    @Test func rawRGBA16RoundTripsBitIdentical() throws {
        let (w, h) = (9, 5)
        let bytes = Self.deepWords(w, h).withUnsafeBytes { Data($0) }
        let pb = try NAFNetRestorePackage.decodeToPixelBuffer(.rawRGBA16(data: bytes, width: w, height: h))
        #expect(CVPixelBufferGetPixelFormatType(pb) == deepPixelFormat)
        let back = try #require(NAFNetRestorePackage.encodeRawRGBA16(pb))
        #expect(back.format == .rawRGBA16)
        #expect(back.width == w && back.height == h)
        #expect(back.data == bytes)
    }

    @Test func rawRGBA16HonorsSourceStride() throws {
        let (w, h) = (3, 2)
        let words = Self.deepWords(w, h)
        var strided = Data(count: 32 * h)      // 24 bytes of pixels + 8 of padding per row
        for row in 0..<h {
            let rowBytes = words[(row * w * 4)..<((row + 1) * w * 4)].withUnsafeBytes { Data($0) }
            strided.replaceSubrange((row * 32)..<(row * 32 + 24), with: rowBytes)
        }
        let pb = try NAFNetRestorePackage.decodeToPixelBuffer(.rawRGBA16(data: strided, width: w, height: h, bytesPerRow: 32))
        let back = try #require(NAFNetRestorePackage.encodeRawRGBA16(pb))
        #expect(back.data == words.withUnsafeBytes { Data($0) })
    }

    @Test func rawRGBA16WithoutDimensionsOrBytesThrows() {
        #expect(throws: NAFNetPackageError.self) {
            _ = try NAFNetRestorePackage.decodeToPixelBuffer(Image(format: .rawRGBA16, data: Data(count: 64)))
        }
        #expect(throws: NAFNetPackageError.self) {
            _ = try NAFNetRestorePackage.decodeToPixelBuffer(.rawRGBA16(data: Data(count: 8), width: 2, height: 2))
        }
    }

    /// A 16-bit PNG decodes at 16 bits (exactly, for an opaque sRGB source) and encodes back to a 16-bit PNG.
    @Test func deepPNGDecodesAndEncodesAtSixteenBits() throws {
        let (w, h) = (16, 8)
        let pb = try NAFNetRestorePackage.decodeToPixelBuffer(Image(format: .png, data: try Self.deepPNG(w, h), width: w, height: h))
        #expect(CVPixelBufferGetPixelFormatType(pb) == deepPixelFormat)
        let raw = try #require(NAFNetRestorePackage.encodeRawRGBA16(pb))
        #expect(raw.data == Self.deepWords(w, h).withUnsafeBytes { Data($0) })

        let png = try #require(NAFNetRestorePackage.encodePNG(pb))
        let source = try #require(CGImageSourceCreateWithData(png as CFData, nil))
        let cg = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        #expect(cg.bitsPerComponent == 16)
    }

    /// An 8-bit PNG keeps the 8-bit path: a 32BGRA buffer and an 8-bit PNG back.
    @Test func eightBitPNGKeepsTheEightBitPath() throws {
        let png = try #require(NAFNetRestoreTests.makePNG(width: 8, height: 8))
        let pb = try NAFNetRestorePackage.decodeToPixelBuffer(Image(format: .png, data: png, width: 8, height: 8))
        #expect(CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_32BGRA)
        let back = try #require(NAFNetRestorePackage.encodePNG(pb))
        let source = try #require(CGImageSourceCreateWithData(back as CFData, nil))
        let cg = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        #expect(cg.bitsPerComponent == 8)
    }

    /// The NHWC converters: 16 bits → float → 16 bits is exact.
    @Test func nhwcConvertersRoundTripExactly() throws {
        let (w, h) = (13, 7)
        let bytes = Self.deepWords(w, h).withUnsafeBytes { Data($0) }
        let pb = try NAFNetRestorePackage.decodeToPixelBuffer(.rawRGBA16(data: bytes, width: w, height: h))
        let x = try Device.withDefaultDevice(Device(.cpu)) {
            try #require(rgbNHWC(fromRGBA16: pb, width: w, height: h))
        }
        let back = try #require(Device.withDefaultDevice(Device(.cpu)) { pixelBufferRGBA16(fromRGBNHWC: x, width: w, height: h) })
        let raw = try #require(NAFNetRestorePackage.encodeRawRGBA16(back))
        #expect(raw.data == bytes)
    }

    /// The signage forward at 16 bits (CPU): the output carries 16-bit precision, not a half-precision grid (AB-L-0238:
    /// count the levels), and a widened 8-bit image's 16-bit output agrees with the 8-bit path's.
    @Test func deepForwardKeepsSixteenBits() throws {
        let (w, h) = (64, 48)
        // A smooth 8-bit-representable photo-like field, widened ×257, plus a gentle 16-bit ramp copy.
        var bgra = [UInt8](repeating: 255, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 4
                bgra[i + 2] = UInt8(40 + x * 2); bgra[i + 1] = UInt8(60 + y * 3); bgra[i] = UInt8(120 + (x + y) % 50)
            }
        }
        var words = [UInt16](repeating: UInt16.max, count: w * h * 4)
        for p in 0..<(w * h) {
            words[p * 4] = UInt16(bgra[p * 4 + 2]) * 257
            words[p * 4 + 1] = UInt16(bgra[p * 4 + 1]) * 257
            words[p * 4 + 2] = UInt16(bgra[p * 4]) * 257
        }
        let model = NAFNet(width: 24, middleBlockNum: 1, encoderBlockNums: [1, 1, 1, 1], decoderBlockNums: [1, 1, 1, 1])
        let url = try #require(bundledSignageWeightsURL)
        let (deepOut, eightOut): (CVPixelBuffer, CVPixelBuffer) = try Device.withDefaultDevice(Device(.cpu)) {
            try model.loadWeights(from: url)
            let pb16 = try NAFNetRestorePackage.decodeToPixelBuffer(
                .rawRGBA16(data: words.withUnsafeBytes { Data($0) }, width: w, height: h))
            let pb8 = try NAFNetRestorePackage.decodeToPixelBuffer(.rawBGRA8(data: Data(bgra), width: w, height: h))
            let x16 = try #require(rgbNHWC(fromRGBA16: pb16, width: w, height: h))
            let x8 = try #require(rgbNHWC(from: pb8, width: w, height: h))
            return (try #require(pixelBufferRGBA16(fromRGBNHWC: model(x16), width: w, height: h)),
                    try #require(pixelBuffer(fromRGBNHWC: model(x8), width: w, height: h)))
        }
        let out16 = try #require(NAFNetRestorePackage.encodeRawRGBA16(deepOut)).data
            .withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
        let out8 = try #require(NAFNetRestorePackage.encodeRawBGRA8(eightOut)).data
        // A half-precision output in [⅛, 1) sits on a grid of 8–32 sixteen-bit units (an 11-bit significand), so its
        // values are ≡ 0 or 7 mod 8; a float32 output's residues are uniform.
        var midResidue = 0, offGrid = 0, worst = 0.0
        for p in 0..<(w * h) {
            for c in 0..<3 where out16[p * 4 + c] % 257 != 0 { offGrid += 1 }
            for c in 0..<3 where (2...5).contains(out16[p * 4 + c] % 8) { midResidue += 1 }
            // The 8-bit path truncates (v·255 → UInt8), so its value is the floor of the 16-bit one's 8-bit reading.
            let bgr = [2, 1, 0]
            for c in 0..<3 {
                let deep = Double(out16[p * 4 + c]) / 257
                worst = max(worst, abs(deep - Double(out8[p * 4 + bgr[c]]) - 0.5))
            }
        }
        #expect(offGrid > w * h * 3 / 2)        // the output lives off the 8-bit grid
        #expect(midResidue > w * h * 3 * 3 / 10) // 16-bit precision, not a half-precision grid (uniform ≈ 50%)
        #expect(worst <= 0.51)                   // same forward: within the 8-bit path's truncation
    }
}
