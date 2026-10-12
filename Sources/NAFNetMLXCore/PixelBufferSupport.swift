//
//  PixelBufferSupport.swift
//  NAFNetMLXCore
//
//  CVPixelBuffer ↔ MLXArray round-trip utilities for full-frame restoration.
//  Extracted from forge-studio-optimizer's NAFNetProcessor / PixelBufferBridge.
//
//  NV12 hazard (forge convention): any byte-level reader must convert to BGRA
//  first — reading NV12 base addresses as packed pixels yields sheared garbage.
//

import CoreImage
import CoreVideo
import Foundation
import MLX

/// Convert any `CVPixelBuffer` to 32BGRA via CoreImage (no-op when already BGRA).
public func ensureBGRA(_ input: CVPixelBuffer) -> CVPixelBuffer {
    if CVPixelBufferGetPixelFormatType(input) == kCVPixelFormatType_32BGRA {
        return input
    }
    let w = CVPixelBufferGetWidth(input)
    let h = CVPixelBufferGetHeight(input)
    var out: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: w,
        kCVPixelBufferHeightKey as String: h,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &out) == kCVReturnSuccess,
          let buffer = out else { return input }
    let ctx = CIContext(options: [.cacheIntermediates: false])
    ctx.render(CIImage(cvPixelBuffer: input), to: buffer)
    return buffer
}

/// BGRA `CVPixelBuffer` → `[1, H, W, 3]` RGB float NHWC in [0,1].
public func rgbNHWC(from bgra: CVPixelBuffer, width: Int, height: Int) -> MLXArray? {
    CVPixelBufferLockBaseAddress(bgra, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(bgra, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(bgra) else { return nil }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(bgra)
    let src = base.assumingMemoryBound(to: UInt8.self)

    var rgb = [Float](repeating: 0, count: height * width * 3)
    for y in 0 ..< height {
        let row = y * bytesPerRow
        let drow = y * width * 3
        for x in 0 ..< width {
            let s = row + x * 4
            let d = drow + x * 3
            rgb[d + 0] = Float(src[s + 2]) / 255.0  // R  (BGRA byte 2)
            rgb[d + 1] = Float(src[s + 1]) / 255.0  // G
            rgb[d + 2] = Float(src[s + 0]) / 255.0  // B
        }
    }
    return MLXArray(rgb, [1, height, width, 3])
}

/// `[1, H, W, 3]` RGB float (any range; clamped) → BGRA `CVPixelBuffer`.
public func pixelBuffer(fromRGBNHWC array: MLXArray, width: Int, height: Int) -> CVPixelBuffer? {
    let rgb = array.asArray(Float.self)
    guard rgb.count >= width * height * 3 else { return nil }

    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                              attrs as CFDictionary, &pb) == kCVReturnSuccess,
          let buffer = pb else { return nil }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let dstBase = CVPixelBufferGetBaseAddress(buffer) else { return nil }
    let dst = dstBase.assumingMemoryBound(to: UInt8.self)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

    @inline(__always) func clamp(_ v: Float) -> UInt8 {
        UInt8(max(0, min(255, v * 255)))
    }
    for y in 0 ..< height {
        let row = y * bytesPerRow
        let srow = y * width * 3
        for x in 0 ..< width {
            let d = row + x * 4
            let s = srow + x * 3
            dst[d + 0] = clamp(rgb[s + 2])  // B
            dst[d + 1] = clamp(rgb[s + 1])  // G
            dst[d + 2] = clamp(rgb[s + 0])  // R
            dst[d + 3] = 255
        }
    }
    return buffer
}

// MARK: - 16 bits (MLXToolKit contract 1.52.0, AB-T-0230)

/// The 16-bit pixel format: RGBA, 16 bits per component, little-endian — `.rawRGBA16`'s layout, and the shared tile
/// driver's `MLXTileProcessor.deepPixelFormat` in the upscale siblings.
public let deepPixelFormat: OSType = kCVPixelFormatType_64RGBALE

/// Whether `buffer` is a 16-bit (`deepPixelFormat`) buffer.
public func isDeep(_ buffer: CVPixelBuffer) -> Bool {
    CVPixelBufferGetPixelFormatType(buffer) == deepPixelFormat
}

/// RGBA16 `CVPixelBuffer` (`deepPixelFormat`) → `[1, H, W, 3]` RGB float NHWC in [0,1] (each component / 65535).
@inline(never)
public func rgbNHWC(fromRGBA16 buffer: CVPixelBuffer, width: Int, height: Int) -> MLXArray? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

    var rgb = [Float](repeating: 0, count: height * width * 3)
    for y in 0 ..< height {
        let src = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt16.self)
        let drow = y * width * 3
        for x in 0 ..< width {
            rgb[drow + x * 3 + 0] = Float(UInt16(littleEndian: src[x * 4 + 0])) / 65535.0  // R
            rgb[drow + x * 3 + 1] = Float(UInt16(littleEndian: src[x * 4 + 1])) / 65535.0  // G
            rgb[drow + x * 3 + 2] = Float(UInt16(littleEndian: src[x * 4 + 2])) / 65535.0  // B
        }
    }
    return MLXArray(rgb, [1, height, width, 3])
}

/// `[1, H, W, 3]` RGB float (any range; clamped) → RGBA16 `CVPixelBuffer` (`deepPixelFormat`), each component rounded
/// to 16 bits, alpha opaque.
@inline(never)
public func pixelBufferRGBA16(fromRGBNHWC array: MLXArray, width: Int, height: Int) -> CVPixelBuffer? {
    let rgb = array.asType(.float32).asArray(Float.self)
    guard rgb.count >= width * height * 3 else { return nil }

    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: deepPixelFormat,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    guard CVPixelBufferCreate(nil, width, height, deepPixelFormat, attrs as CFDictionary, &pb) == kCVReturnSuccess,
          let buffer = pb else { return nil }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let dstBase = CVPixelBufferGetBaseAddress(buffer) else { return nil }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

    @inline(__always) func word(_ v: Float) -> UInt16 {
        UInt16(max(0, min(65535, (v * 65535).rounded()))).littleEndian
    }
    for y in 0 ..< height {
        let dst = dstBase.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt16.self)
        let srow = y * width * 3
        for x in 0 ..< width {
            dst[x * 4 + 0] = word(rgb[srow + x * 3 + 0])
            dst[x * 4 + 1] = word(rgb[srow + x * 3 + 1])
            dst[x * 4 + 2] = word(rgb[srow + x * 3 + 2])
            dst[x * 4 + 3] = UInt16.max
        }
    }
    return buffer
}

/// The signage-trained NAFNet checkpoint vendored in this package's bundle
/// (width=24, [1,1,1,1]; fp16; best val PSNR 41.515 dB — forge ADR-0010).
public var bundledSignageWeightsURL: URL? {
    Bundle.module.url(forResource: "nafnet", withExtension: "safetensors")
}
