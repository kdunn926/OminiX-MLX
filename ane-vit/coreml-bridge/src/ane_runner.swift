// ane_runner.swift — Swift glue exposing a C ABI for Core ML predict on ANE.
//
// Compiled by build.rs (`swiftc -emit-object`) and linked into the Rust
// crate. Functions are declared @_cdecl so they appear under their plain
// C names in the symbol table.

import Accelerate
import CoreML
import Foundation

// Compute-unit selector codes — match build.rs documentation.
private func computeUnits(from code: Int32) -> MLComputeUnits {
    switch code {
    case 0: return .all
    case 1: return .cpuOnly
    case 2: return .cpuAndGPU
    case 3: return .cpuAndNeuralEngine
    // No `.neuralEngineOnly` exists; "neural-engine-preferred" maps to
    // `.cpuAndNeuralEngine` (matches the Rust enum docs and build.rs).
    case 4: return .cpuAndNeuralEngine
    default:
        return .all
    }
}

// Resolve a (possibly uncompiled) model URL to a compiled .mlmodelc.
//
// `MLModel.compileModel` writes a fresh .mlmodelc into a temp dir on every
// call, and Apple's docs make the caller responsible for moving/cleaning
// it. The old code recompiled on every `coreml_load` and leaked the temp
// dirs. Compiled models are now cached under
// ~/Library/Caches/ominix-coreml-bridge keyed by package name + mtime.
private func compiledModelURL(for url: URL) throws -> URL {
    if url.path.hasSuffix(".mlmodelc") { return url }
    let fm = FileManager.default
    let mtime = (try? fm.attributesOfItem(atPath: url.path)[.modificationDate] as? Date)
        .flatMap { $0 }?.timeIntervalSince1970 ?? 0
    let cacheRoot = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("ominix-coreml-bridge", isDirectory: true)
    try fm.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
    let cached = cacheRoot.appendingPathComponent(
        "\(url.deletingPathExtension().lastPathComponent)-\(UInt64(mtime)).mlmodelc")
    if fm.fileExists(atPath: cached.path) { return cached }
    let tmp = try MLModel.compileModel(at: url)
    do {
        try fm.moveItem(at: tmp, to: cached)
    } catch {
        // A racing process may have populated the cache first; otherwise
        // clean up the temp dir and propagate.
        try? fm.removeItem(at: tmp)
        if !fm.fileExists(atPath: cached.path) { throw error }
    }
    return cached
}

// Expected element count of the model's input MLMultiArray (and the shape).
private func inputShape(of handle: Handle) -> [NSNumber] {
    let inputDesc = handle.model.modelDescription.inputDescriptionsByName[handle.inputName]
    return (inputDesc?.multiArrayConstraint?.shape).flatMap {
        $0.isEmpty ? nil : $0
    } ?? [1, 3, 224, 224].map { NSNumber(value: $0) }
}

// Container the Rust side holds via opaque pointer.
private final class Handle {
    let model: MLModel
    let inputName: String
    let outputName: String

    init(model: MLModel, inputName: String, outputName: String) {
        self.model = model
        self.inputName = inputName
        self.outputName = outputName
    }
}

@_cdecl("coreml_load")
public func coreml_load(_ pathPtr: UnsafePointer<CChar>?,
                        _ computeUnitsCode: Int32) -> UnsafeMutableRawPointer? {
    guard let pathPtr = pathPtr else { return nil }
    let path = String(cString: pathPtr)
    let url = URL(fileURLWithPath: path)

    do {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits(from: computeUnitsCode)
        let compiledURL = try compiledModelURL(for: url)
        let model = try MLModel(contentsOf: compiledURL, configuration: config)

        // Pick first input + output by convention.
        let inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "pixels"
        let outputName = model.modelDescription.outputDescriptionsByName.keys.first ?? "output"

        let handle = Handle(model: model, inputName: inputName, outputName: outputName)
        return Unmanaged.passRetained(handle).toOpaque()
    } catch {
        // Log to stderr; opaque pointer NULL signals failure to Rust.
        FileHandle.standardError.write(
            Data("coreml_load failed: \(error)\n".utf8)
        )
        return nil
    }
}

/// Zero-copy predict — the input Float buffer is wrapped as an MLMultiArray
/// with a no-op deallocator. The caller (Rust) MUST keep `pixels` alive
/// for the duration of the call; the prediction completes synchronously so
/// the lifetime requirement is bounded by the call.
///
/// Saves one copy per inference vs the regular `coreml_predict` (which
/// allocates a fresh MLMultiArray and copies). For 224×224×3 = 600 KB it
/// shaves ~0.2-0.4 ms per predict; for 448×448×3 = 2.4 MB it's 1-2 ms.
@_cdecl("coreml_predict_zero_copy")
public func coreml_predict_zero_copy(_ rawHandle: UnsafeMutableRawPointer?,
                                     _ pixels: UnsafeMutablePointer<Float>?,
                                     _ pixelCount: Int32,
                                     _ outBuffer: UnsafeMutablePointer<Float>?,
                                     _ outCapacity: Int32,
                                     _ outActualLen: UnsafeMutablePointer<Int32>?) -> Int32 {
    guard let rawHandle = rawHandle else { return -1 }
    guard let pixels = pixels, pixelCount > 0 else { return -2 }
    guard let outBuffer = outBuffer, outCapacity > 0 else { return -3 }
    guard let outActualLen = outActualLen else { return -4 }

    let handle = Unmanaged<Handle>.fromOpaque(rawHandle).takeUnretainedValue()
    do {
        let shape = inputShape(of: handle)
        // The MLMultiArray below is a view over exactly shape-product
        // floats of caller memory; a shorter Rust buffer would let Core ML
        // read out of bounds of Rust-owned memory.
        let expected = shape.reduce(1) { $0 * $1.intValue }
        guard Int(pixelCount) == expected else {
            FileHandle.standardError.write(Data(
                "coreml_predict_zero_copy: pixelCount \(pixelCount) != model input size \(expected)\n".utf8))
            return -10
        }
        // Row-major float32 strides.
        var strides: [NSNumber] = Array(repeating: NSNumber(value: 1), count: shape.count)
        var acc = 1
        for i in stride(from: shape.count - 1, through: 0, by: -1) {
            strides[i] = NSNumber(value: acc)
            acc *= shape[i].intValue
        }
        let arr = try MLMultiArray(
            dataPointer: UnsafeMutableRawPointer(pixels),
            shape: shape,
            dataType: .float32,
            strides: strides,
            deallocator: nil  // caller owns memory; MLMultiArray does not free
        )
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [handle.inputName: MLFeatureValue(multiArray: arr)]
        )
        let output = try handle.model.prediction(from: provider)
        guard let outFeat = output.featureValue(for: handle.outputName)?.multiArrayValue else {
            return -5
        }
        let n = outFeat.count
        if n > Int(outCapacity) {
            outActualLen.pointee = Int32(n)
            return -6
        }
        switch outFeat.dataType {
        case .float32:
            let outPtr = outFeat.dataPointer.bindMemory(to: Float.self, capacity: n)
            outBuffer.update(from: outPtr, count: n)
        case .float16:
            let srcPtr = outFeat.dataPointer.bindMemory(to: UInt16.self, capacity: n)
            var srcImg = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: srcPtr),
                height: 1, width: vImagePixelCount(n), rowBytes: n * 2
            )
            var dstImg = vImage_Buffer(
                data: UnsafeMutableRawPointer(outBuffer),
                height: 1, width: vImagePixelCount(n), rowBytes: n * 4
            )
            let err = vImageConvert_Planar16FtoPlanarF(&srcImg, &dstImg, 0)
            if err != kvImageNoError { return -8 }
        case .double:
            let src = outFeat.dataPointer.bindMemory(to: Double.self, capacity: n)
            for i in 0..<n { outBuffer[i] = Float(src[i]) }
        default:
            return -9
        }
        outActualLen.pointee = Int32(n)
        return 0
    } catch {
        FileHandle.standardError.write(Data("coreml_predict_zero_copy failed: \(error)\n".utf8))
        return -7
    }
}

@_cdecl("coreml_predict")
public func coreml_predict(_ rawHandle: UnsafeMutableRawPointer?,
                           _ pixels: UnsafePointer<Float>?,
                           _ pixelCount: Int32,
                           _ outBuffer: UnsafeMutablePointer<Float>?,
                           _ outCapacity: Int32,
                           _ outActualLen: UnsafeMutablePointer<Int32>?) -> Int32 {
    guard let rawHandle = rawHandle else { return -1 }
    guard let pixels = pixels, pixelCount > 0 else { return -2 }
    guard let outBuffer = outBuffer, outCapacity > 0 else { return -3 }
    guard let outActualLen = outActualLen else { return -4 }

    let handle = Unmanaged<Handle>.fromOpaque(rawHandle).takeUnretainedValue()

    do {
        // Build input MLMultiArray. Shape pulled from model description.
        let shape = inputShape(of: handle)
        // The fresh MLMultiArray holds exactly shape-product floats;
        // copying `pixelCount` unvalidated caller floats into it was a
        // heap buffer overflow for oversized inputs (and left the tail
        // uninitialized for undersized ones).
        let expected = shape.reduce(1) { $0 * $1.intValue }
        guard Int(pixelCount) == expected else {
            FileHandle.standardError.write(Data(
                "coreml_predict: pixelCount \(pixelCount) != model input size \(expected)\n".utf8))
            return -10
        }
        let arr = try MLMultiArray(shape: shape, dataType: .float32)
        let buf = arr.dataPointer.bindMemory(to: Float.self, capacity: expected)
        buf.update(from: pixels, count: expected)

        let provider = try MLDictionaryFeatureProvider(
            dictionary: [handle.inputName: MLFeatureValue(multiArray: arr)]
        )
        let output = try handle.model.prediction(from: provider)
        guard let outFeat = output.featureValue(for: handle.outputName)?.multiArrayValue else {
            return -5
        }

        let n = outFeat.count
        if n > Int(outCapacity) {
            outActualLen.pointee = Int32(n)
            return -6  // buffer too small; caller can re-allocate
        }
        // Convert to float32 based on the output's dtype. ANE/Core ML
        // commonly returns fp16; we widen on the Swift side so the Rust
        // FFI surface stays float32-only.
        switch outFeat.dataType {
        case .float32:
            let outPtr = outFeat.dataPointer.bindMemory(to: Float.self, capacity: n)
            outBuffer.update(from: outPtr, count: n)
        case .float16:
            // Use vDSP to widen fp16 → fp32.
            let srcPtr = outFeat.dataPointer.bindMemory(to: UInt16.self, capacity: n)
            var srcImg = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: srcPtr),
                height: 1, width: vImagePixelCount(n), rowBytes: n * 2
            )
            var dstImg = vImage_Buffer(
                data: UnsafeMutableRawPointer(outBuffer),
                height: 1, width: vImagePixelCount(n), rowBytes: n * 4
            )
            let err = vImageConvert_Planar16FtoPlanarF(&srcImg, &dstImg, 0)
            if err != kvImageNoError {
                FileHandle.standardError.write(Data("[swift] vImage fp16→fp32 failed: \(err)\n".utf8))
                return -8
            }
        case .double:
            let src = outFeat.dataPointer.bindMemory(to: Double.self, capacity: n)
            for i in 0..<n { outBuffer[i] = Float(src[i]) }
        default:
            FileHandle.standardError.write(Data("[swift] unsupported output dtype: \(outFeat.dataType.rawValue)\n".utf8))
            return -9
        }
        outActualLen.pointee = Int32(n)
        return 0
    } catch {
        FileHandle.standardError.write(
            Data("coreml_predict failed: \(error)\n".utf8)
        )
        return -7
    }
}

@_cdecl("coreml_free")
public func coreml_free(_ rawHandle: UnsafeMutableRawPointer?) {
    guard let rawHandle = rawHandle else { return }
    Unmanaged<Handle>.fromOpaque(rawHandle).release()
}
