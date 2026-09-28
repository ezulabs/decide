// decide-ane: the Core ML / Neural Engine shim behind a C ABI.
//
// One `Session` holds one compiled Core ML model per layer (the layer's ANE
// MLP slice: fixed shape [chunk_rows, in_dim] fp16 in, [chunk_rows, out_dim]
// fp16 out), MLMultiArrays over the engine's own shared buffers (no copies on
// this side; Core ML stages into its IOSurfaces internally), and the
// MTLSharedEvent the engine hands over. `ane_enqueue` registers one job: when
// the event reaches `wait_value` (the GPU signals it after writing the normed
// rows), the job runs the layer's predicts for `chunks` chunk slots on a
// serial queue and then sets the event to `signal_value`, which the GPU's
// next command buffer waits for. Jobs of one request fire in order because
// their wait values increase. A failed predict still signals (so the GPU
// never blocks forever) and leaves the error for `ane_error`.
import Foundation
import CoreML
import Metal
import QuartzCore

final class Job {
    let layer: Int, chunks: Int, waitValue: UInt64, signalValue: UInt64
    init(layer: Int, chunks: Int, waitValue: UInt64, signalValue: UInt64) {
        self.layer = layer; self.chunks = chunks; self.waitValue = waitValue; self.signalValue = signalValue
    }
}

final class Session {
    let chunkRows: Int, inDim: Int, outDim: Int
    var models: [MLModel] = []
    var loadMs: [Double] = []
    var event: MTLSharedEvent? = nil
    var listener: MTLSharedEventListener? = nil
    let queue = DispatchQueue(label: "decide.ane", qos: .userInteractive)
    var inputs: [MLDictionaryFeatureProvider] = []
    var options: [MLPredictionOptions] = []
    var outArrays: [MLMultiArray] = []
    /// Let Core ML write its result straight into the engine's buffer
    /// (`outputBackings`), the default. `DECIDE_ANE_BACKING=0` falls back to a
    /// CPU copy of the result; both are correct (measured identical), the copy
    /// costs a memcpy per predict.
    let useBacking = ProcessInfo.processInfo.environment["DECIDE_ANE_BACKING"] != "0"
    var error: String? = nil
    var predicts = 0
    var predictSeconds = 0.0
    var pending = 0
    let lock = NSLock()
    let idle = NSCondition()

    init(chunkRows: Int, inDim: Int, outDim: Int) {
        self.chunkRows = chunkRows; self.inDim = inDim; self.outDim = outDim
    }

    func load(dir: String, layers: Int) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        for l in 0..<layers {
            let url = URL(fileURLWithPath: dir).appendingPathComponent("L\(l).mlmodelc")
            let t0 = CACurrentMediaTime()
            let m = try MLModel(contentsOf: url, configuration: cfg)
            loadMs.append((CACurrentMediaTime() - t0) * 1e3)
            models.append(m)
        }
    }

    /// Arrays over the engine's buffers: slot s reads rows [s·chunkRows, (s+1)·chunkRows) of `h` and
    /// writes the same rows of `out`; both row-major fp16 with the given row strides (elements).
    func bind(event: MTLSharedEvent, h: UnsafeMutableRawPointer, hStride: Int, out: UnsafeMutableRawPointer, outStride: Int, slots: Int) throws {
        self.event = event
        listener = MTLSharedEventListener(dispatchQueue: queue)
        inputs = []; options = []; outArrays = []
        for s in 0..<slots {
            let hp = h.advanced(by: s * chunkRows * hStride * 2)
            let op = out.advanced(by: s * chunkRows * outStride * 2)
            let ia = try MLMultiArray(dataPointer: hp, shape: [NSNumber(value: chunkRows), NSNumber(value: inDim)], dataType: .float16,
                                      strides: [NSNumber(value: hStride), 1], deallocator: nil)
            let oa = try MLMultiArray(dataPointer: op, shape: [NSNumber(value: chunkRows), NSNumber(value: outDim)], dataType: .float16,
                                      strides: [NSNumber(value: outStride), 1], deallocator: nil)
            inputs.append(try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: ia)]))
            let o = MLPredictionOptions()
            if useBacking { o.outputBackings = ["y": oa] }
            options.append(o); outArrays.append(oa)
        }
    }

    /// The first predict of every model, so the ANE programs are resident before the first request.
    /// A model that cannot run is an error here, not a surprise on the first request.
    func warmUp() throws {
        guard !inputs.isEmpty else { throw NSError(domain: "decide.ane", code: 1, userInfo: [NSLocalizedDescriptionKey: "warm-up before bind"]) }
        for (l, m) in models.enumerated() {
            do {
                _ = try m.prediction(from: inputs[0], options: options[0])
            } catch {
                throw NSError(domain: "decide.ane", code: 2, userInfo: [NSLocalizedDescriptionKey: "warm-up of layer \(l): \(error)"])
            }
        }
    }

    /// fp16 rows of `src` into `dst`, honouring both row strides.
    func copyOut(_ src: MLMultiArray, into dst: MLMultiArray) {
        let n = outDim
        let sp = src.dataPointer.assumingMemoryBound(to: UInt16.self)
        let dp = dst.dataPointer.assumingMemoryBound(to: UInt16.self)
        let ss = src.strides[0].intValue, ds = dst.strides[0].intValue
        if ss == n && ds == n {
            dp.update(from: sp, count: chunkRows * n)
        } else {
            for r in 0..<chunkRows { dp.advanced(by: r * ds).update(from: sp.advanced(by: r * ss), count: n) }
        }
    }

    func enqueue(_ job: Job) -> Bool {
        guard let event, let listener else { return false }
        idle.lock(); pending += 1; idle.unlock()
        event.notify(listener, atValue: job.waitValue) { [self] ev, _ in
            let t0 = CACurrentMediaTime()
            for s in 0..<job.chunks {
                do {
                    let out = try models[job.layer].prediction(from: inputs[s], options: options[s])
                    // With outputBackings honoured the result already sits in the engine's buffer;
                    // otherwise (or with DECIDE_ANE_BACKING=0) copy it there.
                    if let arr = out.featureValue(for: "y")?.multiArrayValue, arr.dataPointer != outArrays[s].dataPointer {
                        copyOut(arr, into: outArrays[s])
                    }
                } catch {
                    // The first error is the one that explains the rest; keep it.
                    lock.lock(); if self.error == nil { self.error = "layer \(job.layer) chunk \(s): \(error)" }; lock.unlock()
                }
            }
            lock.lock(); predicts += job.chunks; predictSeconds += CACurrentMediaTime() - t0; lock.unlock()
            ev.signaledValue = job.signalValue
            idle.lock(); pending -= 1; if pending == 0 { idle.broadcast() }; idle.unlock()
        }
        return true
    }

    /// Wait until every enqueued job has signalled (or `timeoutS` passes; returns false then).
    func waitIdle(timeoutS: Double) -> Bool {
        idle.lock(); defer { idle.unlock() }
        let deadline = Date(timeIntervalSinceNow: timeoutS)
        while pending > 0 {
            if !idle.wait(until: deadline) { return pending == 0 }
        }
        return true
    }
}

private func setErr(_ err: UnsafeMutablePointer<CChar>?, _ len: Int, _ msg: String) {
    guard let err, len > 0 else { return }
    let bytes = Array(msg.utf8.prefix(len - 1)) + [0]
    bytes.withUnsafeBufferPointer { p in err.update(from: UnsafeRawPointer(p.baseAddress!).assumingMemoryBound(to: CChar.self), count: bytes.count) }
}

/// Load `L0.mlmodelc … L{layers-1}.mlmodelc` from `dir`. Returns a session, or NULL with the error in `err`.
@_cdecl("ane_open")
public func ane_open(_ dir: UnsafePointer<CChar>, _ layers: UInt32, _ chunkRows: UInt32, _ inDim: UInt32, _ outDim: UInt32,
                     _ err: UnsafeMutablePointer<CChar>?, _ errLen: Int) -> UnsafeMutableRawPointer? {
    let s = Session(chunkRows: Int(chunkRows), inDim: Int(inDim), outDim: Int(outDim))
    do {
        try s.load(dir: String(cString: dir), layers: Int(layers))
    } catch {
        setErr(err, errLen, "\(error)")
        return nil
    }
    return Unmanaged.passRetained(s).toOpaque()
}

@_cdecl("ane_close")
public func ane_close(_ handle: UnsafeMutableRawPointer) {
    Unmanaged<Session>.fromOpaque(handle).release()
}

/// Milliseconds `MLModel(contentsOf:)` took for layer `layer`.
@_cdecl("ane_load_ms")
public func ane_load_ms(_ handle: UnsafeMutableRawPointer, _ layer: UInt32) -> Double {
    let s = Unmanaged<Session>.fromOpaque(handle).takeUnretainedValue()
    return Int(layer) < s.loadMs.count ? s.loadMs[Int(layer)] : -1
}

/// `event` is an `id<MTLSharedEvent>` (retained by the session). See `Session.bind`.
@_cdecl("ane_bind")
public func ane_bind(_ handle: UnsafeMutableRawPointer, _ event: UnsafeMutableRawPointer, _ h: UnsafeMutableRawPointer, _ hStride: Int,
                     _ out: UnsafeMutableRawPointer, _ outStride: Int, _ slots: UInt32, _ err: UnsafeMutablePointer<CChar>?, _ errLen: Int) -> Int32 {
    let s = Unmanaged<Session>.fromOpaque(handle).takeUnretainedValue()
    let ev = Unmanaged<AnyObject>.fromOpaque(event).takeUnretainedValue()
    guard let ev = ev as? MTLSharedEvent else { setErr(err, errLen, "not an MTLSharedEvent"); return 1 }
    do {
        try s.bind(event: ev, h: h, hStride: hStride, out: out, outStride: outStride, slots: Int(slots))
    } catch {
        setErr(err, errLen, "\(error)"); return 1
    }
    return 0
}

/// Run every model once on slot 0; 0, or 1 with the error in `err`.
@_cdecl("ane_warm_up")
public func ane_warm_up(_ handle: UnsafeMutableRawPointer, _ err: UnsafeMutablePointer<CChar>?, _ errLen: Int) -> Int32 {
    do {
        try Unmanaged<Session>.fromOpaque(handle).takeUnretainedValue().warmUp()
    } catch {
        setErr(err, errLen, "\(error.localizedDescription)"); return 1
    }
    return 0
}

/// Register one job (see the file comment). Returns 0, or 1 if the session is not bound.
@_cdecl("ane_enqueue")
public func ane_enqueue(_ handle: UnsafeMutableRawPointer, _ layer: UInt32, _ chunks: UInt32, _ waitValue: UInt64, _ signalValue: UInt64) -> Int32 {
    let s = Unmanaged<Session>.fromOpaque(handle).takeUnretainedValue()
    guard Int(layer) < s.models.count, Int(chunks) <= s.inputs.count, chunks > 0 else { return 2 }
    return s.enqueue(Job(layer: Int(layer), chunks: Int(chunks), waitValue: waitValue, signalValue: signalValue)) ? 0 : 1
}

/// Wait until every enqueued job has run; 0, or 1 on timeout.
@_cdecl("ane_wait_idle")
public func ane_wait_idle(_ handle: UnsafeMutableRawPointer, _ timeoutS: Double) -> Int32 {
    Unmanaged<Session>.fromOpaque(handle).takeUnretainedValue().waitIdle(timeoutS: timeoutS) ? 0 : 1
}

/// The first error since the last call (cleared), as a C string; 0 if none.
@_cdecl("ane_error")
public func ane_error(_ handle: UnsafeMutableRawPointer, _ err: UnsafeMutablePointer<CChar>?, _ errLen: Int) -> Int32 {
    let s = Unmanaged<Session>.fromOpaque(handle).takeUnretainedValue()
    s.lock.lock(); defer { s.lock.unlock() }
    guard let e = s.error else { return 0 }
    setErr(err, errLen, e); s.error = nil
    return 1
}

/// Predicts run so far and the seconds they took (both cumulative).
@_cdecl("ane_stats")
public func ane_stats(_ handle: UnsafeMutableRawPointer, _ predicts: UnsafeMutablePointer<UInt64>, _ seconds: UnsafeMutablePointer<Double>) {
    let s = Unmanaged<Session>.fromOpaque(handle).takeUnretainedValue()
    s.lock.lock(); predicts.pointee = UInt64(s.predicts); seconds.pointee = s.predictSeconds; s.lock.unlock()
}
