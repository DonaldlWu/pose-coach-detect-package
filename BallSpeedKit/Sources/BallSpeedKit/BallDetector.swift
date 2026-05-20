import CoreML
import Vision
import CoreGraphics

struct Detection {
    let center: CGPoint
    let boundingBox: CGRect  // normalized, top-left origin (0...1)
    let confidence: Float
}

final class BallDetector {
    private let mlModel: MLModel

    init() throws {
        // SPM .process() compiles .mlpackage → .mlmodelc at build time;
        // fall back to .mlpackage for local/preview builds.
        let modelURL = Bundle.module.url(forResource: "yolov8n", withExtension: "mlmodelc")
            ?? Bundle.module.url(forResource: "yolov8n", withExtension: "mlpackage")
        guard let url = modelURL else {
            throw BallSpeedKitError.modelNotFound
        }
        mlModel = try MLModel(contentsOf: url)
    }

    /// Returns the highest-confidence sports ball detection in the frame, if any.
    /// Coordinates are normalized (0…1) with top-left origin in the pixel buffer's own space.
    func detect(in pixelBuffer: CVPixelBuffer) throws -> Detection? {
        let vnModel = try VNCoreMLModel(for: mlModel)
        let request = VNCoreMLRequest(model: vnModel)
        // scaleFill keeps a 1:1 coordinate mapping (no letterbox offset to undo).
        // Detect in the buffer's native space; callers handle any display transform.
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try handler.perform([request])

        // Path A: Vision parsed the model output as object detections
        if let observations = request.results as? [VNRecognizedObjectObservation],
           !observations.isEmpty {
            return bestBall(from: observations)
        }

        // Path B: YOLOv8 NMS pipeline returns raw tensors (VNCoreMLFeatureValueObservation)
        if let features = request.results as? [VNCoreMLFeatureValueObservation] {
            return bestBall(fromFeatures: features)
        }

        return nil
    }

    // MARK: - Path A

    private func bestBall(from observations: [VNRecognizedObjectObservation]) -> Detection? {
        let candidates = observations.compactMap { obs -> (VNRecognizedObjectObservation, Float)? in
            guard let top = obs.labels.first,
                  top.identifier.lowercased().contains("ball") else { return nil }
            return (obs, top.confidence)
        }
        guard let (obs, conf) = candidates.max(by: { $0.1 < $1.1 }) else { return nil }

        // Vision uses bottom-left origin; flip to top-left
        let box = obs.boundingBox
        let flipped = CGRect(x: box.minX, y: 1 - box.maxY, width: box.width, height: box.height)
        return Detection(center: CGPoint(x: flipped.midX, y: flipped.midY),
                         boundingBox: flipped,
                         confidence: conf)
    }

    // MARK: - Path B
    // YOLOv8 NMS pipeline outputs two MLMultiArrays:
    //   coordinates – shape [1, N, 4]  (cx, cy, w, h) normalised to 0...1
    //   confidence  – shape [1, N, 80] (per-class score, COCO 80 classes)
    // Sports ball is COCO class index 32.

    private let sportsBallClassIndex = 32
    private let confidenceThreshold: Float = 0.25

    private func bestBall(fromFeatures features: [VNCoreMLFeatureValueObservation]) -> Detection? {
        var coordArray: MLMultiArray?
        var confArray: MLMultiArray?

        for feat in features {
            guard let arr = feat.featureValue.multiArrayValue else { continue }
            switch feat.featureName {
            case "coordinates": coordArray = arr
            case "confidence":  confArray  = arr
            default:
                // Fallback: match by last-dimension size
                let shape = arr.shape.map { $0.intValue }
                if shape.last == 4  { coordArray = arr }
                if shape.last == 80 { confArray  = arr }
            }
        }

        guard let coords = coordArray, let confs = confArray else { return nil }

        // Model outputs are 2-D: [N, 4] and [N, 80]
        let n = confs.shape[0].intValue
        var bestConf: Float = confidenceThreshold
        var bestIdx = -1

        for i in 0..<n {
            let conf = confs[[i, sportsBallClassIndex] as [NSNumber]].floatValue
            if conf > bestConf { bestConf = conf; bestIdx = i }
        }

        guard bestIdx >= 0 else { return nil }

        let cx = coords[[bestIdx, 0] as [NSNumber]].floatValue
        let cy = coords[[bestIdx, 1] as [NSNumber]].floatValue
        let w  = coords[[bestIdx, 2] as [NSNumber]].floatValue
        let h  = coords[[bestIdx, 3] as [NSNumber]].floatValue

        let box = CGRect(x: CGFloat(cx - w / 2), y: CGFloat(cy - h / 2),
                         width: CGFloat(w), height: CGFloat(h))
        return Detection(center: CGPoint(x: CGFloat(cx), y: CGFloat(cy)),
                         boundingBox: box,
                         confidence: bestConf)
    }
}
