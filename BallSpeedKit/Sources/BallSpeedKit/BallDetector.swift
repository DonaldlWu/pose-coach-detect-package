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
        guard let modelURL = Bundle.module.url(forResource: "yolov8n", withExtension: "mlpackage") else {
            throw BallSpeedKitError.modelNotFound
        }
        mlModel = try MLModel(contentsOf: modelURL)
    }

    /// Returns the highest-confidence sports ball detection in the frame, if any.
    func detect(in pixelBuffer: CVPixelBuffer) throws -> Detection? {
        let vnModel = try VNCoreMLModel(for: mlModel)
        let request = VNCoreMLRequest(model: vnModel)
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
            let shape = arr.shape.map { $0.intValue }
            if shape.count == 3 && shape[2] == 4  { coordArray = arr }
            if shape.count == 3 && shape[2] == 80 { confArray  = arr }
        }

        guard let coords = coordArray, let confs = confArray else { return nil }

        let n = coords.shape[1].intValue
        var bestConf: Float = confidenceThreshold
        var bestIdx = -1

        for i in 0..<n {
            let conf = confs[[0, i, sportsBallClassIndex] as [NSNumber]].floatValue
            if conf > bestConf { bestConf = conf; bestIdx = i }
        }

        guard bestIdx >= 0 else { return nil }

        let cx = coords[[0, bestIdx, 0] as [NSNumber]].floatValue
        let cy = coords[[0, bestIdx, 1] as [NSNumber]].floatValue
        let w  = coords[[0, bestIdx, 2] as [NSNumber]].floatValue
        let h  = coords[[0, bestIdx, 3] as [NSNumber]].floatValue

        let box = CGRect(x: CGFloat(cx - w / 2), y: CGFloat(cy - h / 2),
                         width: CGFloat(w), height: CGFloat(h))
        return Detection(center: CGPoint(x: CGFloat(cx), y: CGFloat(cy)),
                         boundingBox: box,
                         confidence: bestConf)
    }
}
