import CoreML
import Vision
import CoreGraphics

struct Detection {
    let center: CGPoint
    let boundingBox: CGRect  // normalized (0...1)
    let confidence: Float
}

final class BallDetector {
    private let request: VNCoreMLRequest

    init() throws {
        guard let modelURL = Bundle.module.url(forResource: "yolov8n", withExtension: "mlpackage") else {
            throw BallSpeedKitError.modelNotFound
        }
        let mlModel = try MLModel(contentsOf: modelURL)
        let vnModel = try VNCoreMLModel(for: mlModel)

        request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .scaleFill
    }

    // Returns the highest-confidence sports ball detection in the frame, if any.
    func detect(in pixelBuffer: CVPixelBuffer) throws -> Detection? {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try handler.perform([request])

        guard let observations = request.results as? [VNRecognizedObjectObservation] else {
            return nil
        }

        // COCO label for sports ball is "sports ball"
        let sportsBall = observations
            .compactMap { obs -> (VNRecognizedObjectObservation, Float)? in
                guard let label = obs.labels.first,
                      label.identifier == "sports ball" else { return nil }
                return (obs, label.confidence)
            }
            .max(by: { $0.1 < $1.1 })

        guard let (obs, conf) = sportsBall else { return nil }

        let box = obs.boundingBox  // Vision coords: origin bottom-left, y flipped
        // Flip y to top-left origin
        let flipped = CGRect(
            x: box.minX,
            y: 1 - box.maxY,
            width: box.width,
            height: box.height
        )
        let center = CGPoint(x: flipped.midX, y: flipped.midY)
        return Detection(center: center, boundingBox: flipped, confidence: conf)
    }
}
