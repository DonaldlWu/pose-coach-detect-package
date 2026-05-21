import CoreML
import Foundation
import Vision

public enum BallSpeedModelResource: String {
    case objectDetection = "yolov8n"
    case poseDetection = "yolov8s-pose"
}

enum BallSpeedModelResourceResolver {
    static func resolve(
        resourceName: String,
        fileExists: (String) -> Bool
    ) -> String? {
        let compiled = "\(resourceName).mlmodelc"
        if fileExists(compiled) { return compiled }

        let sourcePackage = "\(resourceName).mlpackage"
        if fileExists(sourcePackage) { return sourcePackage }

        return nil
    }
}

public enum BallSpeedModelProvider {
    public static func coreMLModel(for resource: BallSpeedModelResource) throws -> MLModel {
        guard let url = modelURL(for: resource) else {
            throw BallSpeedKitError.modelNotFound
        }
        return try MLModel(contentsOf: url)
    }

    public static func visionModel(for resource: BallSpeedModelResource) throws -> VNCoreMLModel {
        try VNCoreMLModel(for: coreMLModel(for: resource))
    }

    public static func objectDetectionVisionModel() throws -> VNCoreMLModel {
        try visionModel(for: .objectDetection)
    }

    public static func poseDetectionVisionModel() throws -> VNCoreMLModel {
        try visionModel(for: .poseDetection)
    }

    static func modelURL(for resource: BallSpeedModelResource) -> URL? {
        guard let filename = BallSpeedModelResourceResolver.resolve(
            resourceName: resource.rawValue,
            fileExists: { filename in
                let parts = filename.split(separator: ".", maxSplits: 1).map(String.init)
                guard parts.count == 2 else { return false }
                return Bundle.module.url(forResource: parts[0], withExtension: parts[1]) != nil
            }
        ) else {
            return nil
        }

        let parts = filename.split(separator: ".", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { return nil }
        return Bundle.module.url(forResource: parts[0], withExtension: parts[1])
    }
}
