import Foundation

public enum BallSpeedKitError: Error, LocalizedError {
    case modelNotFound
    case noVideoTrack
    case writeFailed

    public var errorDescription: String? {
        switch self {
        case .modelNotFound:  return "YOLOv8 CoreML model not found in bundle."
        case .noVideoTrack:   return "Input file contains no video track."
        case .writeFailed:    return "Failed to write output video."
        }
    }
}

/// Detects and tracks a sports ball in a video, writing an annotated output video.
public struct BallSpeedKit {
    public init() {}

    /// Process a video file, overlaying ball detection bounding boxes and trail.
    /// - Parameters:
    ///   - inputURL:  URL of the source video file.
    ///   - outputURL: URL where the annotated video will be written (MP4).
    public func process(inputURL: URL, outputURL: URL) async throws {
        let detector = try BallDetector()
        let processor = VideoProcessor(detector: detector)
        try await processor.process(inputURL: inputURL, outputURL: outputURL)
    }
}
