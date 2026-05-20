# BallSpeedKit

A Swift Package for detecting and tracking a sports ball in video, producing an annotated output video with bounding box and trajectory trail.

## Requirements

| Platform | Minimum Version |
|----------|----------------|
| iOS      | 16.0+          |
| macOS    | 13.0+          |

Swift 5.9 / Xcode 15+

## Installation

### Swift Package Manager

In Xcode: **File → Add Package Dependencies**, enter the local path or repository URL.

In `Package.swift`:

```swift
dependencies: [
    .package(path: "../ballspeed/BallSpeedKit")
]
```

## Quick Start

```swift
import BallSpeedKit

let kit = BallSpeedKit()

do {
    try await kit.process(
        inputURL: URL(fileURLWithPath: "/path/to/input.mov"),
        outputURL: URL(fileURLWithPath: "/path/to/output.mp4")
    )
} catch {
    print("Failed: \(error.localizedDescription)")
}
```

## API Reference

### `BallSpeedKit`

The main entry point. Stateless; safe to create multiple instances.

```swift
public struct BallSpeedKit {
    public init()
    public func process(inputURL: URL, outputURL: URL) async throws
}
```

#### `process(inputURL:outputURL:)`

Reads the input video frame by frame, detects a sports ball using a bundled YOLOv8n CoreML model, draws overlays, and writes the result to `outputURL`.

| Parameter  | Type  | Description                                 |
|------------|-------|---------------------------------------------|
| `inputURL` | `URL` | Source video file (MOV, MP4, or any format supported by AVFoundation) |
| `outputURL`| `URL` | Destination path for the annotated MP4. Existing file is overwritten. |

- **Throws**: `BallSpeedKitError`
- **Concurrency**: `async`, must be called from an async context.

---

### `BallSpeedKitError`

```swift
public enum BallSpeedKitError: Error, LocalizedError {
    case modelNotFound  // CoreML model missing from bundle
    case noVideoTrack   // Input file has no video track
    case writeFailed    // AVAssetWriter failed to finalize output
}
```

---

## Output Video

The output MP4 contains the original video with the following overlays:

| Overlay | Color | Description |
|---------|-------|-------------|
| Bounding box | Green | Detected ball region per frame |
| Trail line | Red | Accumulated path of detected centers |
| Trail dots | Yellow | Individual detection points |

**Trail reset**: if a new detection is more than ~15% of the frame width away from the previous point, the trail resets. This separates distinct throws or scenes.

---

## Internals

| Component | File | Responsibility |
|-----------|------|----------------|
| `BallDetector` | `BallDetector.swift` | Runs `VNCoreMLRequest` on each `CVPixelBuffer`; returns the highest-confidence `sports ball` detection |
| `VideoProcessor` | `VideoProcessor.swift` | `AVAssetReader` → detect → draw → `AVAssetWriter` pipeline |
| Model | `Resources/yolov8n.mlpackage` | YOLOv8n exported to CoreML with NMS, input 640×640, COCO classes |

### Detection Model

- **Model**: YOLOv8n (nano), COCO-pretrained
- **Class**: `sports ball` (COCO class 32), covers baseball, tennis ball, soccer ball, etc.
- **Input**: 640×640 image, scaled to fill
- **Confidence threshold**: detections below the Vision framework default (~0.3) are discarded
- **Limitation**: small balls (< ~20px on a 1080p frame) may not be reliably detected by the nano model; consider `yolov8m.mlpackage` for better accuracy

### Coordinate System

`BallDetector` converts Vision's bottom-left origin coordinates to top-left origin (standard UIKit/AppKit) before returning. All `CGPoint` and `CGRect` values in `Detection` use top-left origin with values normalized to `0...1`.

---

## Replacing the Model

To use a larger or custom-trained model:

1. Export to CoreML:
   ```bash
   python3 -c "from ultralytics import YOLO; YOLO('yolov8m.pt').export(format='coreml', nms=True, imgsz=640)"
   ```
2. Replace `Sources/BallSpeedKit/Resources/yolov8n.mlpackage` with the new `.mlpackage`.
3. Update the resource name in `BallDetector.swift`:
   ```swift
   Bundle.module.url(forResource: "yolov8m", withExtension: "mlpackage")
   ```
