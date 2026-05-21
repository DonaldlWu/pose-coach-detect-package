# BallSpeedKit

A Swift package for detecting and tracking a sports ball in video, producing an annotated output video with a bounding box and trajectory trail.

## Requirements

| Platform | Minimum Version |
|----------|----------------|
| iOS      | 16.0+          |
| macOS    | 13.0+          |

Swift 5.9 / Xcode 15+

## Installation

### Xcode

Use **File > Add Package Dependencies...** and enter:

```text
https://github.com/DonaldlWu/pose-coach-detect-package.git
```

Choose **Up to Next Major Version** from `0.1.1`.

### Package.swift

In `Package.swift`:

```swift
dependencies: [
    .package(
        url: "https://github.com/DonaldlWu/pose-coach-detect-package.git",
        from: "0.1.1"
    )
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
    public func process(
        inputURL: URL,
        outputURL: URL,
        onProgress: ((Double) -> Void)? = nil
    ) async throws
}
```

#### `process(inputURL:outputURL:)`

Reads the input video frame by frame, detects a sports ball using the bundled YOLOv8n CoreML model, draws overlays, and writes the result to `outputURL`.

| Parameter  | Type  | Description                                 |
|------------|-------|---------------------------------------------|
| `inputURL` | `URL` | Source video file (MOV, MP4, or any format supported by AVFoundation) |
| `outputURL`| `URL` | Destination path for the annotated MP4. Existing file is overwritten. |
| `onProgress` | `((Double) -> Void)?` | Optional callback reporting processing progress from `0.0` to `1.0` |

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
| `BallSpeedModelProvider` | `BallSpeedModelProvider.swift` | Resolves bundled CoreML resources from `Bundle.module` |
| `VideoProcessor` | `VideoProcessor.swift` | `AVAssetReader` → detect → draw → `AVAssetWriter` pipeline |
| Object model | `Resources/yolov8n.mlpackage` | YOLOv8n exported to CoreML with NMS, input 640x640, COCO classes |
| Pose model | `Resources/yolov8s-pose.mlpackage` | Bundled pose model for downstream pose-analysis consumers |

### Detection Model

- **Model**: YOLOv8n (nano), COCO-pretrained
- **Class**: `sports ball` (COCO class 32), covers baseball, tennis ball, soccer ball, etc.
- **Input**: 640x640 image, scaled to fill
- **Confidence threshold**: detections below `0.45` are discarded
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

## Python Prototype Tool

`Tools/ballspeed.py` is a local prototype script for quick YOLO validation outside the Swift package. It is not part of the SwiftPM product.

### Setup

Install the Python dependencies in your local environment:

```bash
pip install ultralytics opencv-python numpy tqdm
```

Place the source YOLO weight file at the repo root:

```text
yolov8n.pt
```

Place input videos under:

```text
input/
```

`input/`, `output/`, and `*.pt` are intentionally ignored by git.

### Usage

From the repo root:

```bash
python3 Tools/ballspeed.py hitting_video.mp4
```

The script reads:

```text
input/hitting_video.mp4
```

and writes:

```text
output/hitting_video_output.mp4
```

The script resolves paths relative to the repository root, so it can also be launched from another working directory.

## Known Limitations

- Output is written through `AVAssetWriter` as MP4 and may not preserve Photos slow-motion time mapping or edited playback metadata.
- The bundled YOLOv8n object model is optimized for package size and speed, not maximum ball-detection recall.
- The detector accepts exact `sports ball` / `ball` labels only, to avoid false positives such as `baseball bat`.
