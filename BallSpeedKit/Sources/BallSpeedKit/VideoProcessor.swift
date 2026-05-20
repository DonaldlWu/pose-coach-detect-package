import AVFoundation
import CoreGraphics
import CoreImage
import CoreVideo

final class VideoProcessor {
    private let detector: BallDetector
    private let maxTrailJump: CGFloat = 0.15  // normalized units (~300px on 1920px frame)

    init(detector: BallDetector) {
        self.detector = detector
    }

    func process(inputURL: URL, outputURL: URL) async throws {
        // --- Reader setup ---
        let asset = AVURLAsset(url: inputURL)
        let duration = try await asset.load(.duration)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let videoTrack = tracks.first else {
            throw BallSpeedKitError.noVideoTrack
        }

        let naturalSize = try await videoTrack.load(.naturalSize)
        let transform   = try await videoTrack.load(.preferredTransform)
        let nominalFPS  = try await videoTrack.load(.nominalFrameRate)

        let reader = try AVAssetReader(asset: asset)
        let readerOutput = AVAssetReaderTrackOutput(
            track: videoTrack,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            ]
        )
        reader.add(readerOutput)
        reader.startReading()

        // --- Writer setup ---
        try? FileManager.default.removeItem(at: outputURL)
        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)

        // Write at naturalSize; transform is applied as display metadata only.
        // Frames from AVAssetReader are always in naturalSize dimensions.
        let frameWidth  = abs(naturalSize.width)
        let frameHeight = abs(naturalSize.height)

        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: frameWidth,
            AVVideoHeightKey: frameHeight,
        ]
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        writerInput.expectsMediaDataInRealTime = false
        writerInput.transform = transform

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: frameWidth,
                kCVPixelBufferHeightKey as String: frameHeight,
            ]
        )
        writer.add(writerInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        // --- Frame-by-frame processing ---
        let detectionOrientation = cgOrientation(from: transform)
        var trail: [CGPoint] = []
        var frameIndex = 0
        let frameDuration = CMTime(value: 1, timescale: CMTimeScale(nominalFPS))
        let width  = frameWidth
        let height = frameHeight

        while reader.status == .reading {
            guard let sampleBuffer = readerOutput.copyNextSampleBuffer(),
                  let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { break }

            // Run ball detection (pass orientation so Vision upright-corrects the frame)
            let detection = try? detector.detect(in: pixelBuffer, orientation: detectionOrientation)

            if let det = detection {
                // CGContext uses bottom-left origin; YOLO uses top-left → flip y
                let pixelPt = CGPoint(x: det.center.x * width, y: (1.0 - det.center.y) * height)
                if let last = trail.last,
                   hypot(pixelPt.x - last.x, pixelPt.y - last.y) > maxTrailJump * width {
                    trail.removeAll()
                }
                trail.append(pixelPt)
            }

            // Draw overlays onto a new pixel buffer
            let outputBuffer = draw(
                on: pixelBuffer,
                detection: detection,
                trail: trail,
                size: CGSize(width: width, height: height)
            )

            let presentationTime = CMTime(value: CMTimeValue(frameIndex), timescale: frameDuration.timescale)
            while !writerInput.isReadyForMoreMediaData { await Task.yield() }
            if let buf = outputBuffer {
                adaptor.append(buf, withPresentationTime: presentationTime)
            }
            frameIndex += 1
        }

        writerInput.markAsFinished()
        await writer.finishWriting()

        if writer.status == .failed {
            throw writer.error ?? BallSpeedKitError.writeFailed
        }
    }

    // MARK: - Drawing

    private func draw(
        on pixelBuffer: CVPixelBuffer,
        detection: Detection?,
        trail: [CGPoint],
        size: CGSize
    ) -> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = Int(size.width)
        let height = Int(size.height)
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        guard let ctx = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }

        // Draw original frame
        if let ciImage = CIImage(cvPixelBuffer: pixelBuffer).cgImage(using: CIContext()) {
            ctx.draw(ciImage, in: CGRect(origin: .zero, size: size))
        }

        // Draw trail lines
        if trail.count >= 2 {
            ctx.setStrokeColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
            ctx.setLineWidth(3)
            ctx.setLineCap(.round)
            ctx.setLineJoin(.round)
            ctx.beginPath()
            ctx.move(to: trail[0])
            for pt in trail.dropFirst() { ctx.addLine(to: pt) }
            ctx.strokePath()
        }

        // Draw trail dots
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 0, alpha: 1))
        for pt in trail {
            ctx.fillEllipse(in: CGRect(x: pt.x - 4, y: pt.y - 4, width: 8, height: 8))
        }

        // Draw detection bounding box
        if let det = detection {
            // Flip y: YOLO top-left origin → CGContext bottom-left origin
            let box = CGRect(
                x: det.boundingBox.minX * size.width,
                y: (1.0 - det.boundingBox.maxY) * size.height,
                width: det.boundingBox.width * size.width,
                height: det.boundingBox.height * size.height
            )
            ctx.setStrokeColor(CGColor(red: 0, green: 1, blue: 0, alpha: 1))
            ctx.setLineWidth(2)
            ctx.stroke(box)
        }

        guard let cgImage = ctx.makeImage() else { return nil }

        // Wrap back into a CVPixelBuffer
        var outBuffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, nil, &outBuffer)
        guard let out = outBuffer else { return nil }
        CVPixelBufferLockBaseAddress(out, [])
        defer { CVPixelBufferUnlockBaseAddress(out, []) }
        let outCtx = CGContext(
            data: CVPixelBufferGetBaseAddress(out),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(out),
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )
        outCtx?.draw(cgImage, in: CGRect(origin: .zero, size: size))
        return out
    }
}

/// Convert a video track's preferredTransform to a CGImagePropertyOrientation
/// so that VNImageRequestHandler can upright-correct the raw pixel buffer.
private func cgOrientation(from transform: CGAffineTransform) -> CGImagePropertyOrientation {
    switch (transform.a, transform.b, transform.c, transform.d) {
    case (0,  1, -1, 0): return .right   // 90° CW  (portrait, natural=landscape)
    case (0, -1,  1, 0): return .left    // 90° CCW
    case (-1, 0,  0, -1): return .down   // 180°
    default:             return .up      // identity / already upright
    }
}

private extension CGSize {
    var standardized: CGSize {
        CGSize(width: abs(width), height: abs(height))
    }
}

private extension CIImage {
    func cgImage(using context: CIContext) -> CGImage? {
        context.createCGImage(self, from: extent)
    }
}
