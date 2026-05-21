import XCTest
@testable import BallSpeedKit

final class VideoProcessingProgressTests: XCTestCase {
    func test_progressUsesCompletedFrameCount() {
        XCTAssertEqual(VideoProcessingProgress.value(completedFrames: 1, totalFrames: 4), 0.25)
        XCTAssertEqual(VideoProcessingProgress.value(completedFrames: 4, totalFrames: 4), 1.0)
    }

    func test_progressClampsToOne() {
        XCTAssertEqual(VideoProcessingProgress.value(completedFrames: 5, totalFrames: 4), 1.0)
    }

    func test_progressReturnsNilWhenTotalFramesIsInvalid() {
        XCTAssertNil(VideoProcessingProgress.value(completedFrames: 1, totalFrames: 0))
    }
}
