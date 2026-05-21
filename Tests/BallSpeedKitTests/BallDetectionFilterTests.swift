import XCTest
@testable import BallSpeedKit

final class BallDetectionFilterTests: XCTestCase {
    func test_acceptsOnlyExactSportsBallLabels() {
        XCTAssertTrue(BallDetectionFilter.isSportsBallLabel("sports ball"))
        XCTAssertTrue(BallDetectionFilter.isSportsBallLabel("Sports Ball"))
        XCTAssertTrue(BallDetectionFilter.isSportsBallLabel(" ball "))

        XCTAssertFalse(BallDetectionFilter.isSportsBallLabel("baseball bat"))
        XCTAssertFalse(BallDetectionFilter.isSportsBallLabel("bat"))
        XCTAssertFalse(BallDetectionFilter.isSportsBallLabel("balloon"))
    }

    func test_requiresRaisedConfidenceThreshold() {
        XCTAssertFalse(BallDetectionFilter.acceptsSportsBall(label: "sports ball", confidence: 0.44))
        XCTAssertTrue(BallDetectionFilter.acceptsSportsBall(label: "sports ball", confidence: 0.45))
    }
}
