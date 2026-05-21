import XCTest
@testable import BallSpeedKit

final class BallSpeedModelProviderTests: XCTestCase {
    func test_resolverPrefersCompiledModelOverSourcePackage() throws {
        let existing: Set<String> = ["yolov8n.mlmodelc", "yolov8n.mlpackage"]

        let resolved = BallSpeedModelResourceResolver.resolve(
            resourceName: "yolov8n",
            fileExists: { existing.contains($0) }
        )

        XCTAssertEqual(resolved, "yolov8n.mlmodelc")
    }

    func test_resolverFallsBackToSourcePackage() throws {
        let existing: Set<String> = ["yolov8n.mlpackage"]

        let resolved = BallSpeedModelResourceResolver.resolve(
            resourceName: "yolov8n",
            fileExists: { existing.contains($0) }
        )

        XCTAssertEqual(resolved, "yolov8n.mlpackage")
    }

    func test_resolverReturnsNilWhenNoSupportedResourceExists() throws {
        let resolved = BallSpeedModelResourceResolver.resolve(
            resourceName: "yolov8n",
            fileExists: { _ in false }
        )

        XCTAssertNil(resolved)
    }

    func test_modelURLsResolveBundledObjectAndPoseModels() throws {
        XCTAssertNotNil(BallSpeedModelProvider.modelURL(for: .objectDetection))
        XCTAssertNotNil(BallSpeedModelProvider.modelURL(for: .poseDetection))
    }
}
