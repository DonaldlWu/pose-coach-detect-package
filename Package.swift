// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "BallSpeedKit",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(name: "BallSpeedKit", targets: ["BallSpeedKit"]),
    ],
    targets: [
        .target(
            name: "BallSpeedKit",
            path: "Sources/BallSpeedKit",
            resources: [
                .copy("Resources/yolov8n.mlpackage"),
                .copy("Resources/yolov8s-pose.mlpackage"),
            ]
        ),
        .testTarget(
            name: "BallSpeedKitTests",
            dependencies: ["BallSpeedKit"]
        ),
    ]
)
