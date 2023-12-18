load("@mediapipe//:mediapipe_deps.bzl", "mediapipe_deps")
mediapipe_deps()

cc_binary(
    name = "hand_tracking",
    srcs = ["hand_tracking.cpp"],
    deps = [
        "@opencv//:core",
        "@mediapipe//:hand_landmarks_calculator",
        "@mediapipe//:framework",
    ],
)
