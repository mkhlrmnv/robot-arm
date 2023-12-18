#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/landmark.pb.h>
#include <mediapipe/framework/formats/landmark_list.pb.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <mediapipe/framework/port/opencv_highgui_inc.h>
#include <mediapipe/framework/port/opencv_imgproc_inc.h>
#include <mediapipe/framework/port/parse_text_proto.h>
#include <mediapipe/framework/port/status.h>
#include <mediapipe/framework/port/statusor.h>
#include <mediapipe/framework/port/ret_check.h>
#include <mediapipe/framework/port/status.h>
#include <mediapipe/framework/port/status_macros.h>

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";

mediapipe::Status RunHandTracking() {
  // Initialize MediaPipe Hands.
  mediapipe::calculator::GraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<mediapipe::calculator::GraphConfig>(R"(
        input_stream: "input_video"
        output_stream: "output_video"
        node {
          calculator: "HandLandmarksCalculator"
          input_stream: "input_video"
          output_stream: "output_video"
        }
      )");

  mediapipe::calculator::Graph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(graph_config));

  mediapipe::OutputStreamPoller poller(graph_config.output_stream(0));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  // OpenCV setup.
  cv::VideoCapture cap(0);
  RET_CHECK(cap.isOpened());

  while (cap.isOpened()) {
    cv::Mat frame;
    cap.read(frame);
    if (frame.empty()) {
      break;
    }

    // Convert OpenCV Mat to MediaPipe ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, frame.cols, frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_mat = mediapipe::formats::MatView(input_frame.get());
    frame.copyTo(input_mat);

    // Send input image to the graph.
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream,
        mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp())));

    // Process the graph.
    MP_RETURN_IF_ERROR(graph.WaitUntilIdle());

    // Get the output packet.
    mediapipe::Packet packet;
    if (!poller.Next(&packet)) {
      break;
    }

    // Retrieve hand landmarks from the output packet.
    const auto& hand_landmarks =
        packet.Get<mediapipe::NormalizedLandmarkList>().landmark();

    // Print wrist coordinates for each hand.
    for (const auto& landmark : hand_landmarks) {
      std::cout << "Wrist Coordinates: "
                << "X: " << landmark.x() << ", "
                << "Y: " << landmark.y() << ", "
                << "Z: " << landmark.z() << std::endl;
    }

    // Render hand landmarks on the frame (for visualization).
    mediapipe::ImageFrame output_frame = packet.Get<mediapipe::ImageFrame>();
    cv::Mat output_mat = mediapipe::formats::MatView(&output_frame);
    cv::imshow("Hand Tracking", output_mat);

    // Break the loop if 'q' is pressed.
    if (cv::waitKey(10) == 'q') {
      break;
    }
  }

  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return mediapipe::OkStatus();
}

int main() {
  mediapipe::Status run_status = RunHandTracking();
  if (!run_status.ok()) {
    std::cerr << "Failed to run the hand tracking program: "
              << run_status.message() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
