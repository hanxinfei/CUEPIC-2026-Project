import argparse
import time

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="USB camera bootstrap for Raspberry Pi 5 + future YOLO pipeline"
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--window-name", type=str, default="USB Camera 480p", help="Preview window title")
    return parser.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open USB camera at index={index}. Check connection and index."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Keep this format flexible for later YOLO models that prefer BGR 8-bit frames.
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened. Requested: {width}x{height}, actual: {actual_width}x{actual_height}")

    return cap


def draw_fps(frame, fps: float) -> None:
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2

    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size

    margin = 10
    x = frame.shape[1] - text_w - margin
    y = margin + text_h

    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)



def process_frame(frame):
    # Future hook: run YOLO inference here and draw detections before display.
    return frame



def main() -> None:
    args = parse_args()

    cap = open_camera(args.camera_index, args.width, args.height)

    prev_time = time.perf_counter()
    smoothed_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed, exiting loop.")
                break

            frame = process_frame(frame)

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now

            instant_fps = 1.0 / dt if dt > 0 else 0.0
            if smoothed_fps == 0.0:
                smoothed_fps = instant_fps
            else:
                smoothed_fps = 0.9 * smoothed_fps + 0.1 * instant_fps

            draw_fps(frame, smoothed_fps)
            cv2.imshow(args.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
