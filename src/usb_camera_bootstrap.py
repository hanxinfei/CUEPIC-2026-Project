import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="USB camera bootstrap for Raspberry Pi 5 + future YOLO pipeline"
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--window-name", type=str, default="USB Camera 480p", help="Preview window title")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parent / "best.onnx",
        help="Path to YOLO model (.pt or .onnx)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device: cpu or cuda:0")
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


def load_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    return YOLO(str(model_path), task="detect")



def process_frame(frame, model: YOLO, conf: float, device: str):
    results = model.predict(frame, conf=conf, device=device, verbose=False)
    return results[0].plot()



def main() -> None:
    args = parse_args()
    model = load_model(args.model_path)

    cap = open_camera(args.camera_index, args.width, args.height)

    prev_time = time.perf_counter()
    smoothed_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed, exiting loop.")
                break

            frame = process_frame(frame, model, args.conf, args.device)

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
