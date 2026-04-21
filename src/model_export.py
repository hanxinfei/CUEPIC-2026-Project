import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO model to accelerated formats")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parent / "best.pt",
        help="Path to source .pt model",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="onnx",
        help="Comma-separated formats, e.g. onnx,tflite,ncnn,openvino",
    )
    parser.add_argument("--imgsz", type=int, default=320, help="Export image size")
    parser.add_argument("--device", type=str, default="cpu", help="Export device, e.g. cpu or 0")
    parser.add_argument("--half", action="store_true", help="Use FP16 where supported")
    parser.add_argument("--int8", action="store_true", help="Use INT8 where supported")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic shape where supported")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "exports",
        help="Directory for exported files",
    )
    return parser.parse_args()


def build_kwargs(fmt: str, args: argparse.Namespace) -> dict:
    kwargs = {
        "format": fmt,
        "imgsz": args.imgsz,
        "device": args.device,
        "project": str(args.output_dir),
        "name": fmt,
    }

    if args.half:
        kwargs["half"] = True

    if args.int8:
        kwargs["int8"] = True

    if args.dynamic and fmt in {"onnx", "engine", "openvino"}:
        kwargs["dynamic"] = True

    if fmt == "onnx":
        kwargs["opset"] = args.opset
        kwargs["simplify"] = True

    return kwargs


def main() -> None:
    args = parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    if not formats:
        raise ValueError("No export format provided")

    model = YOLO(str(args.model_path))

    print(f"Source model: {args.model_path}")
    print(f"Output directory: {args.output_dir}")

    for fmt in formats:
        print(f"\n[Export] format={fmt}")
        kwargs = build_kwargs(fmt, args)
        try:
            out = model.export(**kwargs)
            print(f"[OK] {fmt} -> {out}")
        except Exception as exc:
            print(f"[FAILED] {fmt}: {exc}")


if __name__ == "__main__":
    main()
