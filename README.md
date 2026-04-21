# CUEPIC-2026-Project
CUEPIC-2026-Project

## Model acceleration format conversion

You can export `best.pt` to faster deployment formats (for example ONNX).

### 1) Export model

From the `CUEPIC-2026-Project` directory:

```powershell
python src/model_export.py --model-path src/best.pt --formats onnx --imgsz 320
```

Optional multi-format export:

```powershell
python src/model_export.py --model-path src/best.pt --formats onnx,tflite --imgsz 320
```

Export outputs are written under `src/exports/<format>/`.

### 2) Run camera pipeline with exported model

```powershell
python src/usb_camera_bootstrap.py --model-path src/exports/onnx/best.onnx --width 640 --height 480
```

### 3) Practical tips for FPS

- Start with `onnx` and `--imgsz 320`.
- Keep camera at `640x480` or lower while testing.
- If FPS is still low, combine this with frame skipping (for example infer every 2 to 3 frames).
