#!/usr/bin/env python3
"""
sp_vision_25 推理后端 Benchmark（Python 版）
=============================================
对比以下后端的 YOLO 推理性能:
  1) OpenVINO (CPU / Intel GPU)
  2) ONNX Runtime (CPU / CUDA)
  3) TensorRT (CUDA) — 需要先转 engine

用法:
  python scripts/benchmark.py --model assets/best2-sim.onnx --img assets/demo/demo.avi

依赖:
  pip install opencv-python openvino onnxruntime-gpu
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path


# ── 公共预处理 (letterbox → NCHW float32) ──────────────────────────────
def preprocess(img: np.ndarray, input_size: int = 640) -> tuple:
    """Letterbox resize + NCHW float32 normalize, return (tensor, scale)."""
    h, w = img.shape[:2]
    scale = min(input_size / h, input_size / w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized

    # NCHW float32, keep BGR order
    blob = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = blob[np.newaxis, ...]  # [1, 3, H, W]
    return blob, scale


def parse_output(
    output: np.ndarray,           # [1, channels, 8400] float32
    scale: float,
    img_shape: tuple,             # (H, W)
    score_thresh: float = 0.7,
    nms_thresh: float = 0.3,
) -> list[dict]:
    """NMS + filter, return list of {bbox, corners, confidence, color_id, num_id}."""
    # output shape: let's handle both layouts
    if output.ndim == 3:
        _, channels, dets = output.shape
        output = output.squeeze(0).T  # [8400, channels]
    elif output.ndim == 2:
        dets, channels = output.shape
    else:
        raise ValueError(f"Unexpected output shape {output.shape}")

    boxes = []
    confs = []
    corners = []
    color_ids = []
    num_ids = []

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x)) if x > 0 else np.exp(x) / (1.0 + np.exp(x))

    for r in range(dets):
        row = output[r]
        obj_score = sigmoid(row[4].item())
        if obj_score < score_thresh:
            continue

        # Corners: assume same layout as YOLOV5 code
        # col 0,1 = center; 2,3 = pt2; 4,5 = pt3; 6,7 = pt4
        pts = np.array([
            [row[0] / scale, row[1] / scale],
            [row[6] / scale, row[7] / scale],
            [row[4] / scale, row[5] / scale],
            [row[2] / scale, row[3] / scale],
        ], dtype=np.float32)

        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        box = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

        # Class scores
        # For 22-channel: [9:13]=color, [13:22]=num
        # For 15-channel: [9:15]=mixed
        if channels >= 22:
            color_ids.append(int(np.argmax(row[9:13])))
            num_ids.append(int(np.argmax(row[13:22])))
        else:
            # 15-channel: assume first 2 = color, rest = num
            n_color = min(2, channels - 9)
            color_ids.append(int(np.argmax(row[9:9 + n_color])))
            num_ids.append(0)

        boxes.append(box)
        confs.append(float(obj_score))
        corners.append(pts)

    # NMS
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, confs, score_thresh, nms_thresh)
    if isinstance(indices, tuple):
        indices = indices[0]

    results = []
    for i in indices.flatten() if len(indices.shape) > 1 else indices:
        results.append({
            "bbox": boxes[i],
            "confidence": confs[i],
            "corners": corners[i],
            "color_id": color_ids[i],
            "num_id": num_ids[i],
        })
    return results


# ── 后端封装 ──────────────────────────────────────────────────────────

class OpenVINORunner:
    def __init__(self, model_xml: str, device: str = "CPU"):
        import openvino as ov
        core = ov.Core()
        model = core.read_model(model_xml)
        # compile
        self.compiled = core.compile_model(model, device)
        self.infer_req = self.compiled.create_infer_request()

    def infer(self, blob: np.ndarray) -> np.ndarray:
        self.infer_req.set_input_tensor(blob)  # assuming NHWC for OpenVINO
        # Convert NCHW to NHWC if needed
        # The compiled model's input layout determines this
        self.infer_req.infer()
        return self.infer_req.get_output_tensor().data

    @property
    def name(self):
        return "OpenVINO"


class ONNXRuntimeRunner:
    def __init__(self, model_onnx: str, provider: str = "CPUExecutionProvider"):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_onnx, providers=[provider])
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, blob: np.ndarray) -> np.ndarray:
        # ONNX model expects NCHW
        return self.session.run(None, {self.input_name: blob})[0]

    @property
    def name(self):
        return "ONNXRuntime"


class TensorRTRunner:
    def __init__(self, engine_path: str):
        """Load TensorRT engine and allocate buffers.
        Requires: pip install tensorrt
        """
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = None

        import pycuda.driver as cuda
        import pycuda.autoinit

        # Allocate device memory
        self.d_input = cuda.mem_alloc(self.engine.get_binding_shape(0).num_elements() * 4)
        self.d_output = cuda.mem_alloc(self.engine.get_binding_shape(1).num_elements() * 4)
        self.h_output = np.empty(
            self.engine.get_binding_shape(1).num_elements(), dtype=np.float32
        )

    def infer(self, blob: np.ndarray) -> np.ndarray:
        import pycuda.driver as cuda
        cuda.memcpy_htod(self.d_input, blob)
        self.context.execute_v2([int(self.d_input), int(self.d_output)])
        cuda.memcpy_dtoh(self.h_output, self.d_output)
        # Reshape to original dims
        out_shape = self.engine.get_binding_shape(1)
        return self.h_output.reshape(out_shape)

    @property
    def name(self):
        return "TensorRT"


# ── Benchmark ──────────────────────────────────────────────────────────

def benchmark_runner(runner, frames: list[np.ndarray], warmup: int = 10):
    """Measure average inference time over a list of frames."""
    # Warmup
    blob, _ = preprocess(frames[0])
    for _ in range(warmup):
        _ = runner.infer(blob)

    # Timed runs
    times = []
    det_results = []
    for frame in frames:
        blob, scale = preprocess(frame)
        t0 = time.perf_counter()
        output = runner.infer(blob)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        times.append(elapsed)
        results = parse_output(output, scale, frame.shape[:2])
        det_results.append(results)

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    fps = 1000.0 / avg_ms
    avg_dets = np.mean([len(r) for r in det_results])

    return {
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "fps": fps,
        "avg_dets": avg_dets,
        "times": times,
    }


def main():
    parser = argparse.ArgumentParser(description="YOLO inference backend benchmark")
    parser.add_argument("--model", "-m", default="assets/best2-sim.onnx",
                        help="Path to ONNX model (or .xml/.bin for OpenVINO)")
    parser.add_argument("--img", "-i", default="assets/demo/demo.avi",
                        help="Path to demo video file")
    parser.add_argument("--num-frames", "-n", type=int, default=100,
                        help="Number of frames to benchmark")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup iterations")
    parser.add_argument("--openvino-device", default="CPU",
                        help="OpenVINO device: CPU, GPU, NPU")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    # Load frames from video
    cap = cv2.VideoCapture(str(args.img))
    frames = []
    while len(frames) < args.num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames from {args.img}")

    if not frames:
        print("[ERROR] No frames loaded!")
        return

    runners = []

    # 1. OpenVINO (if model is .xml)
    xml_path = model_path.with_suffix(".xml")
    if xml_path.exists():
        try:
            runners.append(OpenVINORunner(str(xml_path), device=args.openvino_device))
            print(f"[OK] OpenVINO ({args.openvino_device})")
        except Exception as e:
            print(f"[FAIL] OpenVINO: {e}")
    else:
        print(f"[SKIP] OpenVINO: {xml_path} not found")

    # 2. ONNX Runtime
    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        for prov in providers:
            try:
                runners.append(ONNXRuntimeRunner(str(model_path), provider=prov))
                print(f"[OK] ONNXRuntime ({prov})")
                break
            except Exception:
                continue
    except Exception as e:
        print(f"[SKIP] ONNXRuntime: {e}")

    # 3. TensorRT
    engine_path = model_path.with_suffix(".engine")
    if engine_path.exists():
        try:
            runners.append(TensorRTRunner(str(engine_path)))
            print(f"[OK] TensorRT")
        except Exception as e:
            print(f"[FAIL] TensorRT: {e}")
    else:
        print(f"[SKIP] TensorRT: {engine_path} not found (run scripts/onnx2trt.py first)")

    if not runners:
        print("[ERROR] No runners available!")
        return

    # Run benchmark
    print()
    print(f"{'Backend':<30} {'Avg(ms)':<10} {'Std(ms)':<10} {'FPS':<10} {'Avg Dets':<10}")
    print("-" * 70)
    for runner in runners:
        stats = benchmark_runner(runner, frames, warmup=args.warmup)
        print(
            f"{runner.name:<30} "
            f"{stats['avg_ms']:<10.2f} "
            f"{stats['std_ms']:<10.2f} "
            f"{stats['fps']:<10.1f} "
            f"{stats['avg_dets']:<10.1f}"
        )

    # Speedup vs baseline
    if len(runners) >= 2:
        baseline = runners[0]
        baseline_ms = benchmark_runner(runners[0], frames, warmup=args.warmup)["avg_ms"]
        print()
        print(f"Speedup vs {baseline.name}:")
        for runner in runners[1:]:
            r_ms = benchmark_runner(runner, frames, warmup=args.warmup)["avg_ms"]
            print(f"  {runner.name:<25} {baseline_ms / r_ms:.2f}x")


if __name__ == "__main__":
    main()
