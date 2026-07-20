#!/usr/bin/env python3
"""
ONNX → TensorRT Engine 转换脚本
==================================
将 best2-sim.onnx (或任意 YOLO ONNX) 转为 TensorRT engine，支持 FP16 / INT8。

用法:
  python scripts/onnx2trt.py assets/best2-sim.onnx --fp16
  python scripts/onnx2trt.py assets/best2-sim.onnx --int8 --calib-dir ./calib_images

依赖:
  pip install tensorrt onnx
  或使用 trtexec: trtexec --onnx=best2-sim.onnx --saveEngine=best2-sim.engine --fp16
"""

import argparse
import os
import sys
from pathlib import Path


def build_engine_trtexec(onnx_path: str, output_path: str, precision: str = "fp16"):
    """Use trtexec (TensorRT CLI) to build engine."""
    import subprocess

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
    ]
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[OK] Engine saved to {output_path}")


def build_engine_python_api(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    calib_cache: str = "",
    calib_images_dir: str = "",
):
    """Use TensorRT Python API to build engine (requires tensorrt package)."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for err in range(parser.num_errors):
                print(f"Parse error {err}: {parser.get_error(err)}")
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("[WARN] Platform does not have fast FP16, falling back to FP32")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] FP16 enabled")
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            print("[WARN] Platform does not have fast INT8, falling back to FP32")
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            if calib_cache:
                config.int8_calibrator = trt.IInt8Calibrator(calib_cache)
            print("[INFO] INT8 enabled")

    # Build engine
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    input_shape = network.get_input(0).shape
    # Use static shape from ONNX
    profile.set_shape(input_name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[ERROR] Failed to build engine!")
        sys.exit(1)

    with open(output_path, "wb") as f:
        f.write(serialized)

    print(f"[OK] Engine saved to {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX YOLO model to TensorRT engine"
    )
    parser.add_argument("onnx_path", type=str, help="Path to ONNX model")
    parser.add_argument(
        "--output", "-o", type=str, default="",
        help="Output engine path (default: <model>.engine)"
    )
    parser.add_argument(
        "--precision", "-p", type=str, default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="Inference precision"
    )
    parser.add_argument(
        "--use-trtexec", action="store_true",
        help="Use trtexec CLI instead of Python API (recommended)"
    )
    parser.add_argument(
        "--calib-dir", type=str, default="",
        help="Directory with calibration images for INT8"
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        print(f"[ERROR] ONNX model not found: {onnx_path}")
        sys.exit(1)

    output_path = args.output or str(onnx_path.with_suffix(f".{args.precision}.engine"))
    print(f"Converting {onnx_path} → {output_path} ({args.precision})")

    if args.use_trtexec:
        build_engine_trtexec(str(onnx_path), output_path, args.precision)
    else:
        build_engine_python_api(
            str(onnx_path), output_path, args.precision, calib_images_dir=args.calb_dir
        )


if __name__ == "__main__":
    main()
