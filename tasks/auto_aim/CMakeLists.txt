cmake_minimum_required(VERSION 3.16)

set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# ── Optional TensorRT ────────────────────────────────────────────────────────
option(USE_TENSORRT "Build with TensorRT CUDA inference backend" OFF)
if(USE_TENSORRT)
  find_package(CUDA REQUIRED)
  # Adjust path to match your TensorRT installation
  set(TensorRT_DIR "/usr/lib/x86_64-linux-gnu" CACHE PATH "TensorRT library path")
  find_library(NVINFER_LIB nvinfer REQUIRED PATHS ${TensorRT_DIR})
  find_library(NVONNXPARSER_LIB nvonnxparser REQUIRED PATHS ${TensorRT_DIR})
  message(STATUS "TensorRT enabled — found nvinfer at ${NVINFER_LIB}")
  add_compile_definitions(USE_TENSORRT)
endif()

add_subdirectory(planner/tinympc)

add_library(auto_aim OBJECT 
    armor.cpp
    classifier.cpp 
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    yolos/yolo_trt.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
)

target_link_libraries(auto_aim io openvino::runtime tinympcstatic)

if(USE_TENSORRT)
  target_include_directories(auto_aim PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(auto_aim ${NVINFER_LIB} ${NVONNXPARSER_LIB} ${CUDA_LIBRARIES})
  target_link_libraries(auto_aim ${CUDA_cuda_LIBRARY} ${CUDA_cudart_LIBRARY})
endif()
