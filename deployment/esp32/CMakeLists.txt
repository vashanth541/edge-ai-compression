# ==============================================================================
# Edge AI Model Compression Pipeline - ESP32-S3 Firmware Project
# Target: ESP32-S3 (Xtensa Dual-Core 240 MHz)
# Framework: ESP-IDF v5.x with TensorFlow Lite for Microcontrollers (TFLM)
# ==============================================================================

cmake_minimum_required(VERSION 3.16)

# Include ESP-IDF build system
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(kws_dscnn_esp32)
