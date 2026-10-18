# ==============================================================================
# ESP32-S3 Main Component CMakeLists
# ==============================================================================

idf_component_register(
    SRCS 
        "main.cpp"
    INCLUDE_DIRS 
        "../include"
        "."
    REQUIRES 
        esp-tflite-micro
        esp_timer
        freertos
        esp_system
        log
)
