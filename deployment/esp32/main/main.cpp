/* ==============================================================================
 * Edge AI Model Compression Pipeline - ESP32-S3 Firmware Template
 * Target: ESP32-S3 (Xtensa LX7 Dual-Core @ 240 MHz)
 * Model: Compact DS-CNN for 12-Class Keyword Spotting (5,248 parameters, INT8)
 *
 * NOTE: Software-only deployment template. Run on ESP32-S3 hardware using ESP-IDF.
 * ============================================================================== */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_system.h"

// TensorFlow Lite for Microcontrollers Headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model Data and Label Headers
#define MODEL_DATA_IMPLEMENTATION
#include "model_data.h"

#define MODEL_LABELS_IMPLEMENTATION
#include "model_labels.h"

static const char* TAG = "KWS_ESP32";

// ==============================================================================
// Memory Allocation & Tensor Arena Configuration
// ==============================================================================
// Tensor Arena size is estimated based on intermediate activation map sizing:
// Input (40x47x1) = 1,880 B; Conv1 (20x24x28) = 13,440 B; DS-Block1 = 13,440 B.
// 30 KB allows comfortable double-buffering headroom on ESP32 SRAM.
constexpr int kTensorArenaSize = 30 * 1024;
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

// TFLM Runtime Pointers
static const tflite::Model* g_model = nullptr;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_input = nullptr;
static TfLiteTensor* g_output = nullptr;

// ==============================================================================
// Model Initialization & Operator Registration
// ==============================================================================
esp_err_t init_kws_engine() {
    ESP_LOGI(TAG, "Initializing Keyword Spotting Engine...");
    ESP_LOGI(TAG, "Model binary size: %u bytes", g_model_data_len);
    ESP_LOGI(TAG, "Expected SHA-256: %s", MODEL_DATA_SHA256);

    // 1. Verify FlatBuffer schema
    g_model = tflite::GetModel(g_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch! Model version: %lu, Runtime version: %d",
                 g_model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_FAIL;
    }

    // 2. Register required operators (exactly 5 verified kernels)
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAveragePool2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();

    // 3. Build interpreter
    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, tensor_arena, kTensorArenaSize);
    g_interpreter = &static_interpreter;

    // 4. Allocate memory from Tensor Arena
    TfLiteStatus allocate_status = g_interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed! Consider increasing kTensorArenaSize.");
        return ESP_FAIL;
    }

    // 5. Inspect input tensor (Expected: [1, 40, 47, 1], INT8)
    g_input = g_interpreter->input(0);
    if (g_input->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "Input tensor type mismatch! Expected INT8, found %d", g_input->type);
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "Input Tensor: shape=[%d, %d, %d, %d], scale=%.5f, zp=%ld",
             g_input->dims->data[0], g_input->dims->data[1],
             g_input->dims->data[2], g_input->dims->data[3],
             g_input->params.scale, g_input->params.zero_point);

    // 6. Inspect output tensor (Expected: [1, 12], INT8)
    g_output = g_interpreter->output(0);
    if (g_output->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "Output tensor type mismatch! Expected INT8, found %d", g_output->type);
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "Output Tensor: shape=[%d, %d], scale=%.5f, zp=%ld",
             g_output->dims->data[0], g_output->dims->data[1],
             g_output->params.scale, g_output->params.zero_point);

    ESP_LOGI(TAG, "KWS Engine initialized successfully. Tensor Arena used: %zu / %d bytes",
             g_interpreter->arena_used_bytes(), kTensorArenaSize);

    return ESP_OK;
}

// ==============================================================================
// Real-Time Inference Execution & Softmax Dequantization
// ==============================================================================
esp_err_t run_kws_inference(const int8_t* mfcc_features,
                            float* out_probabilities,
                            int* out_predicted_class) {
    if (!g_interpreter || !g_input || !g_output) {
        return ESP_ERR_INVALID_STATE;
    }

    // Copy input features (40 x 47 x 1 = 1,880 bytes)
    std::memcpy(g_input->data.int8, mfcc_features, g_input->bytes);

    int64_t t_start = esp_timer_get_time();
    TfLiteStatus invoke_status = g_interpreter->Invoke();
    int64_t t_end = esp_timer_get_time();

    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "TFLM Invoke() failed!");
        return ESP_FAIL;
    }

    float latency_ms = (t_end - t_start) / 1000.0f;

    // Dequantize INT8 logits and compute Softmax probabilities
    const int8_t* raw_output = g_output->data.int8;
    float dequantized[MODEL_NUM_CLASSES];
    float max_logit = -1e9f;

    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        dequantized[i] = (raw_output[i] - g_output->params.zero_point) * g_output->params.scale;
        if (dequantized[i] > max_logit) {
            max_logit = dequantized[i];
        }
    }

    // Numerically stable Softmax
    float sum_exp = 0.0f;
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        dequantized[i] = std::exp(dequantized[i] - max_logit);
        sum_exp += dequantized[i];
    }

    int best_class = 0;
    float best_prob = 0.0f;
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        out_probabilities[i] = dequantized[i] / sum_exp;
        if (out_probabilities[i] > best_prob) {
            best_prob = out_probabilities[i];
            best_class = i;
        }
    }

    *out_predicted_class = best_class;

    ESP_LOGI(TAG, "Inference completed in %.2f ms | Detected: %s (Confidence: %.1f%%)",
             latency_ms, kModelLabels[best_class], best_prob * 100.0f);

    return ESP_OK;
}

// ==============================================================================
// Application Entrypoint
// ==============================================================================
extern "C" void app_main() {
    ESP_LOGI(TAG, "=== Edge AI Model Compression: ESP32-S3 Virtual Demonstrator ===");
    ESP_LOGI(TAG, "CPU Clock: 240 MHz | RAM: 512 KB Internal SRAM");

    if (init_kws_engine() != ESP_OK) {
        ESP_LOGE(TAG, "FATAL: Failed to initialize KWS engine.");
        return;
    }

    // Synthetic smoke test buffer (40x47 = 1,880 int8 values)
    static int8_t sample_mfcc_input[40 * 47];
    std::memset(sample_mfcc_input, 0, sizeof(sample_mfcc_input));

    float probs[MODEL_NUM_CLASSES];
    int pred_class = -1;

    ESP_LOGI(TAG, "Starting continuous inference loop (simulated audio stream)...");
    while (true) {
        run_kws_inference(sample_mfcc_input, probs, &pred_class);
        vTaskDelay(pdMS_TO_TICKS(1000)); // Run once per second
    }
}
