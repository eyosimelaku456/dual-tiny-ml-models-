/*
 * ============================================================
 * main.cpp - ESP32 Battery SOH Inference Firmware
 * ============================================================
 * Author  : TinyML Battery SOH Pipeline
 * Target  : ESP32 (Xtensa LX6, 520 KB RAM)
 * Framework: TensorFlow Lite for Microcontrollers (TFLM)
 *
 * Description:
 *   Loads an INT8-quantized TFLite model and performs real-time
 *   Battery State of Health (SOH) estimation from EIS impedance
 *   features received over Serial.
 *
 * Pipeline on-device:
 *   1. Read raw impedance features (from Serial or hardcoded test)
 *   2. Normalize: (x - mean) / std  [using precomputed scaler params]
 *   3. Quantize input: int8 = (float / scale) + zero_point
 *   4. Run TFLM interpreter
 *   5. Dequantize output: float = (int8 - zero_point) * scale
 *   6. Print SOH prediction + inference latency
 *
 * Memory Budget (ESP32 ~520 KB SRAM):
 *   - Model:        < 100 KB (from model_data.h)
 *   - Tensor Arena: 100 KB
 *   - Stack/Globals: ~30 KB
 *   Total:          < 230 KB  ✓
 *
 * Build with PlatformIO:
 *   platform = espressif32
 *   board = esp32dev
 *   framework = arduino
 *   lib_deps = tensorflow/TensorFlowLite_ESP32
 * ============================================================
 */

#include <Arduino.h>

// TensorFlow Lite for Microcontrollers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Auto-generated model byte array (from convert_to_c_array.py)
#include "model_data.h"
#include "model_data_rul.h"

// Auto-generated scaler parameters (from preprocess.py)
#include "scaler_params.h"
#include "scaler_params_rul.h"

// ─────────────────────────────────────────────────────────────
// CONFIGURATION
// ─────────────────────────────────────────────────────────────

// Tensor arena size: must fit all tensors for the model
// Measured empirically; increase if TFLM reports insufficient memory
static constexpr size_t kTensorArenaSize = 60 * 1024;  // 60 KB
static constexpr size_t kTensorArenaSizeRul = 15 * 1024;  // 15 KB for RUL model

// Serial baud rate for debug output
static constexpr uint32_t kBaudRate = 115200;

// ─────────────────────────────────────────────────────────────
// GLOBAL TFLM OBJECTS
// ─────────────────────────────────────────────────────────────

// Tensor arena buffer (static allocation on DRAM)
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
alignas(16) static uint8_t tensor_arena_rul[kTensorArenaSizeRul];

// TFLM interpreter and model (SOH)
static const tflite::Model*         g_model     = nullptr;
static tflite::MicroInterpreter*    g_interpreter = nullptr;
static TfLiteTensor*                g_input     = nullptr;
static TfLiteTensor*                g_output    = nullptr;

// TFLM interpreter and model (RUL)
static const tflite::Model*         g_model_rul     = nullptr;
static tflite::MicroInterpreter*    g_interpreter_rul = nullptr;
static TfLiteTensor*                g_input_rul     = nullptr;
static TfLiteTensor*                g_output_rul    = nullptr;

// Op resolver: registers all supported operations
static tflite::AllOpsResolver       g_resolver;

// ─────────────────────────────────────────────────────────────
// TEST SAMPLE (Replace with real sensor data)
// ─────────────────────────────────────────────────────────────
// This is a real sample from the EIS dataset (un-normalized raw values).
// Replace with your actual sensor readings during deployment.
// Array length must equal INPUT_SIZE (defined in scaler_params.h)

static const float raw_sample[INPUT_SIZE] = {
    // Example values for r_1 .. r_40 (real impedance at 40 frequencies)
    // !! Replace these with actual measured impedance values !!
    0.00152f, 0.00148f, 0.00145f, 0.00143f, 0.00141f,
    0.00140f, 0.00139f, 0.00138f, 0.00138f, 0.00137f,
    0.00137f, 0.00137f, 0.00137f, 0.00137f, 0.00138f,
    0.00138f, 0.00139f, 0.00140f, 0.00141f, 0.00143f,
    0.00145f, 0.00148f, 0.00151f, 0.00155f, 0.00159f,
    0.00164f, 0.00169f, 0.00175f, 0.00182f, 0.00189f,
    0.00197f, 0.00206f, 0.00215f, 0.00225f, 0.00236f,
    0.00248f, 0.00261f, 0.00275f, 0.00290f, 0.00306f,
    // temperature (°C) and soc (0-1) if INPUT_SIZE > 40
    // 25.0f, 0.80f,  // ← uncomment if your model uses temp+soc
};

static const float raw_sample_rul[INPUT_SIZE_RUL] = {
    // Discharge Time, Decrement, Max Vol, Min Vol, Time at 4.15, Time CC, Charging time
    2595.3f, 1151.4885f, 3.67f, 3.211f, 5460.001f, 6755.01f, 10777.82f
};

// ─────────────────────────────────────────────────────────────
// HELPER: NORMALIZE INPUT
// ─────────────────────────────────────────────────────────────

/**
 * @brief Normalize raw feature array using precomputed StandardScaler params.
 *
 * Formula: normalized[i] = (raw[i] - FEATURE_MEAN[i]) / FEATURE_STD[i]
 *
 * CRITICAL: Must use exactly the same mean/std as Python preprocessing.
 * These values are embedded in scaler_params.h (auto-generated).
 *
 * @param raw       Input raw feature array [INPUT_SIZE]
 * @param normalized Output normalized array [INPUT_SIZE]
 */
void normalize_input(const float* raw, float* normalized) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        normalized[i] = (raw[i] - FEATURE_MEAN[i]) / FEATURE_STD[i];
    }
}

// ─────────────────────────────────────────────────────────────
// HELPER: QUANTIZE INPUT (float → int8)
// ─────────────────────────────────────────────────────────────

/**
 * @brief Quantize a float value to int8 using TFLite scale/zero_point.
 *
 * Formula: quantized = round(float_val / scale) + zero_point
 * Clamps result to valid int8 range [-128, 127].
 *
 * @param val        Normalized float value
 * @param scale      Input tensor quantization scale
 * @param zero_point Input tensor quantization zero_point
 * @return int8_t quantized value
 */
int8_t quantize_input(float val, float scale, int32_t zero_point) {
    float q = (val / scale) + static_cast<float>(zero_point);
    // Clamp to int8 range
    if (q < -128.0f) q = -128.0f;
    if (q >  127.0f) q =  127.0f;
    return static_cast<int8_t>(roundf(q));
}

// ─────────────────────────────────────────────────────────────
// HELPER: DEQUANTIZE OUTPUT (int8 → float)
// ─────────────────────────────────────────────────────────────

/**
 * @brief Dequantize INT8 output back to float SOH value.
 *
 * Formula: float_val = (int8_val - zero_point) * scale
 *
 * @param q_val      Quantized INT8 output from model
 * @param scale      Output tensor quantization scale
 * @param zero_point Output tensor quantization zero_point
 * @return float dequantized SOH prediction
 */
float dequantize_output(int8_t q_val, float scale, int32_t zero_point) {
    return (static_cast<float>(q_val) - static_cast<float>(zero_point)) * scale;
}

// ─────────────────────────────────────────────────────────────
// PRINT UTILITIES
// ─────────────────────────────────────────────────────────────

void print_separator(char c = '-', int width = 50) {
    for (int i = 0; i < width; i++) Serial.print(c);
    Serial.println();
}

void print_banner() {
    print_separator('=');
    Serial.println("  TinyML Battery SOH Estimator");
    Serial.println("  ESP32 + TensorFlow Lite Micro");
    Serial.println("  EIS Impedance → State of Health");
    print_separator('=');
}

// ─────────────────────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(kBaudRate);
    delay(1000);  // Allow Serial to initialize

    print_banner();

    // ── Step 1: Load model ──────────────────────────────────
    Serial.println("[1] Loading model from flash...");
    g_model = tflite::GetModel(g_model_data);

    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[ERROR] Model schema version mismatch: %d vs %d\n",
                      g_model->version(), TFLITE_SCHEMA_VERSION);
        while (true) delay(1000);  // Halt
    }

    Serial.printf("    Model size    : %u bytes (%.2f KB)\n",
                  g_model_data_len, g_model_data_len / 1024.0f);
    Serial.println("    Model loaded  : OK ✓");

    // ── Step 2: Build interpreter ───────────────────────────
    Serial.println("[2] Building TFLM interpreter...");
    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    static tflite::MicroInterpreter static_interpreter(
        g_model, g_resolver, tensor_arena, kTensorArenaSize, error_reporter
    );
    g_interpreter = &static_interpreter;

    // ── Step 3: Allocate tensors ────────────────────────────
    TfLiteStatus alloc_status = g_interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        Serial.println("[ERROR] AllocateTensors() failed!");
        Serial.printf("        Tensor arena may be too small (%d KB)\n",
                      kTensorArenaSize / 1024);
        while (true) delay(1000);
    }

    // Report actual arena usage
    size_t used = g_interpreter->arena_used_bytes();
    Serial.printf("    Tensor arena  : %d KB allocated, %d KB used\n",
                  kTensorArenaSize / 1024, (int)(used / 1024));
    Serial.println("    Tensors ready : OK ✓");

    // ── Step 4: Get I/O tensor references ──────────────────
    g_input  = g_interpreter->input(0);
    g_output = g_interpreter->output(0);

    // Print tensor info
    Serial.println("[3] Tensor details:");
    Serial.printf("    Input  shape  : [%d, %d]\n",
                  g_input->dims->data[0], g_input->dims->data[1]);
    Serial.printf("    Input  type   : %s\n",
                  (g_input->type == kTfLiteFloat32) ? "FLOAT32" :
                  (g_input->type == kTfLiteInt8)    ? "INT8"    : "OTHER");
    Serial.printf("    Output type   : %s\n",
                  (g_output->type == kTfLiteFloat32) ? "FLOAT32" :
                  (g_output->type == kTfLiteInt8)    ? "INT8"    : "OTHER");
    Serial.printf("    Input  scale  : %.6f, zero_point: %d\n",
                  g_input->params.scale, g_input->params.zero_point);
    Serial.printf("    Output scale  : %.6f, zero_point: %d\n",
                  g_output->params.scale, g_output->params.zero_point);

    // ── Step 5: Scaler summary ──────────────────────────────
    Serial.printf("[4] Scaler params : %d features loaded ✓\n", INPUT_SIZE);

    // ── Setup RUL Model ──────────────────────────────────────
    Serial.println("[5] Loading RUL model...");
    g_model_rul = tflite::GetModel(g_model_data_rul);
    static tflite::MicroInterpreter static_interpreter_rul(
        g_model_rul, g_resolver, tensor_arena_rul, kTensorArenaSizeRul, error_reporter
    );
    g_interpreter_rul = &static_interpreter_rul;
    g_interpreter_rul->AllocateTensors();
    g_input_rul  = g_interpreter_rul->input(0);
    g_output_rul = g_interpreter_rul->output(0);
    Serial.println("    RUL Model loaded : OK ✓");

    print_separator('=');
    Serial.println("  System ready. Starting Dual-Inference loop.");
    print_separator('=');
    Serial.println();
}

// ─────────────────────────────────────────────────────────────
// INFERENCE FUNCTION
// ─────────────────────────────────────────────────────────────

/**
 * @brief Run one complete inference cycle.
 *
 * Steps:
 *   1. Normalize raw input using scaler params
 *   2. Fill model input tensor (handles float32 or int8 input type)
 *   3. Run interpreter
 *   4. Read and decode output
 *   5. Print results with timing
 *
 * @param raw_features  Raw (un-normalized) EIS feature array [INPUT_SIZE]
 */
void run_inference(const float* raw_features) {
    print_separator('-');

    // ── Normalize ─────────────────────────────────────────────
    float normalized[INPUT_SIZE];
    normalize_input(raw_features, normalized);

    // ── Fill input tensor ────────────────────────────────────
    bool is_float_input = (g_input->type == kTfLiteFloat32);
    bool is_int8_input  = (g_input->type == kTfLiteInt8);

    if (is_float_input) {
        // Model takes float input directly
        for (int i = 0; i < INPUT_SIZE; i++) {
            g_input->data.f[i] = normalized[i];
        }
    } else if (is_int8_input) {
        // Model takes INT8 input: quantize normalized floats
        float  in_scale = g_input->params.scale;
        int32_t in_zp   = g_input->params.zero_point;
        for (int i = 0; i < INPUT_SIZE; i++) {
            g_input->data.int8[i] = quantize_input(normalized[i], in_scale, in_zp);
        }
    } else {
        Serial.println("[ERROR] Unsupported input tensor type!");
        return;
    }

    // ── Run inference (measure latency) ──────────────────────
    uint32_t t_start = micros();
    TfLiteStatus invoke_status = g_interpreter->Invoke();
    uint32_t t_end   = micros();

    if (invoke_status != kTfLiteOk) {
        Serial.println("[ERROR] Inference invoke() failed!");
        return;
    }

    float latency_ms = (t_end - t_start) / 1000.0f;

    // ── Decode output ─────────────────────────────────────────
    float soh_prediction = 0.0f;

    if (g_output->type == kTfLiteFloat32) {
        soh_prediction = g_output->data.f[0];
    } else if (g_output->type == kTfLiteInt8) {
        float  out_scale = g_output->params.scale;
        int32_t out_zp   = g_output->params.zero_point;
        soh_prediction = dequantize_output(g_output->data.int8[0], out_scale, out_zp);
    }

    // ── Print results ─────────────────────────────────────────
    Serial.println("  INPUT (first 5 features):");
    Serial.print("    Raw      : ");
    for (int i = 0; i < min(5, INPUT_SIZE); i++) {
        Serial.printf("%.6f  ", raw_features[i]);
    }
    Serial.println();

    Serial.print("    Normalized: ");
    for (int i = 0; i < min(5, INPUT_SIZE); i++) {
        Serial.printf("%.4f  ", normalized[i]);
    }
    Serial.println();

    if (is_int8_input) {
        Serial.print("    Quantized : ");
        for (int i = 0; i < min(5, INPUT_SIZE); i++) {
            Serial.printf("%4d  ", g_input->data.int8[i]);
        }
        Serial.println();
    }

    Serial.println();
    Serial.print("  >> Predicted SOH    : ");
    Serial.print(soh_prediction, 2);
    Serial.println("% battery health");
    Serial.print("  Inference latency : ");
    Serial.print(latency_ms, 3);
    Serial.println(" ms");
    Serial.print("  Latency target    : ");
    Serial.println((latency_ms < 50.0f) ? "MET (< 50 ms)" : "EXCEEDED (> 50 ms)");
    Serial.flush();
}

// ─────────────────────────────────────────────────────────────
// LOOP
// ─────────────────────────────────────────────────────────────

void loop() {
    Serial.println("[INFERENCE] Running SOH test sample...");
    Serial.flush();
    run_inference(raw_sample);

    Serial.println("[INFERENCE] Running RUL test sample...");
    Serial.flush();

    // ── Normalize RUL ─────────────────────────────────────────
    float normalized_rul[INPUT_SIZE_RUL];
    for(int i=0; i<INPUT_SIZE_RUL; i++) {
        normalized_rul[i] = normalize_feature_rul(raw_sample_rul[i], i);
    }

    // ── Fill RUL tensor ──────────────────────────────────────
    if (g_input_rul->type == kTfLiteFloat32) {
        for(int i=0; i<INPUT_SIZE_RUL; i++) g_input_rul->data.f[i] = normalized_rul[i];
    } else if (g_input_rul->type == kTfLiteInt8) {
        float in_sc = g_input_rul->params.scale;
        int32_t in_zp = g_input_rul->params.zero_point;
        for(int i=0; i<INPUT_SIZE_RUL; i++) g_input_rul->data.int8[i] = quantize_input(normalized_rul[i], in_sc, in_zp);
    } else {
        Serial.println("[ERROR] RUL: Unsupported input tensor type!");
        Serial.flush();
        delay(5000);
        return;
    }

    // ── Run RUL inference ────────────────────────────────────
    Serial.println("    Running RUL Invoke()...");
    Serial.flush();
    TfLiteStatus rul_status = g_interpreter_rul->Invoke();
    if (rul_status != kTfLiteOk) {
        Serial.println("[ERROR] RUL Invoke() failed!");
        Serial.flush();
        delay(5000);
        return;
    }

    float rul_pred = 0.0f;
    if (g_output_rul->type == kTfLiteFloat32) {
        rul_pred = g_output_rul->data.f[0];
    } else {
        rul_pred = dequantize_output(g_output_rul->data.int8[0],
                                     g_output_rul->params.scale,
                                     g_output_rul->params.zero_point);
    }

    Serial.print("  >> Predicted RUL   : ");
    Serial.print(rul_pred, 1);
    Serial.println(" Cycles Remaining");
    Serial.flush();
    print_separator('-');

    // Run inference every 5 seconds
    // In production: replace with actual EIS sensor data acquisition
    Serial.println("\n  [Waiting 5s for next inference...]\n");
    delay(5000);
}

/*
 * ============================================================
 * SERIAL COMMAND INTERFACE (Optional Extension)
 * ============================================================
 * To send custom features via Serial Monitor:
 *   - Send 40+ comma-separated float values (e.g., r_1,r_2,...,r_40)
 *   - Firmware will parse and run inference
 *   - Format: "0.00152,0.00148,0.00145,...\n"
 *
 * Enable by adding to loop():
 *   if (Serial.available()) {
 *     String line = Serial.readStringUntil('\n');
 *     float features[INPUT_SIZE];
 *     parse_features(line, features);
 *     run_inference(features);
 *   }
 * ============================================================
 */
