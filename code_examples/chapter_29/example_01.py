"""
Chapter 29 - Example 1
Extracted from Healthcare AI Implementation Guide
"""

\# Excerpt from sustainable_ai_healthcare_code.py
\# --- 1. Quantization Example ---

def train_and_quantize_model(model, x_train, y_train, x_test, y_test):
    \# ... (model compilation and training)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    def representative_data_gen():
        for input_value in x_train.take(100):
            yield [input_value]
    converter.representative_dataset = representative_data_gen
    try:
        quantized_tflite_model = converter.convert()
        print("Model successfully quantized to INT8.")
        return quantized_tflite_model
    except Exception as e:
        print(f"Error during quantization: {e}")
        return None