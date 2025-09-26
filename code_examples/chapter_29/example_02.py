"""
Chapter 29 - Example 2
Extracted from Healthcare AI Implementation Guide
"""

\# Excerpt from sustainable_ai_healthcare_code.py
\# --- 2. Pruning Example (using TensorFlow Model Optimization Toolkit) ---

import tensorflow_model_optimization as tfmot

def create_prunable_model():
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.50,
            final_sparsity=0.80,
            begin_step=0,
            end_step=1000)
    }
    model = tf.keras.models.Sequential([
        tfmot.sparsity.keras.prune_low_magnitude(
            tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
            **pruning_params),
        tfmot.sparsity.keras.prune_low_magnitude(
            tf.keras.layers.Dense(10, activation='softmax'),
            **pruning_params)
    ])
    return model

def train_and_prune_model(model, x_train, y_train, x_test, y_test):
    \# ... (model compilation and training with callbacks)
    model_for_export = tfmot.sparsity.keras.strip_pruning(model)
    print("Model successfully pruned and wrappers stripped.")
    return model_for_export