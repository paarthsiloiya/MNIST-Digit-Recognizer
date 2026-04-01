import sys
import argparse
import os

# Ensure we use Keras 3 to load the Keras 3 .keras file, preventing tensorflowjs from downgrading us
os.environ['TF_USE_LEGACY_KERAS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras

def convert_model(input_path, output_dir):
    try:
        print(f"Loading Keras model from {input_path}...")
        model = keras.models.load_model(input_path)
        
        # Save it as a TF SavedModel temporarily
        temp_saved_model_dir = "temp_saved_model"
        tf.saved_model.save(model, temp_saved_model_dir)
        print("Exported to SavedModel format temporarily.")

        # Delay importing tensorflowjs until we've safely saved our keras 3 model
        from unittest.mock import MagicMock
        sys.modules['tensorflow_decision_forests'] = MagicMock()
        sys.modules['jax'] = MagicMock()
        sys.modules['jax.experimental'] = MagicMock()
        sys.modules['jax.experimental.jax2tf'] = MagicMock()
        sys.modules['flax'] = MagicMock()
        import tensorflowjs as tfjs

        os.makedirs(output_dir, exist_ok=True)
        print(f"Converting to tfjs graph model in {output_dir}...")
        
        # Use convert_tf_saved_model instead of save_keras_model
        tfjs.converters.convert_tf_saved_model(temp_saved_model_dir, output_dir)
        
        print("Conversion successful.")
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert Keras model to TFJS format")
    parser.add_argument('--input', type=str, required=True, help="Input .keras model path")
    parser.add_argument('--output', type=str, required=True, help="Output directory for TFJS model")
    args = parser.parse_args()
    
    convert_model(args.input, args.output)
