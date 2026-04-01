import sys
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflowjs as tfjs

def convert_model(input_path, output_dir):
    try:
        print(f"Loading Keras model from {input_path}...")
        model = tf.keras.models.load_model(input_path)
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Converting to tfjs format in {output_dir}...")
        tfjs.converters.save_keras_model(model, output_dir)
        
        print("Conversion successful.")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert Keras model to TFJS format")
    parser.add_argument('--input', type=str, required=True, help="Input .keras model path")
    parser.add_argument('--output', type=str, required=True, help="Output directory for TFJS model")
    args = parser.parse_args()
    
    convert_model(args.input, args.output)
