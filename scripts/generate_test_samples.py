import numpy as np
import tensorflow as tf
import json
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(n, output_file):
    print(f"Generating {n} test samples from MNIST dataset...")
    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    X_test_sub = X_test[:n]
    y_test_sub = y_test[:n]
    
    X_test_processed = np.expand_dims(X_test_sub, axis=-1).astype('float32') / 255.0
    
    samples = {
        "images": X_test_processed.reshape(n, -1).tolist(), # Flatten each (28,28,1) into a list of 784 floats
        "labels": y_test_sub.tolist()
    }
    
    with open(output_file, 'w') as f:
        json.dump(samples, f)
        
    print(f"Saved {n} samples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MNIST test samples")
    parser.add_argument('--n', type=int, default=1000, help="Number of test samples to generate")
    parser.add_argument('--output', type=str, default='scripts/test_samples.json', help="Output JSON file")
    args = parser.parse_args()
    
    main(args.n, args.output)
