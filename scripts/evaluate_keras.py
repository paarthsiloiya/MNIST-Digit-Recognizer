import numpy as np
import tensorflow as tf
import json
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def evaluate(model_path, samples_file, output_file):
    print(f"Loading test samples from {samples_file}...")
    with open(samples_file, 'r') as f:
        data = json.load(f)
        
    X_test = np.array(data["images"]).reshape(-1, 28, 28, 1).astype('float32')
    y_test = np.array(data["labels"])
    
    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Running predictions...")
    preds = model.predict(X_test, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    
    accuracy = np.mean(pred_labels == y_test)
    print(f"Keras Accuracy evaluated on {len(y_test)} samples: {accuracy*100:.2f}%")
    
    results = {
        "accuracy": float(accuracy),
        "predicted_labels": pred_labels.tolist(),
        "confidences": preds.tolist()
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f)
        
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Keras Model")
    parser.add_argument('--model', type=str, default='Digit_Recognizer.keras', help="Keras model path")
    parser.add_argument('--samples', type=str, default='scripts/test_samples.json')
    parser.add_argument('--output', type=str, default='scripts/keras_results.json')
    args = parser.parse_args()
    
    evaluate(args.model, args.samples, args.output)
