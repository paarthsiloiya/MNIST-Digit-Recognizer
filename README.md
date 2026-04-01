# MNIST Digit Recognizer - Static Web UI

This project hosts a static, in-browser MNIST digit recognizer using [TensorFlow.js](https://www.tensorflow.org/js). 
The core capability is driven by an original Keras model (`Digit_Recognizer.keras`), which has been converted for web inference into a Tensor JS GraphModel.

## Features
- **In-browser Inference**: No backend required. Models run purely in your local browser environment using `tf.loadGraphModel()`.
- **Conversion Pipeline**: Automated conversion from offline Keras (`.keras`) file formats, wrapping an intermediate `SavedModel` extraction to parse correctly into `JSON` Graph structures for `@tensorflow/tfjs`.
- **Draw Canvas**: Clean integrated canvas drawing tool with automatic grayscale resizing and thresholding matching the offline preprocessing pipeline.
- **Automated Deployment**: GitHub Actions pipeline is pre-configured to build, convert, and host via GitHub Pages.

## Setup & Local Usage

1. Create a Python virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   # or
   .\.venv\Scripts\activate   # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
   # For Windows users: if you face issues with full tensorflowjs wheels (e.g. jax / orbax missing), 
   # use the isolated python script (`python scripts/convert_model.py`) to bypass unsupported dependencies using python Mocks natively!
   ```

3. Convert the offline model:
   ```bash
   python scripts/convert_model.py --input Digit_Recognizer.keras --output docs/model
   ```

4. Host the application locally:
   ```bash
   python -m http.server --directory docs 8000
   ```
   Navigate to [http://localhost:8000](http://localhost:8000)

## Verification Pipeline

To systematically test the numerical parity of the original Keras model versus the converted TFJS bundle, run:

```bash
# 1. Generate predictable test samples
python scripts/generate_test_samples.py --n 1000

# 2. Evaluate Keras
python scripts/evaluate_keras.py --samples scripts/test_samples.json --output scripts/keras_results.json

# 3. Evaluate TF.js (ensure `@tensorflow/tfjs-node` is correctly compiled)
node scripts/evaluate_tfjs.js --samples scripts/test_samples.json --output scripts/tfjs_results.json --model docs/model/model.json

# 4. Compare inference matrices
python scripts/compare_results.py --keras scripts/keras_results.json --tfjs scripts/tfjs_results.json --report reports/verification_report.json
```

## Results & Acceptance Metrics

- **Model format**: Verified inputs: `28x28`, grayscale, black background, preprocessed inside TF.js exactly matching the original Keras standard mapping of `pixel / 255.0`.
- **Target Accuracy**: ~99.5% on test batches.
- **Parity**: Identical predictions between Python native Keras and TensorFlow.js (Diff <= 0.5% or Match rate >= 98%).

Detailed reports are generated in `reports/verification_report.json`.

## Cleanup
Run `rm -rf scripts/*.json` to remove local temporary testing artifacts.