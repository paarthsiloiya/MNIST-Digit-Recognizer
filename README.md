# MNIST Digit Recognizer - Static Web UI

This project hosts a static, in-browser MNIST digit recognizer using [TensorFlow.js](https://www.tensorflow.org/js). 
The core capability is driven by an original Keras 3 model (`Digit_Recognizer.keras`), which has been converted for web inference into a Tensor JS GraphModel.

**Live Demo:** [https://paarthsiloiya.github.io/MNIST-Digit-Recognizer/](https://paarthsiloiya.github.io/MNIST-Digit-Recognizer/)

## Features
- **In-browser Inference**: No backend required. Models run purely in your local browser environment using `tf.loadGraphModel()`.
- **Conversion Pipeline**: Automated conversion from offline Keras 3 (`.keras`) file formats. The workflow extracts an intermediate `SavedModel` to strip problematic parameters before converting to a `JSON` Graph structure via `tensorflowjs_converter`.
- **Draw Canvas**: Clean integrated canvas drawing tool with automatic grayscale resizing and thresholding matching the offline preprocessing pipeline.
- **Automated Deployment**: GitHub Actions pipeline is configured to build, convert, and host the web UI via GitHub Pages.

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
   pip install tensorflowjs
   ```

3. Convert the offline model to TFJS:
   The conversion process is handled in two stages to ensure full compatibility with Keras 3:
   ```bash
   # 1. Export the Keras model to a generic SavedModel (patches deserialization errors)
   python scripts/export_saved_model.py --input Digit_Recognizer.keras --output temp_saved_model

   # 2. Convert the intermediate SavedModel to a TensorFlow.js GraphModel
   mkdir -p docs/model
   tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve temp_saved_model docs/model
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

# 3. Evaluate TF.js (ensure @tensorflow/tfjs depends are installed)
# NOTE: Make sure the model is converted first before running this script
npm install
node scripts/evaluate_tfjs.js --samples scripts/test_samples.json --output scripts/tfjs_results.json --model docs/model/model.json

# 4. Compare inference matrices
python scripts/compare_results.py --keras scripts/keras_results.json --tfjs scripts/tfjs_results.json --report reports/verification_report.json
```

## Results & Acceptance Metrics

- **Model format**: Verified inputs: `28x28`, grayscale, black background, preprocessed inside TF.js exactly matching the original Keras standard mapping of `pixel / 255.0`.
- **Target Accuracy**: ~99.5% on test batches.
- **Parity**: Identical predictions between Python native Keras and TensorFlow.js (Diff <= 0.5% or Match rate >= 98%).

Detailed reports are generated in `reports/verification_report.json`.