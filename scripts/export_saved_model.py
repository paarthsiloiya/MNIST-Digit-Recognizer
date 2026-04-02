import os
import argparse
os.environ['TF_USE_LEGACY_KERAS'] = '0'
import keras
import tensorflow as tf

def patch_dense():
    original_dense_init = keras.layers.Dense.__init__
    def new_dense_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        original_dense_init(self, *args, **kwargs)
    keras.layers.Dense.__init__ = new_dense_init
    
    if hasattr(keras.layers.Dense, 'from_config'):
        original_dense_from_config = keras.layers.Dense.from_config
        @classmethod
        def new_dense_from_config(cls, config):
            config.pop('quantization_config', None)
            return original_dense_from_config(config)
        keras.layers.Dense.from_config = new_dense_from_config

def main():
    patch_dense()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='Digit_Recognizer.keras')
    parser.add_argument('--output', type=str, default='temp_saved_model')
    args = parser.parse_args()

    print(f"Loading model from {args.input}...")
    model = keras.models.load_model(args.input)
    
    print(f"Saving temporary SavedModel to {args.output}...")
    model.export(args.output)
    print("Export successful.")

if __name__ == '__main__':
    main()
