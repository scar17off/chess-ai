import pickle
import numpy as np
import tensorflow as tf

def convert_model_to_binary(model_path, output_path):
    # Load the TensorFlow model
    chess_nn = pickle.load(open(model_path, 'rb'))
    model = chess_nn.model
    
    with open(output_path, 'wb') as f:
        # write number of dense layers
        dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        np.array([len(dense_layers)], dtype=np.int32).tofile(f)
        
        for layer in dense_layers:
            weights, biases = layer.get_weights()
            
            # convert to float32 for c++ compatibility
            weights = weights.astype(np.float32) # no transpose needed (weights are already in the correct shape)
            biases = biases.astype(np.float32)
            
            # write input and output dimensions
            input_dim = weights.shape[0] # number of rows (input dimension)
            output_dim = weights.shape[1] # number of columns (output dimension)
            
            np.array([input_dim, output_dim], dtype=np.int32).tofile(f)
            np.array([output_dim], dtype=np.int32).tofile(f)  # bias shape
            
            weights.tofile(f)
            biases.tofile(f)
            
        print(f"Model converted successfully!")
        print("\nLayer dimensions:")
        for i, layer in enumerate(dense_layers):
            weights, biases = layer.get_weights()
            print(f"Layer {i}:")
            print(f"  Input size: {weights.shape[0]}")
            print(f"  Output size: {weights.shape[1]}")
            print(f"  Weights shape: {weights.shape}")
            print(f"  Biases shape: {biases.shape}")

if __name__ == "__main__":
    convert_model_to_binary("./models/default.pkl", "default.bin")