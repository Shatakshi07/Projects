import numpy as np
from numpy import load
import tensorflow as tf

def load_real_data(filename):

    data = np.load(filename)
    X1, X2, X3 = data['arr_0'], data['arr_1'], data['arr_2']

    # normalize from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    X3 = (X3 - 127.5) / 127.5
    return [X1, X2, X3]

def generate_real_data(data, batch_id, batch_size, patch_shape):

    trainA, trainB, trainC = data

    start = batch_id*batch_size
    end = start+batch_size
    X1, X2, X3  = trainA[start:end], trainB[start:end], trainC[start:end]

    y1 = -np.ones((batch_size, patch_shape[0], patch_shape[0], 1))
    y2 = -np.ones((batch_size, patch_shape[1], patch_shape[1], 1))
    return [X1, X2, X3], [y1,y2]

def generate_real_data_random(data, random_samples, patch_shape):

    trainA, trainB, trainC = data

    # Check if dataset is empty
    if trainA.shape[0] == 0:
        print("Dataset is empty!")
        return None, None

    # Limit random_samples to dataset size
    random_samples = min(random_samples, trainA.shape[0])

    id = np.random.randint(0, trainA.shape[0], random_samples)
    X1, X2, X3  = trainA[id], trainB[id], trainC[id]

    y1 = -np.ones((random_samples, patch_shape[0], patch_shape[0], 1))
    y2 = -np.ones((random_samples, patch_shape[1], patch_shape[1], 1))
    
    return [X1, X2, X3], [y1, y2]



def generate_fake_data_fine(g_model, batch_data, batch_mask, x_global, patch_shape):

    X = g_model.predict([batch_data,batch_mask,x_global])
    y1 = np.ones((len(X), patch_shape[0], patch_shape[0], 1))

    return X, y1

def generate_fake_data_coarse(g_model, batch_data, batch_mask, patch_shape):

    X, X_global = g_model.predict([batch_data,batch_mask])
    y1 = np.ones((len(X), patch_shape[1], patch_shape[1], 1))

    return [X,X_global], y1

def resize(X_realA,X_realB,X_realC,out_shape):

    #if X_realA.shape[1:3] != out_shape or X_realB.shape[1:3] != out_shape or X_realC.shape[1:3] != out_shape:
        #raise ValueError("Input tensor shapes must match the output shape dimensions.")
    # Convert input arrays to TensorFlow tensors
    X_realA = tf.convert_to_tensor(X_realA)
    X_realB = tf.convert_to_tensor(X_realB)
    X_realC = tf.convert_to_tensor(X_realC)
    print("Shape of X_realA:", X_realA.shape)
    print("Shape of X_realB:", X_realB.shape)
    print("Shape of X_realC:", X_realC.shape)
    print("Desired output shape:", out_shape)
    #X_realA = np.array(X_realA)

    
    X_realA = tf.image.resize(X_realA, out_shape, method=tf.image.ResizeMethod.LANCZOS3)

    X_realB = tf.image.resize(X_realB, out_shape, method=tf.image.ResizeMethod.LANCZOS3)

    #X_realB = np.array(X_realB)

    X_realC = tf.image.resize(X_realC, out_shape, method=tf.image.ResizeMethod.LANCZOS3)
    #X_realC = np.array(X_realC)

    
    print("Shape of res X_realA:", X_realA.shape)
    print("Shape of res X_realB:", X_realB.shape)
    print("Shape of res X_realC:", X_realC.shape)
    if not isinstance(X_realA, tf.Tensor) or not isinstance(X_realB, tf.Tensor) or not isinstance(X_realC, tf.Tensor):
        raise TypeError("Input tensors must be TensorFlow tensors.")
    if X_realA.shape[1:3] != out_shape or X_realB.shape[1:3] != out_shape or X_realC.shape[1:3] != out_shape:
                  raise ValueError("Resized tensor shapes do not match the desired output shape.")

    #if len(X_realA.shape) != len(out_shape) or len(X_realB.shape) != len(out_shape) or len(X_realC.shape) != len(out_shape):
        #raise ValueError("Input tensor shapes must match the output shape dimensions.")
    return [X_realA,X_realB,X_realC]

def resize2(X_realA, X_realB, X_realC, out_shape):
    X_realA = tf.convert_to_tensor(X_realA, dtype=tf.float32)
    X_realA = tf.image.resize(X_realA, out_shape, method=tf.image.ResizeMethod.LANCZOS3)
    #X_realA = np.array(X_realA)
    
    X_realB = tf.convert_to_tensor(X_realB, dtype=tf.float32)
    X_realB = tf.image.resize(X_realB, out_shape, method=tf.image.ResizeMethod.LANCZOS3)
    #X_realB = np.array(X_realB)

    X_realC = tf.convert_to_tensor(X_realC, dtype=tf.float32)
    X_realC = tf.image.resize(X_realC, out_shape, method=tf.image.ResizeMethod.LANCZOS3)
    #X_realC = np.array(X_realC)
    
    return [X_realA, X_realB, X_realC]
