import tensorflow as tf
from tensorflow.keras import layers, Model

def build_model(vocab_size, max_length):
    inputs = layers.Input(shape=(max_length,))
    x = layers.Embedding(input_dim=vocab_size, output_dim=64)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model