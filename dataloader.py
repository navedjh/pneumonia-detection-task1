"""Data loading and preprocessing module"""

import numpy as np
import tensorflow as tf
from medmnist import INFO
import medmnist

class PneumoniaDataLoader:
    """Data loader for PneumoniaMNIST"""
    
    def __init__(self):
        self.info = INFO['pneumoniamnist']
        self.DataClass = getattr(medmnist, self.info['python_class'])
        self.class_names = ['Normal', 'Pneumonia']
        
    def load_data(self):
        """Load all data splits"""
        print("\n" + "="*70)
        print("LOADING PNEUMONIAMNIST DATASET")
        print("="*70)
        
        # Load splits
        train_data = self.DataClass(split='train', download=True)
        val_data = self.DataClass(split='val', download=True)
        test_data = self.DataClass(split='test', download=True)
        
        # Extract images and labels
        X_train = train_data.imgs
        y_train = train_data.labels.flatten()
        X_val = val_data.imgs
        y_val = val_data.labels.flatten()
        X_test = test_data.imgs
        y_test = test_data.labels.flatten()
        
        # Normalize to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_val = X_val.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Add channel dimension
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_val = X_val.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        
        # Convert to RGB for Vision Transformer
        X_train_rgb = np.repeat(X_train, 3, axis=-1)
        X_val_rgb = np.repeat(X_val, 3, axis=-1)
        X_test_rgb = np.repeat(X_test, 3, axis=-1)
        
        # Print statistics
        self.print_statistics(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Convert labels to categorical
        y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
        y_val_cat = tf.keras.utils.to_categorical(y_val, 2)
        y_test_cat = tf.keras.utils.to_categorical(y_test, 2)
        
        return {
            'train': (X_train, X_train_rgb, y_train, y_train_cat),
            'val': (X_val, X_val_rgb, y_val, y_val_cat),
            'test': (X_test, X_test_rgb, y_test, y_test_cat)
        }
    
    def print_statistics(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Print dataset statistics"""
        print(f"\nDataset Statistics:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Class distribution - Normal: {np.sum(y_train==0)}, Pneumonia: {np.sum(y_train==1)}")