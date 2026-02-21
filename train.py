# ============================================================================
# COMPLETE PIPELINE WITH GRAD-CAM (Run all cells in order)
# ============================================================================

# Cell 1: Install and Import
!pip install medmnist tensorflow scikit-learn matplotlib seaborn pandas scipy opencv-python

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                            precision_recall_curve, f1_score, accuracy_score,
                            precision_score, recall_score, cohen_kappa_score,
                            matthews_corrcoef, roc_auc_score, average_precision_score)
import pandas as pd
import medmnist
from medmnist import INFO
import json
import warnings
from datetime import datetime
import cv2
from tensorflow.keras import models as keras_models
warnings.filterwarnings('ignore')

# Print versions
print(f"\nTensorFlow version: {tf.__version__}")
print(f"MedMNIST version: {medmnist.__version__}")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DATA_FLAG = "pneumoniamnist"
    IMG_SIZE = 28
    NUM_CLASSES = 2
    CLASS_NAMES = ['Normal', 'Pneumonia']

    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.3
    PATIENCE = 5

    BASE_DIR = './pneumonia_detection_journal'
    SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

    SEED = 42

# Create directories
for dir_path in [Config.SAVE_DIR, Config.RESULTS_DIR, Config.FIGURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Set seeds
tf.random.set_seed(Config.SEED)
np.random.seed(Config.SEED)

# ============================================================================
# DATA LOADING
# ============================================================================

class PneumoniaDataLoader:
    """Data loader for PneumoniaMNIST"""

    def __init__(self):
        self.info = INFO[Config.DATA_FLAG]
        self.DataClass = getattr(medmnist, self.info['python_class'])

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
        print(f"\nDataset Statistics:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Class distribution - Normal: {np.sum(y_train==0)}, Pneumonia: {np.sum(y_train==1)}")

        # Convert labels to categorical
        y_train_cat = tf.keras.utils.to_categorical(y_train, Config.NUM_CLASSES)
        y_val_cat = tf.keras.utils.to_categorical(y_val, Config.NUM_CLASSES)
        y_test_cat = tf.keras.utils.to_categorical(y_test, Config.NUM_CLASSES)

        return {
            'train': (X_train, X_train_rgb, y_train, y_train_cat),
            'val': (X_val, X_val_rgb, y_val, y_val_cat),
            'test': (X_test, X_test_rgb, y_test, y_test_cat)
        }, (X_train, X_train_rgb, y_train, y_train_cat), (X_val, X_val_rgb, y_val, y_val_cat), (X_test, X_test_rgb, y_test, y_test_cat)

# ============================================================================
# DATASET VISUALIZATION
# ============================================================================

def visualize_dataset(data):
    """Visualize dataset samples and distribution"""

    X_train, _, y_train, _ = data['train']
    _, _, y_val, _ = data['val']
    _, _, y_test, _ = data['test']

    # 1. Sample images (10 from each class)
    fig, axes = plt.subplots(2, 10, figsize=(20, 5))

    normal_indices = np.where(y_train == 0)[0][:10]
    pneumonia_indices = np.where(y_train == 1)[0][:10]

    for i, idx in enumerate(normal_indices):
        axes[0, i].imshow(X_train[idx].squeeze(), cmap='gray')
        axes[0, i].set_title(f'Normal #{i+1}', fontsize=8)
        axes[0, i].axis('off')

    for i, idx in enumerate(pneumonia_indices):
        axes[1, i].imshow(X_train[idx].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Pneumonia #{i+1}', fontsize=8, color='red')
        axes[1, i].axis('off')

    plt.suptitle('PneumoniaMNIST Dataset - 10 Sample Images from Each Class',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'dataset_samples.png'), dpi=150)
    plt.show()

    # 2. Class distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    splits = ['Training', 'Validation', 'Test']
    totals = [len(y_train), len(y_val), len(y_test)]
    normals = [np.sum(y_train == 0), np.sum(y_val == 0), np.sum(y_test == 0)]
    pneumonias = [np.sum(y_train == 1), np.sum(y_val == 1), np.sum(y_test == 1)]

    for i, (split, total, normal, pneumonia) in enumerate(zip(splits, totals, normals, pneumonias)):
        axes[i].bar(['Normal', 'Pneumonia'], [normal, pneumonia],
                   color=['#3498db', '#e74c3c'])
        axes[i].set_title(f'{split} Set\nTotal: {total} images', fontweight='bold')
        axes[i].set_ylabel('Number of Images')

        # Add value labels
        for j, count in enumerate([normal, pneumonia]):
            axes[i].text(j, count + 5, str(count), ha='center', va='bottom')

    plt.suptitle('Class Distribution Across Splits', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'class_distribution.png'), dpi=150)
    plt.show()

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

def create_cnn_model():
    """Create CNN model"""
    inputs = layers.Input(shape=(28, 28, 1))

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    return models.Model(inputs, outputs)

def create_vit_model():
    """Create Vision Transformer model"""
    inputs = layers.Input(shape=(28, 28, 3))

    # Patch embedding
    x = layers.Conv2D(64, 7, strides=4, padding='same', activation='relu')(inputs)
    x = layers.Reshape((-1, 64))(x)

    # Position embedding
    positions = tf.range(start=0, limit=49, delta=1)
    pos_embed = layers.Embedding(49, 64)(positions)
    x = x + pos_embed

    # Transformer block
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    return models.Model(inputs, outputs)

# ============================================================================
# TRAINER CLASS
# ============================================================================

class ModelTrainer:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.history = None

    def train(self, X_train, y_train, X_val, y_val):
        """Train the model"""

        # Data augmentation
        aug = tf.keras.Sequential([
            layers.RandomRotation(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
        ])

        # Compile
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
        )

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=Config.PATIENCE,
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            ModelCheckpoint(
                os.path.join(Config.SAVE_DIR, f'{self.model_name}_best.weights.h5'),
                monitor='val_accuracy', save_best_only=True,
                save_weights_only=True, verbose=1
            )
        ]

        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(1000).batch(Config.BATCH_SIZE).map(
            lambda x, y: (aug(x, training=True), y)
        ).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
                                 .batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Train
        self.history = self.model.fit(
            train_ds, validation_data=val_ds,
            epochs=Config.EPOCHS, callbacks=callbacks, verbose=1
        )

        return self.history

# ============================================================================
# ENSEMBLE
# ============================================================================

class Ensemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X_cnn, X_vit):
        """Average predictions from both models"""
        cnn_pred = self.models[0].predict(X_cnn, verbose=0)
        vit_pred = self.models[1].predict(X_vit, verbose=0)
        return (cnn_pred + vit_pred) / 2

# ============================================================================
# METRICS
# ============================================================================

class MetricsCalculator:
    def calculate_all(self, y_true, y_pred, y_probs):
        """Calculate all metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': auc(*roc_curve(y_true, y_probs)[:2]),
            'pr_auc': auc(*precision_recall_curve(y_true, y_probs)[1::-1]),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

    def print_table(self, metrics, name):
        """Print metrics table"""
        print(f"\n{'='*60}")
        print(f"{name} RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR AUC:    {metrics['pr_auc']:.4f}")

# ============================================================================
# GRAD-CAM VISUALIZATION
# ============================================================================

def minimal_gradcam(model, image, layer_name=None):
    """Absolute minimal Grad-CAM implementation"""

    # Find conv layer
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                layer_name = layer.name
                break

    if layer_name is None:
        raise ValueError("No convolutional layer found")

    # Create gradient model
    grad_model = keras_models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    # Prepare image
    if len(image.shape) == 3:
        img_array = np.expand_dims(image, axis=0)
    else:
        img_array = image

    # Get prediction
    preds = model.predict(img_array, verbose=0)[0]
    class_idx = np.argmax(preds)

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    # Get gradients and compute heatmap
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-7)

    # Resize
    heatmap = tf.image.resize(cam[..., tf.newaxis],
                              (image.shape[0], image.shape[1])).numpy().squeeze()

    return heatmap, class_idx, preds[class_idx]

def show_gradcam_samples(model, model_name, X_test, y_test, num_samples=4, save_dir=None):
    """Show Grad-CAM for sample images"""

    print(f"\n📊 Generating Grad-CAM for {model_name}...")

    # Select samples (mix of normal and pneumonia)
    normal_idx = np.where(y_test == 0)[0][:2]
    pneumonia_idx = np.where(y_test == 1)[0][:2]
    sample_indices = np.concatenate([normal_idx, pneumonia_idx])

    fig, axes = plt.subplots(2, len(sample_indices), figsize=(16, 8))

    for i, idx in enumerate(sample_indices):
        img = X_test[idx]

        # Generate Grad-CAM
        heatmap, pred_class, conf = minimal_gradcam(model, img)

        # Original image
        axes[0, i].imshow(img.squeeze(), cmap='gray')
        true_label = "Normal" if y_test[idx] == 0 else "Pneumonia"
        axes[0, i].set_title(f'True: {true_label}', fontsize=10)
        axes[0, i].axis('off')

        # Overlay
        axes[1, i].imshow(img.squeeze(), cmap='gray')
        axes[1, i].imshow(heatmap, cmap='jet', alpha=0.5)
        pred_label = Config.CLASS_NAMES[pred_class]
        color = 'green' if pred_class == y_test[idx] else 'red'
        axes[1, i].set_title(f'Pred: {pred_label}\nConf: {conf:.2f}',
                            fontsize=9, color=color)
        axes[1, i].axis('off')

    plt.suptitle(f'{model_name} - Grad-CAM Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, f'gradcam_{model_name.lower()}.png'),
                   dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ Grad-CAM for {model_name} complete")

# ============================================================================
# VISUALIZATION FUNCTIONS (Confusion Matrix, ROC, Error Analysis)
# ============================================================================

def plot_confusion_matrix(cm, model_name, save_dir=None):
    """
    Plot enhanced confusion matrix with percentages
    """
    plt.figure(figsize=(8, 6))

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.CLASS_NAMES,
                yticklabels=Config.CLASS_NAMES,
                annot_kws={'size': 14})

    # Add percentage labels
    for i in range(2):
        for j in range(2):
            plt.gca().text(j+0.5, i+0.7, f'({cm_percent[i,j]:.1f}%)',
                          ha='center', va='center', fontsize=10)

    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{model_name.lower()}_confusion.png'),
                   dpi=150, bbox_inches='tight')
    plt.show()

    # Print analysis
    print(f"\n📊 {model_name} Confusion Matrix Analysis:")
    print(f"   True Negatives (Normal correct): {cm[0,0]}")
    print(f"   False Positives (Normal→Pneumonia): {cm[0,1]}")
    print(f"   False Negatives (Pneumonia→Normal): {cm[1,0]}")
    print(f"   True Positives (Pneumonia correct): {cm[1,1]}")
    print(f"   Accuracy: {(cm[0,0] + cm[1,1]) / np.sum(cm):.2%}")

def plot_roc_curves_comparison(results_dict, save_dir=None):
    """
    Plot ROC curves for all models comparison
    """
    plt.figure(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (name, data) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_probs'])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC = {roc_auc:.4f})')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, 'roc_curves_comparison.png'),
                   dpi=150, bbox_inches='tight')
    plt.show()

def plot_pr_curves_comparison(results_dict, save_dir=None):
    """
    Plot Precision-Recall curves for all models comparison
    """
    plt.figure(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (name, data) in enumerate(results_dict.items()):
        precision, recall, _ = precision_recall_curve(data['y_true'], data['y_probs'])
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AP = {pr_auc:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, 'pr_curves_comparison.png'),
                   dpi=150, bbox_inches='tight')
    plt.show()

def plot_error_analysis(y_true, y_pred, y_probs, X_test, model_name, save_dir=None, num_samples=5):
    """
    Comprehensive error analysis with visualization of misclassified samples
    """
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS - {model_name}")
    print(f"{'='*60}")

    # Find misclassified indices
    misclassified = np.where(y_true != y_pred)[0]
    false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]

    total_errors = len(misclassified)
    fp_count = len(false_positives)
    fn_count = len(false_negatives)

    print(f"\n📊 Error Statistics:")
    print(f"   Total errors: {total_errors} / {len(y_true)} ({total_errors/len(y_true)*100:.2f}%)")
    print(f"   False Positives (Normal→Pneumonia): {fp_count} ({fp_count/total_errors*100:.1f}% of errors)" if total_errors > 0 else "   False Positives: 0")
    print(f"   False Negatives (Pneumonia→Normal): {fn_count} ({fn_count/total_errors*100:.1f}% of errors)" if total_errors > 0 else "   False Negatives: 0")

    if fp_count > 0:
        fp_conf = np.mean(y_probs[false_positives])
        print(f"   Avg confidence for False Positives: {fp_conf:.3f}")

    if fn_count > 0:
        fn_conf = np.mean(1 - y_probs[false_negatives])
        print(f"   Avg confidence for False Negatives: {fn_conf:.3f}")

    # Visualize false positives
    if fp_count > 0:
        fig, axes = plt.subplots(2, min(5, fp_count), figsize=(15, 6))
        if fp_count == 1:
            axes = axes.reshape(2, 1)

        samples = false_positives[:min(5, fp_count)]

        for i, idx in enumerate(samples):
            # Original image
            img = X_test[idx].squeeze() if len(X_test[idx].shape) > 2 else X_test[idx]
            if len(img.shape) == 3 and img.shape[-1] == 3:
                img = img[:, :, 0]  # Take first channel for RGB

            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'FP #{i+1}\nTrue: Normal', fontsize=10)
            axes[0, i].axis('off')

            # With confidence
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].set_title(f'Pred: Pneumonia\nConf: {y_probs[idx]:.2f}',
                                 fontsize=10, color='red')
            axes[1, i].axis('off')

        plt.suptitle(f'{model_name} - False Positives (Normal misclassified as Pneumonia)',
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'{model_name.lower()}_false_positives.png'),
                       dpi=150, bbox_inches='tight')
        plt.show()

    # Visualize false negatives
    if fn_count > 0:
        fig, axes = plt.subplots(2, min(5, fn_count), figsize=(15, 6))
        if fn_count == 1:
            axes = axes.reshape(2, 1)

        samples = false_negatives[:min(5, fn_count)]

        for i, idx in enumerate(samples):
            # Original image
            img = X_test[idx].squeeze() if len(X_test[idx].shape) > 2 else X_test[idx]
            if len(img.shape) == 3 and img.shape[-1] == 3:
                img = img[:, :, 0]  # Take first channel for RGB

            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'FN #{i+1}\nTrue: Pneumonia', fontsize=10, color='red')
            axes[0, i].axis('off')

            # With confidence
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].set_title(f'Pred: Normal\nConf: {1-y_probs[idx]:.2f}',
                                 fontsize=10, color='red')
            axes[1, i].axis('off')

        plt.suptitle(f'{model_name} - False Negatives (Pneumonia misclassified as Normal)',
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'{model_name.lower()}_false_negatives.png'),
                       dpi=150, bbox_inches='tight')
        plt.show()

    return {
        'total_errors': total_errors,
        'false_positives': fp_count,
        'false_negatives': fn_count,
        'fp_confidence': float(np.mean(y_probs[false_positives])) if fp_count > 0 else 0,
        'fn_confidence': float(np.mean(1 - y_probs[false_negatives])) if fn_count > 0 else 0
    }

def plot_comparison_table(results_dict, save_dir=None):
    """
    Create a comprehensive comparison table with all metrics
    """
    # Create DataFrame with all metrics
    comparison_data = []

    for name, data in results_dict.items():
        row = {
            'Model': name,
            'Accuracy': f"{data['accuracy']:.2%}",
            'Precision': f"{data['precision']:.2%}",
            'Recall': f"{data['recall']:.2%}",
            'F1-Score': f"{data['f1_score']:.2%}",
            'ROC AUC': f"{data['roc_auc']:.2%}",
            'PR AUC': f"{data['pr_auc']:.2%}"
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    df.set_index('Model', inplace=True)

    # Display as styled table
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON TABLE")
    print("="*80)
    print("\n", df)

    # Create a styled figure
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15] * len(df.columns))

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color code the best values
    for i, col in enumerate(df.columns):
        # Convert percentage strings back to float for comparison
        values = [float(v.strip('%'))/100 for v in df[col].values]
        best_idx = np.argmax(values)

        # Highlight best cell
        cell = table[(best_idx + 1, i)]
        cell.set_facecolor('#90EE90')  # Light green

    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, 'comparison_table.png'),
                   dpi=150, bbox_inches='tight')
    plt.show()

    # Save CSV
    csv_path = os.path.join(save_dir, 'comparison_table.csv') if save_dir else 'comparison_table.csv'
    df.to_csv(csv_path)
    print(f"\n✅ Comparison table saved to {csv_path}")

    return df

def plot_all_visualizations(model_names, y_trues, y_preds, y_probss, X_tests,
                           results_dict, save_dir=None):
    """
    Generate all visualizations in one go
    """
    print("\n" + "="*80)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*80)

    # 1. Confusion matrices for each model
    for i, name in enumerate(model_names):
        cm = confusion_matrix(y_trues[i], y_preds[i])
        plot_confusion_matrix(cm, name, save_dir)

    # 2. ROC curves comparison
    roc_data = {name: {'y_true': y_trues[i], 'y_probs': y_probss[i]}
                for i, name in enumerate(model_names)}
    plot_roc_curves_comparison(roc_data, save_dir)

    # 3. PR curves comparison
    plot_pr_curves_comparison(roc_data, save_dir)

    # 4. Error analysis for each model
    for i, name in enumerate(model_names):
        plot_error_analysis(y_trues[i], y_preds[i], y_probss[i],
                           X_tests[i], name, save_dir)

    # 5. Comparison table
    plot_comparison_table(results_dict, save_dir)

    print(f"\n✅ All visualizations saved to {save_dir}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("PNEUMONIA DETECTION SYSTEM WITH GRAD-CAM")
    print("="*70)

    # Load data
    loader = PneumoniaDataLoader()
    data, train_data, val_data, test_data = loader.load_data()

    # Unpack test data for later use
    X_test, X_test_rgb, y_test, y_test_cat = test_data
    X_train, X_train_rgb, y_train, y_train_cat = train_data
    X_val, X_val_rgb, y_val, y_val_cat = val_data

    # Visualize dataset
    visualize_dataset(data)

    # Initialize
    calculator = MetricsCalculator()
    models = []
    results = {}

    # 1. Train CNN
    print("\n" + "#"*70)
    print("Training CNN Model")
    print("#"*70)

    cnn_model = create_cnn_model()
    trainer = ModelTrainer(cnn_model, 'cnn')
    trainer.train(X_train, y_train_cat, X_val, y_val_cat)

    y_pred_probs_cnn = cnn_model.predict(X_test, verbose=0)
    y_pred_cnn = np.argmax(y_pred_probs_cnn, axis=1)
    metrics_cnn = calculator.calculate_all(y_test, y_pred_cnn, y_pred_probs_cnn[:, 1])
    calculator.print_table(metrics_cnn, "CNN")

    results['CNN'] = metrics_cnn
    models.append(cnn_model)

    # 2. Train ViT
    print("\n" + "#"*70)
    print("Training Vision Transformer")
    print("#"*70)

    vit_model = create_vit_model()
    trainer = ModelTrainer(vit_model, 'vit')
    trainer.train(X_train_rgb, y_train_cat, X_val_rgb, y_val_cat)

    y_pred_probs_vit = vit_model.predict(X_test_rgb, verbose=0)
    y_pred_vit = np.argmax(y_pred_probs_vit, axis=1)
    metrics_vit = calculator.calculate_all(y_test, y_pred_vit, y_pred_probs_vit[:, 1])
    calculator.print_table(metrics_vit, "Vision Transformer")

    results['ViT'] = metrics_vit
    models.append(vit_model)

    # ========================================================================
    # GRAD-CAM VISUALIZATION
    # ========================================================================

    print("\n" + "="*70)
    print("GRAD-CAM INTERPRETABILITY ANALYSIS")
    print("="*70)

    # Grad-CAM for CNN
    show_gradcam_samples(
        cnn_model, "CNN",
        X_test, y_test,
        num_samples=4,
        save_dir=Config.FIGURES_DIR
    )

    # Grad-CAM for ViT
    show_gradcam_samples(
        vit_model, "ViT",
        X_test_rgb, y_test,
        num_samples=4,
        save_dir=Config.FIGURES_DIR
    )

    # 3. Ensemble
    print("\n" + "#"*70)
    print("Ensemble Prediction")
    print("#"*70)

    ensemble = Ensemble(models)
    ensemble_probs = ensemble.predict(X_test, X_test_rgb)
    y_pred_ensemble = np.argmax(ensemble_probs, axis=1)
    metrics_ensemble = calculator.calculate_all(y_test, y_pred_ensemble, ensemble_probs[:, 1])
    calculator.print_table(metrics_ensemble, "Ensemble")

    results['Ensemble'] = metrics_ensemble

    # ========================================================================
    # COMPREHENSIVE VISUALIZATIONS
    # ========================================================================

    # Collect data for visualizations
    model_names = ['CNN', 'ViT', 'Ensemble']
    y_trues = [
        y_test,                          # CNN
        y_test,                          # ViT
        y_test                           # Ensemble
    ]
    y_preds = [
        y_pred_cnn,                       # CNN predictions
        y_pred_vit,                       # ViT predictions
        y_pred_ensemble                    # Ensemble predictions
    ]
    y_probss = [
        y_pred_probs_cnn[:, 1],           # CNN probabilities
        y_pred_probs_vit[:, 1],           # ViT probabilities
        ensemble_probs[:, 1]               # Ensemble probabilities
    ]
    X_tests = [
        X_test,                           # CNN test images (grayscale)
        X_test_rgb,                        # ViT test images (RGB)
        X_test                              # Ensemble test images (use grayscale for display)
    ]

    # Generate all visualizations
    plot_all_visualizations(
        model_names=model_names,
        y_trues=y_trues,
        y_preds=y_preds,
        y_probss=y_probss,
        X_tests=X_tests,
        results_dict=results,
        save_dir=Config.FIGURES_DIR
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    summary = pd.DataFrame({
        name: {
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1-Score': r['f1_score'],
            'ROC AUC': r['roc_auc']
        } for name, r in results.items()
    }).T
    print("\n", summary.round(4))

    # Save results
    summary.to_csv(os.path.join(Config.RESULTS_DIR, 'results.csv'))

    # Save all results as JSON
    final_results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config': {
            'batch_size': Config.BATCH_SIZE,
            'epochs': Config.EPOCHS,
            'learning_rate': Config.LEARNING_RATE
        },
        'results': {name: {
            'accuracy': float(r['accuracy']),
            'precision': float(r['precision']),
            'recall': float(r['recall']),
            'f1_score': float(r['f1_score']),
            'roc_auc': float(r['roc_auc']),
            'pr_auc': float(r['pr_auc'])
        } for name, r in results.items()}
    }

    with open(os.path.join(Config.RESULTS_DIR, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"\n✅ Results saved to {Config.RESULTS_DIR}")
    print(f"✅ Grad-CAM visualizations saved to {Config.FIGURES_DIR}")
    print(f"✅ All visualizations saved to {Config.FIGURES_DIR}")
    print(f"\n{'='*70}")
    print("🎉 ALL COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    main()