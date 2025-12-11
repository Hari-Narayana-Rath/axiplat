"""
Age Estimation Model Visualization Script
Creates comprehensive matplotlib visualizations for the age gate application results.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import cv2

# Try to import models (same as app.py)
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Configuration (matching app.py)
MODEL_DIR = os.path.join(os.getcwd(), "model")
AGE_PROTO = os.path.join(MODEL_DIR, "deploy_age.prototxt")
AGE_CAFFE = os.path.join(MODEL_DIR, "age_net.caffemodel")
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
AGE_MIDPOINTS = [(int(a.split('(')[1].split('-')[0]) + int(a.split('-')[1].split(')')[0])) / 2.0 
                 for a in AGE_LIST]

DNN_WEIGHT = 0.6
CNN_WEIGHT = 0.4
DNN_CONFIDENCE_LOCK = 0.55
CALIBRATION_SCALE = 0.9
CALIBRATION_OFFSET = -2.5
EMA_ALPHA = 0.3
SAMPLE_INTERVAL = 0.5
MAX_SECONDS = 10

# Model loading
age_net = None
USE_DNN = False
if os.path.isfile(AGE_PROTO) and os.path.isfile(AGE_CAFFE):
    try:
        age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_CAFFE)
        USE_DNN = True
        print("✓ Loaded OpenCV DNN age model.")
    except Exception as e:
        print(f"✗ Failed to load DNN age model: {e}")

age_cnn_model = None
if TF_AVAILABLE:
    cnn_path = os.path.join(MODEL_DIR, "cnn_age_model.h5")
    if os.path.isfile(cnn_path):
        try:
            age_cnn_model = load_model(cnn_path)
            print("✓ Loaded CNN regression model.")
        except Exception as e:
            print(f"✗ Failed to load CNN: {e}")
    else:
        print("ℹ No CNN model found.")
else:
    print("ℹ TensorFlow not available; CNN regression disabled.")

# Helper functions (simplified from app.py)
def adaptive_blob_mean(face_img):
    DEFAULT_BLOB_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
    DYNAMIC_MEAN_BLEND = 0.35
    if face_img is None or DYNAMIC_MEAN_BLEND <= 0:
        return DEFAULT_BLOB_MEAN
    b_mean, g_mean, r_mean, _ = cv2.mean(face_img)
    blended = []
    for default_val, current_val in zip(DEFAULT_BLOB_MEAN, (b_mean, g_mean, r_mean)):
        blended.append((1.0 - DYNAMIC_MEAN_BLEND) * default_val + DYNAMIC_MEAN_BLEND * current_val)
    return tuple(blended)

def brightness_age_adjust(face_img):
    BRIGHTNESS_REFERENCE = 132.0
    BRIGHTNESS_CORRECTION = 0.018
    BRIGHTNESS_CORRECTION_CLAMP = 6.0
    if face_img is None or BRIGHTNESS_CORRECTION == 0:
        return 0.0
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    delta = BRIGHTNESS_REFERENCE - float(np.mean(gray))
    adjustment = delta * BRIGHTNESS_CORRECTION
    return float(np.clip(adjustment, -BRIGHTNESS_CORRECTION_CLAMP, BRIGHTNESS_CORRECTION_CLAMP))

def get_dnn_predictions(face_img):
    """Get DNN model predictions with probabilities"""
    if age_net is None:
        return None, None, None
    try:
        mean_vals = adaptive_blob_mean(face_img)
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), mean_vals, swapRB=False)
        age_net.setInput(blob)
        preds = age_net.forward()[0]
        
        e = np.exp(preds - np.max(preds))
        probs = e / e.sum()
        
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        discrete_age = AGE_MIDPOINTS[top_idx]
        expected = float(np.dot(probs, AGE_MIDPOINTS))
        
        raw_age = discrete_age if top_prob >= DNN_CONFIDENCE_LOCK else expected
        calibrated = CALIBRATION_SCALE * raw_age + CALIBRATION_OFFSET
        calibrated += brightness_age_adjust(face_img)
        
        return float(calibrated), probs, expected
    except:
        return None, None, None

def get_cnn_prediction(face_img):
    """Get CNN model prediction"""
    if age_cnn_model is None:
        return None
    try:
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        arr = img.astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        pred = age_cnn_model.predict(arr, verbose=0)
        return float(pred[0][0])
    except:
        return None

def ema_update(prev, val, alpha=EMA_ALPHA):
    if prev is None:
        return val
    return prev * (1 - alpha) + val * alpha

def simulate_age_predictions(num_samples=20, base_age=25, noise_std=2.0):
    """Simulate age predictions over time for demonstration"""
    times = np.arange(0, num_samples * SAMPLE_INTERVAL, SAMPLE_INTERVAL)
    
    # Simulate DNN and CNN predictions with some variation
    dnn_ages = base_age + np.random.normal(0, noise_std, num_samples)
    cnn_ages = base_age + np.random.normal(0, noise_std * 1.2, num_samples)
    
    # Combined weighted average
    combined_ages = DNN_WEIGHT * dnn_ages + CNN_WEIGHT * cnn_ages
    
    # Calculate EMA
    ema_ages = []
    ema_val = None
    for age in combined_ages:
        ema_val = ema_update(ema_val, age)
        ema_ages.append(ema_val)
    
    # Simulate DNN probabilities
    dnn_probs_list = []
    for i in range(num_samples):
        # Create realistic probability distribution
        probs = np.random.dirichlet([1, 1, 1, 2, 3, 2, 1, 1])  # Favor middle ages
        dnn_probs_list.append(probs)
    
    return {
        'times': times,
        'dnn_ages': dnn_ages,
        'cnn_ages': cnn_ages,
        'combined_ages': combined_ages,
        'ema_ages': np.array(ema_ages),
        'dnn_probs': np.array(dnn_probs_list)
    }

def create_visualizations(data=None, save_path='age_model_results.png'):
    """Create comprehensive visualization dashboard"""
    
    # Generate simulated data if not provided
    if data is None:
        print("Generating simulated age prediction data...")
        data = simulate_age_predictions(num_samples=20, base_age=27, noise_std=2.5)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Style (try seaborn style, fallback to default if not available)
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    
    # 1. Age Predictions Over Time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(data['times'], data['dnn_ages'], 'o-', label=f'DNN Model (weight={DNN_WEIGHT})', 
             alpha=0.7, linewidth=2, markersize=6)
    if age_cnn_model is not None:
        ax1.plot(data['times'], data['cnn_ages'], 's-', label=f'CNN Model (weight={CNN_WEIGHT})', 
                 alpha=0.7, linewidth=2, markersize=6)
    ax1.plot(data['times'], data['combined_ages'], '^-', label='Weighted Combined', 
             color='purple', linewidth=2.5, markersize=7, alpha=0.8)
    ax1.plot(data['times'], data['ema_ages'], '--', label='EMA Smoothed', 
             color='red', linewidth=3, alpha=0.9)
    
    # Add final median line
    final_median = np.median(data['combined_ages'])
    ax1.axhline(y=final_median, color='green', linestyle=':', linewidth=2, 
                label=f'Final Median: {final_median:.1f} years')
    
    # Add access threshold line
    access_threshold = 14
    ax1.axhline(y=access_threshold, color='orange', linestyle='--', linewidth=2, 
                label=f'Access Threshold: {access_threshold} years', alpha=0.7)
    ax1.fill_between(data['times'], 0, access_threshold, alpha=0.1, color='red', 
                     label='Access Denied Zone')
    ax1.fill_between(data['times'], access_threshold, 100, alpha=0.1, color='green', 
                     label='Access Granted Zone')
    
    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Age (years)', fontsize=12, fontweight='bold')
    ax1.set_title('Age Predictions Over Time', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 60])
    
    # 2. Model Comparison Scatter
    ax2 = fig.add_subplot(gs[1, 0])
    if age_cnn_model is not None:
        ax2.scatter(data['dnn_ages'], data['cnn_ages'], alpha=0.6, s=60, c=data['times'], 
                   cmap='viridis', edgecolors='black', linewidths=0.5)
        min_age = min(np.min(data['dnn_ages']), np.min(data['cnn_ages']))
        max_age = max(np.max(data['dnn_ages']), np.max(data['cnn_ages']))
        ax2.plot([min_age, max_age], [min_age, max_age], 'r--', alpha=0.5, linewidth=2, 
                label='Perfect Agreement')
        ax2.set_xlabel('DNN Age (years)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('CNN Age (years)', fontsize=11, fontweight='bold')
        ax2.set_title('Model Agreement', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Time (s)', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'CNN Model\nNot Available', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        ax2.set_title('Model Agreement', fontsize=12, fontweight='bold')
    
    # 3. Age Distribution Histogram
    ax3 = fig.add_subplot(gs[1, 1])
    combined_ages = data['combined_ages']
    mean_age = np.mean(combined_ages)
    ax3.hist(combined_ages, bins=15, alpha=0.7, color='steelblue', 
             edgecolor='black', linewidth=1.2)
    ax3.axvline(final_median, color='red', linestyle='--', linewidth=2, 
                label=f'Median: {final_median:.1f}')
    ax3.axvline(mean_age, color='green', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_age:.1f}')
    ax3.axvline(access_threshold, color='orange', linestyle=':', linewidth=2, 
                label=f'Threshold: {access_threshold}')
    ax3.set_xlabel('Age (years)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Age Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. DNN Age Group Probabilities (heatmap)
    ax4 = fig.add_subplot(gs[1, 2])
    if len(data['dnn_probs']) > 0:
        # Average probabilities across all samples
        avg_probs = np.mean(data['dnn_probs'], axis=0)
        colors = plt.cm.RdYlGn(avg_probs / np.max(avg_probs))
        bars = ax4.barh(range(len(AGE_LIST)), avg_probs, color=colors, edgecolor='black', linewidth=1)
        ax4.set_yticks(range(len(AGE_LIST)))
        ax4.set_yticklabels(AGE_LIST, fontsize=9)
        ax4.set_xlabel('Average Probability', fontsize=11, fontweight='bold')
        ax4.set_title('DNN Age Group Probabilities', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, avg_probs)):
            ax4.text(prob + 0.01, i, f'{prob:.3f}', va='center', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No DNN\nProbability Data', 
                ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.set_title('DNN Age Group Probabilities', fontsize=12, fontweight='bold')
    
    # 5. EMA Smoothing Effect
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(data['times'], data['combined_ages'], 'o-', label='Raw Combined', 
             alpha=0.5, linewidth=1.5, markersize=5)
    ax5.plot(data['times'], data['ema_ages'], 's-', label=f'EMA (α={EMA_ALPHA})', 
             color='red', linewidth=2.5, markersize=6, alpha=0.8)
    ax5.fill_between(data['times'], data['combined_ages'], data['ema_ages'], 
                     alpha=0.2, color='red', label='Smoothing Effect')
    ax5.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Age (years)', fontsize=11, fontweight='bold')
    ax5.set_title('EMA Smoothing Visualization', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistical Summary
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    # Pre-compute statistics to avoid f-string issues
    combined = data['combined_ages']
    mean_age_val = np.mean(combined)
    median_age_val = np.median(combined)
    std_age_val = np.std(combined)
    min_age_val = np.min(combined)
    max_age_val = np.max(combined)
    range_age_val = max_age_val - min_age_val
    age_group_idx = np.argmin([abs(final_median - mp) for mp in AGE_MIDPOINTS])
    age_group_name = AGE_LIST[age_group_idx]
    access_status_text = '✓ GRANTED' if final_median >= access_threshold else '✗ DENIED'
    
    stats_text = f"""
    MODEL PERFORMANCE STATISTICS
    {'=' * 35}
    
    Combined Age Predictions:
    • Mean: {mean_age_val:.2f} years
    • Median: {median_age_val:.2f} years
    • Std Dev: {std_age_val:.2f} years
    • Min: {min_age_val:.2f} years
    • Max: {max_age_val:.2f} years
    • Range: {range_age_val:.2f} years
    
    Final Decision:
    • Final Age: {final_median:.2f} years
    • Age Group: {age_group_name}
    • Access: {access_status_text}
    
    Model Weights:
    • DNN Weight: {DNN_WEIGHT}
    • CNN Weight: {CNN_WEIGHT}
    
    Configuration:
    • Sample Interval: {SAMPLE_INTERVAL}s
    • EMA Alpha: {EMA_ALPHA}
    • Access Threshold: {access_threshold} years
    """
    
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 7. Access Decision Visualization
    ax7 = fig.add_subplot(gs[2, 2])
    access_status = final_median >= access_threshold
    colors_pie = ['#ff4444', '#44ff44']
    labels_pie = ['Access Denied', 'Access Granted']
    sizes = [1 - int(access_status), int(access_status)]
    
    if access_status:
        explode = (0, 0.1)
    else:
        explode = (0.1, 0)
    
    wedges, texts, autotexts = ax7.pie(sizes, explode=explode, labels=labels_pie, 
                                       colors=colors_pie, autopct='%1.0f%%',
                                       shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax7.set_title(f'Access Decision\n(Age: {final_median:.1f} years)', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Main title
    fig.suptitle('Age Estimation Model - Comprehensive Results Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved to: {save_path}")
    
    return fig

def visualize_from_image(image_path=None, save_path='age_model_single_result.png'):
    """Visualize age estimation results from a single image"""
    if image_path is None or not os.path.isfile(image_path):
        print(f"ℹ No valid image provided. Using simulated data.")
        return create_visualizations(save_path=save_path)
    
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("✗ Failed to load image. Using simulated data.")
        return create_visualizations(save_path=save_path)
    
    # Try to detect face using MediaPipe
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            # Crop face region (simplified)
            h, w, _ = img.shape
            landmarks = results.multi_face_landmarks[0]
            xs = [int(lm.x * w) for lm in landmarks.landmark]
            ys = [int(lm.y * h) for lm in landmarks.landmark]
            x_min, x_max = max(0, min(xs)), min(w-1, max(xs))
            y_min, y_max = max(0, min(ys)), min(h-1, max(ys))
            margin = int(min((x_max-x_min)*0.2, (y_max-y_min)*0.2))
            face = img[max(0,y_min-margin):min(h-1,y_max+margin), 
                      max(0,x_min-margin):min(w-1,x_max+margin)]
        else:
            # Use whole image if no face detected
            face = img
    except:
        face = img
    
    # Get predictions
    dnn_age, dnn_probs, dnn_expected = get_dnn_predictions(face)
    cnn_age = get_cnn_prediction(face)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Single Image Age Estimation Results', fontsize=16, fontweight='bold')
    
    # Show original image
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Show face crop
    if face is not None and face.size > 0:
        axes[0, 1].imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Detected Face Region', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
    
    # Age predictions comparison
    models = []
    ages = []
    if dnn_age is not None:
        models.append('DNN')
        ages.append(dnn_age)
    if cnn_age is not None:
        models.append('CNN')
        ages.append(cnn_age)
    if dnn_age is not None and cnn_age is not None:
        combined = DNN_WEIGHT * dnn_age + CNN_WEIGHT * cnn_age
        models.append('Combined')
        ages.append(combined)
    
    if models:
        axes[1, 0].bar(models, ages, color=['steelblue', 'orange', 'purple'][:len(models)], 
                      edgecolor='black', linewidth=2, alpha=0.7)
        axes[1, 0].axhline(y=14, color='red', linestyle='--', linewidth=2, 
                          label='Access Threshold')
        axes[1, 0].set_ylabel('Predicted Age (years)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Model Predictions', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        for i, (model, age) in enumerate(zip(models, ages)):
            axes[1, 0].text(i, age + 1, f'{age:.1f}', ha='center', fontweight='bold')
    
    # DNN probability distribution
    if dnn_probs is not None:
        colors = plt.cm.RdYlGn(dnn_probs / np.max(dnn_probs))
        axes[1, 1].barh(range(len(AGE_LIST)), dnn_probs, color=colors, edgecolor='black')
        axes[1, 1].set_yticks(range(len(AGE_LIST)))
        axes[1, 1].set_yticklabels(AGE_LIST)
        axes[1, 1].set_xlabel('Probability', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('DNN Age Group Probabilities', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        for i, prob in enumerate(dnn_probs):
            axes[1, 1].text(prob + 0.01, i, f'{prob:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Single image visualization saved to: {save_path}")
    
    return fig

if __name__ == "__main__":
    print("=" * 60)
    print("Age Estimation Model Visualization")
    print("=" * 60)
    print(f"\nModels Loaded:")
    print(f"  DNN: {'✓' if USE_DNN else '✗'}")
    print(f"  CNN: {'✓' if age_cnn_model is not None else '✗'}")
    
    # Create main visualization
    print("\n" + "-" * 60)
    print("Creating comprehensive results dashboard...")
    create_visualizations(save_path='age_model_results.png')
    
    print("\n" + "-" * 60)
    print("Visualization complete!")
    print("\nTo use with a real image, call:")
    print("  visualize_from_image('path/to/image.jpg')")
    print("=" * 60)
    
    # Show plot
    plt.show()

