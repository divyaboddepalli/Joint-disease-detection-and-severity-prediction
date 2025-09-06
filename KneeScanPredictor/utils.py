import os
import uuid
import numpy as np
import random
import cv2
from flask import current_app, flash
from PIL import Image
import io
import base64
import logging
import matplotlib as mpl

# Set non-interactive backend (avoid tkinter threads)
mpl.use('Agg')  

# Set Matplotlib's log level to WARNING to suppress debug messages
mpl.set_loglevel('WARNING')

import matplotlib.pyplot as plt

def is_valid_knee_scan(image_path):
    """
    Validate if the image is likely to be a knee scan.
    This uses multiple validation techniques to filter out non-medical images.
    In a real application, this would use a trained ML model for validation.
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Check image dimensions - knee scans typically have certain aspect ratios
        width, height = img.size
        aspect_ratio = width / height
        
        # Most knee MRI/X-ray scans have aspect ratios between 0.4 and 2.5
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            print(f"Image aspect ratio {aspect_ratio} outside typical range for knee scans")
            return False
        
        # Convert to grayscale for analysis
        gray_img = img.convert('L')
        img_array = np.array(gray_img)
        histogram = gray_img.histogram()
        
        # Calculate standard deviation of histogram - medical images often have specific patterns
        hist_array = np.array(histogram)
        hist_std = np.std(hist_array)
        
        # Medical scans typically have higher contrast, but we'll use a lower threshold
        if hist_std < 25:  # Reduced threshold
            print(f"Image histogram standard deviation {hist_std} too low for typical knee scan")
            return False
            
        # Check the brightness distribution - medical images typically have specific patterns
        mean_brightness = np.mean(img_array)
        
        # Medical knee scans typically have brightness in a specific range (widened)
        if mean_brightness < 30 or mean_brightness > 230:
            print(f"Image mean brightness {mean_brightness} outside typical range for knee scans")
            return False
            
        # Calculate histogram entropy - medical images have specific entropy patterns
        hist_norm = hist_array / hist_array.sum()
        hist_entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        
        print(f"Image entropy {hist_entropy}")
        
        # Medical images typically have entropy in a specific range (widened)
        if hist_entropy < 2.5 or hist_entropy > 7.5:
            print(f"Image entropy {hist_entropy} is allowed for medical scans")
            # We won't reject based on entropy alone
        
        # --- COMPREHENSIVE NON-MEDICAL IMAGE DETECTION ---
        
        # 1. Face/Portrait Detection
        h, w = img_array.shape
        
        # Calculate regions for face detection
        center_region = img_array[h//4:3*h//4, w//4:3*w//4]
        top_region = img_array[0:h//3, w//4:3*w//4]
        
        # Calculate eyes region (typically upper part of center)
        eyes_region = img_array[h//4:h//2, w//4:3*w//4]
        eyes_var = np.var(eyes_region)
        
        # Calculate mouth region (typically lower part of center)
        mouth_region = img_array[h//2:3*h//4, w//4:3*w//4]
        mouth_var = np.var(mouth_region)
        
        # Check for face-like features (strong indicators of a portrait)
        center_variance = np.var(center_region)
        
        # Check for strong facial symmetry
        left_half = img_array[:, 0:w//2]
        right_half = img_array[:, w//2:w]
        right_half_flipped = np.fliplr(right_half)
        
        # Trim to match sizes if needed
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        symmetry_diff = np.mean(np.abs(left_half[:, 0:min_width] - right_half_flipped[:, 0:min_width]))
        
        # Enhanced portrait detection
        is_likely_portrait = (
            (center_variance > 200) and     # High variance in facial features area
            (symmetry_diff < 25) and        # High facial symmetry
            (aspect_ratio > 0.5 and aspect_ratio < 1.8) and  # Portrait-like aspect ratio
            (eyes_var > 150 or mouth_var > 150)  # High variance in eyes or mouth regions
        )
        
        # 2. Check for skin tones (common in portraits)
        has_skin_tones = False
        if img.mode == 'RGB':
            rgb_img = np.array(img)
            r, g, b = rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2]
            
            # Check for skin-like colors in center region
            center_r = np.mean(r[h//4:3*h//4, w//4:3*w//4])
            center_g = np.mean(g[h//4:3*h//4, w//4:3*w//4])
            center_b = np.mean(b[h//4:3*h//4, w//4:3*w//4])
            
            # Calculate skin tone probability (simplified)
            # Most skin tones have higher R channel, followed by G, then B
            has_skin_tones = (center_r > center_g > center_b) and (center_r > 80) and (center_g > 40)
            
            if has_skin_tones and center_variance > 150 and symmetry_diff < 30:
                print("REJECTED: Image has skin tones and portrait-like characteristics")
                return False
        
        if is_likely_portrait:
            print("REJECTED: Image has strong portrait/face characteristics")
            print(f"Center variance: {center_variance}, Symmetry diff: {symmetry_diff}, Aspect ratio: {aspect_ratio}")
            print(f"Eyes var: {eyes_var}, Mouth var: {mouth_var}")
            return False
        
        # 3. Check for landscapes/cityscapes (features common in tourist photos)
        if img.mode == 'RGB':
            rgb_img = np.array(img)
            
            # Calculate color histograms for RGB channels
            r_hist = np.histogram(rgb_img[:,:,0], bins=50, range=(0,255))[0]
            g_hist = np.histogram(rgb_img[:,:,1], bins=50, range=(0,255))[0]
            b_hist = np.histogram(rgb_img[:,:,2], bins=50, range=(0,255))[0]
            
            # Normalize histograms
            r_hist = r_hist / np.sum(r_hist)
            g_hist = g_hist / np.sum(g_hist)
            b_hist = b_hist / np.sum(b_hist)
            
            # Calculate histogram entropy for each channel
            r_entropy = -np.sum(r_hist * np.log2(r_hist + 1e-10))
            g_entropy = -np.sum(g_hist * np.log2(g_hist + 1e-10))
            b_entropy = -np.sum(b_hist * np.log2(b_hist + 1e-10))
            
            # Calculate color distribution metrics
            color_entropy = (r_entropy + g_entropy + b_entropy) / 3
            color_range = np.max(rgb_img) - np.min(rgb_img)
            color_std = np.std(rgb_img)
            
            # Calculate sky detection (usually blue dominant in upper third)
            upper_third = rgb_img[0:h//3, :, :]
            upper_b_avg = np.mean(upper_third[:,:,2])
            upper_r_avg = np.mean(upper_third[:,:,0])
            upper_g_avg = np.mean(upper_third[:,:,1])
            
            has_sky_color = upper_b_avg > upper_r_avg and upper_b_avg > upper_g_avg
            
            # Check color saturation
            hsv_img = rgb_img.astype(np.float32) / 255
            max_vals = np.max(hsv_img, axis=2)
            min_vals = np.min(hsv_img, axis=2)
            delta = max_vals - min_vals
            saturation = np.zeros_like(max_vals)
            saturation[max_vals != 0] = delta[max_vals != 0] / max_vals[max_vals != 0]
            avg_saturation = np.mean(saturation)
            
            # Landscape detection metrics
            is_likely_landscape = (
                color_entropy > 4.5 and  # High color entropy (varied colors)
                color_range > 200 and    # Wide color range
                color_std > 50 and       # High color variation
                avg_saturation > 0.3     # Moderate to high saturation
            )
            
            # Cityscape detection (look for regular patterns and high frequency components)
            # Calculate vertical and horizontal gradients
            h_grad = np.abs(np.diff(np.mean(rgb_img, axis=2), axis=1))
            v_grad = np.abs(np.diff(np.mean(rgb_img, axis=2), axis=0))
            
            # Calculate gradient statistics
            h_grad_mean = np.mean(h_grad)
            v_grad_mean = np.mean(v_grad)
            h_grad_var = np.var(h_grad)
            v_grad_var = np.var(v_grad)
            
            # Cityscape typically has regular patterns and high frequency components
            is_likely_cityscape = (
                h_grad_mean > 15 and
                v_grad_mean > 15 and
                h_grad_var > 200 and
                v_grad_var > 200 and
                color_entropy > 4.0
            )
            
            # Night cityscape detection (look for bright spots on dark background)
            brightness = np.mean(rgb_img, axis=2)
            bright_pixels = np.sum(brightness > 200) / brightness.size
            dark_pixels = np.sum(brightness < 50) / brightness.size
            
            is_night_cityscape = (
                dark_pixels > 0.5 and    # Mostly dark
                bright_pixels > 0.01 and # Some bright spots (lights)
                bright_pixels < 0.2      # But not too many
            )
            
            # Check for specific landscape/cityscape characteristics
            if is_likely_landscape:
                print("REJECTED: Image has landscape photo characteristics")
                print(f"Color entropy: {color_entropy}, Color range: {color_range}, Color std: {color_std}")
                print(f"Has sky-like colors: {has_sky_color}, Saturation: {avg_saturation}")
                return False
                
            if is_likely_cityscape or is_night_cityscape:
                print("REJECTED: Image has cityscape photo characteristics")
                print(f"H-grad: {h_grad_mean}, V-grad: {v_grad_mean}")
                print(f"Bright pixels: {bright_pixels}, Dark pixels: {dark_pixels}")
                return False
        
        # 4. Check for landscape/portrait aspect ratios common in photography
        common_photo_ratio = False
        photo_ratios = [1.33, 1.5, 1.78, 2.0]  # 4:3, 3:2, 16:9, 2:1
        for ratio in photo_ratios:
            if abs(aspect_ratio - ratio) < 0.1 or abs(1/aspect_ratio - ratio) < 0.1:
                common_photo_ratio = True
        
        # 5. Additional portrait detection - check for oval face shape
        # Generate a simple edge map
        edges_h = np.abs(np.diff(img_array, axis=0))
        edges_v = np.abs(np.diff(img_array, axis=1))
        
        # Combine horizontal and vertical edges
        edges = np.zeros((h, w))
        edges[:-1, :] += edges_h
        edges[:, :-1] += edges_v[:, :]
        
        # Check center region for oval shapes
        center_edges = edges[h//4:3*h//4, w//4:3*w//4]
        edge_density = np.sum(center_edges > 30) / (center_edges.size)
        
        # Oval shapes have a characteristic edge distribution
        if edge_density > 0.05 and edge_density < 0.3 and symmetry_diff < 25 and center_variance > 180:
            print("REJECTED: Image has oval face-like edge distribution")
            print(f"Edge density: {edge_density}, Symmetry: {symmetry_diff}")
            return False
        
        # --- Medical imaging feature detection ---

        # 1. Check for dark borders (common in medical scans)
        border_width = min(width, height) // 15
        top_border = img_array[0:border_width, :]
        bottom_border = img_array[height-border_width:height, :]
        left_border = img_array[:, 0:border_width]
        right_border = img_array[:, width-border_width:width]
        
        border_brightness = np.median([np.median(top_border), np.median(bottom_border), 
                                      np.median(left_border), np.median(right_border)])
        center_brightness = np.median(img_array[h//4:3*h//4, w//4:3*w//4])
        
        # Check if borders are darker than center (typical in X-rays)
        has_dark_borders = border_brightness < center_brightness * 0.8
        
        # 2. Check for bone-like edges (high gradients in specific areas)
        h_gradient = np.abs(np.diff(img_array, axis=1))
        v_gradient = np.abs(np.diff(img_array, axis=0))
        
        # Calculate gradient statistics
        mean_h_gradient = np.mean(h_gradient)
        mean_v_gradient = np.mean(v_gradient)
        max_gradient = max(np.max(h_gradient), np.max(v_gradient))
        
        # Medical images typically have strong edges (bone boundaries)
        has_strong_edges = (max_gradient > 50) and (mean_h_gradient > 8 or mean_v_gradient > 8)
        
        # 3. Check intensity distribution (common in knee radiographs)
        # Calculate histogram percentiles
        pixel_values = img_array.flatten()
        p10 = np.percentile(pixel_values, 10)
        p90 = np.percentile(pixel_values, 90)
        
        # Medical images typically have good dynamic range
        dynamic_range = p90 - p10
        has_good_range = dynamic_range > 60
        
        # 4. Check for unnatural color distribution (common in non-medical images)
        if img.mode == 'RGB':
            # Convert to numpy array
            rgb_img = np.array(img)
            r, g, b = rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2]
            
            # Calculate color variance (medical images are more monochromatic)
            r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
            color_variance = np.var([r_mean, g_mean, b_mean])
            
            # High color variance suggests non-medical image
            has_natural_colors = color_variance < 200
        else:
            # Grayscale images are more likely to be medical
            has_natural_colors = True
        
        # --- FINAL DECISION LOGIC ---
        
        # Calculate medical image score
        medical_score = 0
        if has_dark_borders: medical_score += 1
        if has_strong_edges: medical_score += 1
        if has_good_range: medical_score += 1
        if has_natural_colors: medical_score += 1
        
        # Allow simpler medical scans to pass through
        print(f"Medical score: {medical_score}")
        print(f"Medical indicators: dark_borders={has_dark_borders}, strong_edges={has_strong_edges}, " +
              f"good_range={has_good_range}, natural_colors={has_natural_colors}")
        
        # Decision - we'll accept images that have at least some medical characteristics
        # This is more lenient to allow legitimate knee scans through
        if medical_score >= 2:
            print("ACCEPTED: Image passed validation as a potential medical scan")
            return True
        else:
            print("REJECTED: Image does not have sufficient medical scan characteristics")
            return False
        
    except Exception as e:
        print(f"Error validating knee scan: {str(e)}")
        # For debugging, let's print the full stack trace
        import traceback
        traceback.print_exc()
        # In case of error, we'll be lenient and accept the image
        # This prevents technical issues from causing false rejections
        print("ACCEPTED: Allowing image despite validation error")
        return True

# For demo purposes, we'll simulate predictions
def predict_knee_oa(image_path):
    """
    Make a prediction for the knee image
    Returns disease name, severity, confidence, and knee health score
    This is a simulated function for demonstration purposes
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return None
        
        # Validate that this is likely a knee scan image
        if not is_valid_knee_scan(image_path):
            print("Uploaded image does not appear to be a valid knee scan")
            return {
                'disease_name': 'Not a Valid Scan',
                'severity_level': 'Unknown',
                'confidence': 0.05,
                'knee_health_score': 0.0
            }
            
        # For demo, we'll randomly choose a severity level and calculate other values
        severity_levels = ['Normal', 'Mild', 'Moderate', 'Severe']
        severity_weights = [0.15, 0.25, 0.40, 0.20]  # More likely to be moderate
        severity = random.choices(severity_levels, weights=severity_weights, k=1)[0]
        
        # Determine values based on severity
        if severity == 'Normal':
            knee_health_score = random.uniform(85, 100)
            confidence = random.uniform(0.85, 0.98)
            disease_name = 'Healthy'
        elif severity == 'Mild':
            knee_health_score = random.uniform(65, 84)
            confidence = random.uniform(0.75, 0.90)
            disease_name = 'Knee Osteoarthritis'
        elif severity == 'Moderate':
            knee_health_score = random.uniform(40, 64)
            confidence = random.uniform(0.80, 0.95)
            disease_name = 'Knee Osteoarthritis'
        else:  # Severe
            knee_health_score = random.uniform(15, 39)
            confidence = random.uniform(0.85, 0.98)
            disease_name = 'Knee Osteoarthritis'
        
        return {
            'disease_name': disease_name,
            'severity_level': severity,
            'confidence': confidence,
            'knee_health_score': knee_health_score
        }
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def save_uploaded_file(file):
    """
    Save an uploaded file to the uploads directory
    """
    try:
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return file_path
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return None

def generate_comparison_chart(knee_health_score):
    """
    Generate a comparison chart between user's knee health and normal knee health
    Returns the chart as a base64 encoded image
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Data
        categories = ['Joint Space', 'Bone Density', 'Cartilage', 'Overall']
        
        # Calculate values for patient (derived from knee health score)
        # This is a simplified example - real values would come from model
        patient_values = [
            max(0, knee_health_score - 10 + np.random.randint(-5, 5)), 
            max(0, knee_health_score - 5 + np.random.randint(-10, 10)),
            max(0, knee_health_score + 5 + np.random.randint(-15, 5)),
            knee_health_score
        ]
        
        # Normal values
        normal_values = [85, 90, 88, 90]
        
        # Width of bars
        width = 0.35
        
        # Positions
        r1 = np.arange(len(categories))
        r2 = [x + width for x in r1]
        
        # Create the chart
        plt.bar(r1, patient_values, width, label='Patient Assessment', color='#3498db')
        plt.bar(r2, normal_values, width, label='Reference Values', color='#2ecc71')
        
        plt.ylim(0, 100)
        plt.title('Joint Structure Assessment')
        plt.ylabel('Structural Integrity (%)')
        plt.xticks([r + width/2 for r in range(len(categories))], categories)
        plt.legend()
        
        # Save chart to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode to base64 for HTML embedding
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return f'data:image/png;base64,{image_base64}'
    
    except Exception as e:
        print(f"Error generating chart: {str(e)}")
        return None
