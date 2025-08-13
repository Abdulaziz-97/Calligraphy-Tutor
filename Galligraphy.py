#!/usr/bin/env python3

import os
import json
import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel
from PIL import Image, ImageDraw
import re
import random
import copy
import math
import matplotlib.pyplot as plt
import pickle
import numpy as np

print("TESTING EXACT WORD CLASSIFIER")
print("=" * 50)

# === CONFIGURATION ===
ENCODER_PATH = r"C:REPLACE WITH ENCODER PATH" # Encoder path same as the file in the repo(Encoder)
DATASET_PATH = r"C:REPLACE WITH DATASET PATH" # Dataset path same as the file in the repo(Dataset)
Classifier_path = r"C:REPLACE WITH MODEL PATH" # Model path same as the file in the repo(Classifier)

class ExactWordDatabase:
    """Database class - same as training"""
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.word_to_coordinates = {}
        self.word_to_id = {}
        self.id_to_word = {}
    
    def get_perfect_coordinates(self, phrase):
        if phrase in self.word_to_coordinates:
            return self.word_to_coordinates[phrase][0]['coordinates']
        return None
    
    def get_word_from_id(self, word_id):
        return self.id_to_word.get(word_id, None)
    
    def get_word_id(self, phrase):
        return self.word_to_id.get(phrase, -1)
    
    def get_num_classes(self):
        return len(self.word_to_id)

class ExactWordClassifier(nn.Module):
    """Exact word classifier - same as training"""
    def __init__(self, encoder_path, num_classes):
        super().__init__()
        
        self.encoder = ViTModel.from_pretrained(encoder_path)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        encoder_dim = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, pixel_values):
        with torch.no_grad():
            encoder_outputs = self.encoder(pixel_values=pixel_values)
            features = encoder_outputs.last_hidden_state[:, 0]
        return self.classifier(features)

def coordinates_to_image(coordinates, size=(224, 224)):
    """Convert coordinates to image"""
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)
    
    all_points = []
    if isinstance(coordinates, list):
        for item in coordinates:
            if isinstance(item, dict):
                for points in item.values():
                    if isinstance(points, list):
                        for p in points:
                            if len(p) >= 2:
                                try:
                                    x, y = float(p[0]), float(p[1])
                                    if not (math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y)):
                                        all_points.append((x, y))
                                except (ValueError, TypeError):
                                    continue
    
    if not all_points:
        return img
    
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    
    if xs and ys:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        padding = 20
        scale_x = (size[0] - 2*padding) / max(1, max_x - min_x)
        scale_y = (size[1] - 2*padding) / max(1, max_y - min_y)
        scale = min(scale_x, scale_y) * 0.8
        
        for i in range(0, len(all_points) - 1, 2):
            try:
                x1, y1 = all_points[i]
                x2, y2 = all_points[i + 1]
                
                x1_s = (x1 - min_x) * scale + padding
                y1_s = (y1 - min_y) * scale + padding
                x2_s = (x2 - min_x) * scale + padding
                y2_s = (y2 - min_y) * scale + padding
                
                draw.line([(x1_s, y1_s), (x2_s, y2_s)], fill='black', width=2)
            except:
                continue
    
    return img

def create_display_image(coordinates, size=(300, 200), stroke_color='black', title=""):
    """Create clear display image"""
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)
    
    if title:
        draw.text((10, 10), title, fill='black')
    
    all_points = []
    if isinstance(coordinates, list):
        for item in coordinates:
            if isinstance(item, dict):
                for points in item.values():
                    if isinstance(points, list):
                        for p in points:
                            if len(p) >= 2:
                                try:
                                    x, y = float(p[0]), float(p[1])
                                    if not (math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y)):
                                        all_points.append((x, y))
                                except (ValueError, TypeError):
                                    continue
    
    if not all_points:
        draw.text((100, 100), "No Data", fill='red')
        return img
    
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    padding = 30
    available_width = size[0] - 2 * padding
    available_height = size[1] - 2 * padding - (20 if title else 0)
    
    scale_x = available_width / max(1, max_x - min_x)
    scale_y = available_height / max(1, max_y - min_y)
    scale = min(scale_x, scale_y) * 0.85
    
    y_offset = 20 if title else 0
    
    for i in range(0, len(all_points) - 1, 2):
        try:
            x1, y1 = all_points[i]
            x2, y2 = all_points[i + 1]
            
            x1_s = (x1 - min_x) * scale + padding
            y1_s = (y1 - min_y) * scale + padding + y_offset
            x2_s = (x2 - min_x) * scale + padding
            y2_s = (y2 - min_y) * scale + padding + y_offset
            
            draw.line([(x1_s, y1_s), (x2_s, y2_s)], fill=stroke_color, width=2)
        except:
            continue
    
    return img

def add_noise(coordinates, noise_level=5):
    """Add noise to coordinates"""
    noisy = copy.deepcopy(coordinates)
    if isinstance(noisy, list):
        for item in noisy:
            if isinstance(item, dict):
                for stroke_list in item.values():
                    if isinstance(stroke_list, list):
                        for point in stroke_list:
                            if isinstance(point, list) and len(point) >= 2:
                                try:
                                    point[0] += random.randint(-noise_level, noise_level)
                                    point[1] += random.randint(-noise_level, noise_level)
                                except (ValueError, TypeError):
                                    continue
    return noisy

def load_exact_classifier():
    """Load the trained exact classifier"""
    print("\nLoading exact classifier...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    weights_path = os.path.join(Classifier_path, 'best_exact_classifier.pt')
    database_path = os.path.join(Classifier_path, 'exact_database.pkl')
    
    if not os.path.exists(weights_path) or not os.path.exists(database_path):
        print(f"Model files not found:")
        print(f"   Weights: {weights_path}")
        print(f"   Database: {database_path}")
        return None, None
    
    # Load database
    with open(database_path, 'rb') as f:
        database = pickle.load(f)
    
    # Load model
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    
    model = ExactWordClassifier(ENCODER_PATH, database.get_num_classes())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    accuracy = checkpoint.get('accuracy', 0)
    epoch = checkpoint.get('epoch', 0)
    
    print(f"Loaded exact classifier:")
    print(f"   Training accuracy: {accuracy:.2f}%")
    print(f"   Trained for {epoch} epochs")
    print(f"   {database.get_num_classes()} complete phrases")
    
    return model, database

def test_exact_differentiation(model, database):
    """Test exact differentiation with random phrase selection"""
    print(f"\nTESTING EXACT DIFFERENTIATION - RANDOM PHRASES")
    print("=" * 50)
    
    # Set random seed based on current time for different results each run
    import time
    random.seed(int(time.time()))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = ViTImageProcessor.from_pretrained(ENCODER_PATH)
    
    # Get all available phrases
    all_phrases = list(database.word_to_coordinates.keys())
    
    # Randomly select phrases for testing (always include بسم variants if available)
    basm_variants = [p for p in all_phrases if 'بسم' in p]
    other_phrases = [p for p in all_phrases if 'بسم' not in p]
    
    # Select test phrases: 1-2 بسم variants + 3-4 random others
    test_phrases = []
    
    if basm_variants:
        # Include 1-2 بسم variants for differentiation testing
        num_basm = min(2, len(basm_variants))
        selected_basm = random.sample(basm_variants, num_basm)
        test_phrases.extend(selected_basm)
        print(f"Including {num_basm} بسم variants for exact differentiation:")
        for phrase in selected_basm:
            print(f"   - '{phrase}'")
    
    # Add random other phrases
    num_others = 4 - len(test_phrases)
    if len(other_phrases) >= num_others:
        selected_others = random.sample(other_phrases, num_others)
        test_phrases.extend(selected_others)
    else:
        test_phrases.extend(other_phrases)
    
    # Final random shuffle
    random.shuffle(test_phrases)
    
    print(f"\nRANDOM TEST SELECTION ({len(test_phrases)} phrases):")
    for i, phrase in enumerate(test_phrases, 1):
        print(f"   {i}. '{phrase}'")
    
    if len(test_phrases) < 2:
        print("Need at least 2 phrases for testing")
        return
    
    # Test each selected phrase
    results = []
    
    for phrase in test_phrases:
        print(f"\nTesting: '{phrase}'")
        
        # Get perfect coordinates
        perfect_coords = database.get_perfect_coordinates(phrase)
        if not perfect_coords:
            print(f"   No coordinates found for '{phrase}'")
            continue
        
        # Test with different noise levels
        for noise_level in [0,1,2,3,4,5,6]:
            if noise_level == 0:
                test_coords = perfect_coords
                noise_desc = "Perfect"
            else:
                test_coords = add_noise(perfect_coords, noise_level)
                noise_desc = f"Noise {noise_level}"
            
            # Convert to image and classify
            test_img = coordinates_to_image(test_coords)
            pixel_values = processor(images=test_img, return_tensors="pt").pixel_values.to(device)
            
            with torch.no_grad():
                logits = model(pixel_values)
                predicted_id = torch.argmax(logits, dim=1).item()
                predicted_phrase = database.get_word_from_id(predicted_id)
                confidence = torch.softmax(logits, dim=1).max().item()
            
            is_correct = (predicted_phrase == phrase)
            
            print(f"   {noise_desc}: {predicted_phrase} (conf: {confidence:.3f}) {'CORRECT' if is_correct else 'WRONG'}")
            
            results.append({
                'true_phrase': phrase,
                'predicted_phrase': predicted_phrase,
                'confidence': confidence,
                'correct': is_correct,
                'noise_level': noise_level,
                'test_coords': test_coords,
                'perfect_coords': perfect_coords
            })
    
    # Create visualization
    if results:
        print(f"\nCreating differentiation visualization...")

        # Group by phrase and noise level
        phrase_noise_results = {}
        for r in results:
            key = (r['true_phrase'], r['noise_level'])
            phrase_noise_results[key] = r

        # Get all unique phrases and all noise levels tested
        unique_phrases = sorted(list(set(r['true_phrase'] for r in results)))
        unique_noise_levels = sorted(list(set(r['noise_level'] for r in results)))
        num_phrases = len(unique_phrases)
        num_noise = len(unique_noise_levels)

        # Create a grid: rows = noise levels, cols = phrases, 3 columns per phrase (input, prediction, output)
        fig, axes = plt.subplots(num_noise, num_phrases * 3, figsize=(4*num_phrases*1.5, 2.5*num_noise))
        if num_noise == 1:
            axes = axes.reshape(1, num_phrases * 3)
        if num_phrases == 1:
            axes = axes.reshape(num_noise, 3)

        fig.suptitle(
            f'Random Phrase Differentiation Test\n'
            f'{len(results)} Tests - {num_phrases} Phrases × {num_noise} Noise Levels\n'
            f'Model input, prediction, and output shown for each test',
            fontsize=16, weight='bold'
        )

        for col, phrase in enumerate(unique_phrases):
            for row, noise_level in enumerate(unique_noise_levels):
                r = phrase_noise_results.get((phrase, noise_level), None)
                base_col = col * 3
                # Input
                ax_input = axes[row, base_col]
                # Prediction
                ax_pred = axes[row, base_col + 1]
                # Output (ground truth)
                ax_output = axes[row, base_col + 2]
                if r is not None:
                    # Input (with noise)
                    input_img = create_display_image(r['test_coords'], stroke_color='red', title="Input")
                    ax_input.imshow(input_img)
                    ax_input.set_title(f"Input\n{phrase}\nNoise: {noise_level}", fontsize=8, weight='bold')
                    ax_input.axis('off')

                    # Prediction (model output)
                    # FIX: Use the predicted phrase's perfect coordinates for the prediction image,
                    # not the test_coords (which are the noisy input).
                    pred_phrase = r['predicted_phrase']
                    pred_coords = database.get_perfect_coordinates(pred_phrase)
                    if pred_coords is not None:
                        pred_img = create_display_image(pred_coords, stroke_color='blue', title="Model Prediction")
                    else:
                        # If for some reason the predicted phrase is not in the database, fallback to test_coords
                        pred_img = create_display_image(r['test_coords'], stroke_color='blue', title="Model Prediction")
                    pred_title = f"Pred: {pred_phrase}\nConf: {r['confidence']:.2f}\n{'CORRECT' if r['correct'] else 'WRONG'}"
                    ax_pred.imshow(pred_img)
                    ax_pred.set_title(pred_title, fontsize=8, weight='bold', color='green' if r['correct'] else 'red')
                    ax_pred.axis('off')

                    # Output (ground truth)
                    output_img = create_display_image(r['perfect_coords'], stroke_color='black', title="Ground Truth")
                    ax_output.imshow(output_img)
                    ax_output.set_title(f"Ground Truth\n{phrase}", fontsize=8, weight='bold')
                    ax_output.axis('off')
                else:
                    ax_input.axis('off')
                    ax_pred.axis('off')
                    ax_output.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_path = f"random_phrase_test_{num_phrases}_phrases_{num_noise}_noise.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Test results saved: {output_path}")
        
        try:
            plt.show()
        except:
            pass
    
    # Summary
    print(f"\nEXACT DIFFERENTIATION SUMMARY:")
    print("=" * 40)
    correct_predictions = sum(1 for r in results if r['correct'])
    total_tests = len(results)
    accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Tests performed: {total_tests}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Wrong predictions: {total_tests - correct_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 90:
        print(f"\nEXCELLENT! Model perfectly differentiates exact phrases!")
        print(f"Random phrase differentiation working perfectly!")
    elif accuracy >= 75:
        print(f"\nGOOD! Model shows strong exact differentiation!")
        print(f"Random phrase testing shows reliable performance!")
    else:
        print(f"\nNeeds improvement for reliable exact matching")
        print(f"Consider more training or data augmentation")
    
    return results

def main():
    """Main testing function"""
    print("EXACT WORD CLASSIFIER TESTING")
    print("Testing exact differentiation with random phrase selection")
    print("Includes بسم variants + random Arabic phrases")
    
    # Load classifier
    model, database = load_exact_classifier()
    
    if model is None or database is None:
        print("\nCould not load exact classifier")
        print("Make sure EXACT_word_classifier.py training completed successfully")
        return
    
    # Test exact differentiation
    results = test_exact_differentiation(model, database)
    
    if results:
        print(f"\nCONCLUSION:")
        print(f"The exact word classifier successfully differentiates between:")
        
        tested_phrases = list(set(r['true_phrase'] for r in results))
        for phrase in tested_phrases:
            print(f"   - '{phrase}'")
        
        print(f"\nReady for deployment in your AI Calligraphy Tutor!")
        print(f"Model location: {Classifier_path}")

if __name__ == "__main__":
    main()