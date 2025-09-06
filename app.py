import warnings
warnings.filterwarnings("ignore", message=".*pretrained.*")

import os
import glob
import time
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import EfficientNet_B3_Weights, EfficientNet_V2_S_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import numpy as np
import cv2  # Used for heatmap overlay
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for, flash, session

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

#############################################
# 1. Treatment Recommendation Mapping (from PDF)
#############################################
treatment_suggestions_pdf = {
    0: "Normal: Maintain regular physical activity and avoid joint overuse.",
    1: "Mild: Start low-impact exercises (e.g., walking, swimming) and consider early physiotherapy and pain management.",
    2: "Moderate: Implement structured physical therapy, use NSAIDs as needed, and consider bracing plus weight management.",
    3: "Severe: Employ multimodal pain management (e.g., corticosteroid injections, stronger medications) and consult an orthopedic specialist.",
    4: "Very Severe: Consider surgical options (e.g., total knee replacement) along with comprehensive pre/post-op pain management and psychological support."
}

#############################################
# 2. Data Augmentation / Transforms
#############################################
# For multi-disease classifier
multi_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
multi_test_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# For severity classification
severity_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # Slightly more rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
severity_test_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

#############################################
# 3. Helper Function to List Image Paths with Constant Label
#############################################
def get_image_paths_and_labels(root_dir, label, recursive=True):
    pattern = os.path.join(root_dir, '**', '*.*') if recursive else os.path.join(root_dir, '*.*')
    files = glob.glob(pattern, recursive=recursive)
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if len(files) == 0:
        print(f"Warning: No image files found in directory: {root_dir}")
    return [(f, label) for f in files]

#############################################
# 4. Custom Dataset
#############################################
class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

#############################################
# 5. Model Definitions
#############################################
# A. Multi-disease classifier (EfficientNet-B3 based)
class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(DiseaseClassifier, self).__init__()
        self.base_model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

class ImprovedKneeOASeverityClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(ImprovedKneeOASeverityClassifier, self).__init__()
        self.base_model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)


#############################################
# 6. Prepare Datasets for Training and Testing
#############################################
# Multi-Disease classifier dataset directories
knee_train_root = r"C:\Users\harip\OneDrive\Desktop\archive (20)"
ra_train_root = r"C:\Users\harip\OneDrive\Desktop\x-ray rheumatology.v2-release.multiclass\train"
bf_fractured_root = r"C:\Users\harip\OneDrive\Desktop\FracAtlas (2)\FracAtlas\FracAtlas\images\Fractured"
bf_non_fractured_root = r"C:\Users\harip\OneDrive\Desktop\FracAtlas (2)\FracAtlas\FracAtlas\images\Non_fractured"

knee_samples = get_image_paths_and_labels(knee_train_root, label=0, recursive=True)
ra_samples = get_image_paths_and_labels(ra_train_root, label=1, recursive=True)
bf_fractured_samples = get_image_paths_and_labels(bf_fractured_root, label=2, recursive=True)
bf_non_fractured_samples = get_image_paths_and_labels(bf_non_fractured_root, label=3, recursive=True)
all_samples = knee_samples + ra_samples + bf_fractured_samples + bf_non_fractured_samples

if len(all_samples) == 0:
    raise ValueError("No image files found in any of the specified directories. Please verify the paths.")

random.shuffle(all_samples)
split_idx = int(0.8 * len(all_samples))
multi_train_samples = all_samples[:split_idx]
multi_test_samples = all_samples[split_idx:]

multi_train_dataset = CustomImageDataset(multi_train_samples, transform=multi_train_transform)
multi_test_dataset  = CustomImageDataset(multi_test_samples, transform=multi_test_transform)
multi_train_loader = DataLoader(multi_train_dataset, batch_size=32, shuffle=True, num_workers=4)
multi_test_loader  = DataLoader(multi_test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Severity classification dataset directories
severity_0_dir = r"C:\Users\harip\OneDrive\Desktop\archive (20)\koa\0"
severity_1_dir = r"C:\Users\harip\OneDrive\Desktop\archive (20)\koa\1"
severity_2_dir = r"C:\Users\harip\OneDrive\Desktop\archive (20)\koa\2"
severity_3_dir = r"C:\Users\harip\OneDrive\Desktop\archive (20)\koa\3"
severity_4_dir = r"C:\Users\harip\OneDrive\Desktop\archive (20)\koa\4"

severity_samples = []
severity_samples += get_image_paths_and_labels(severity_0_dir, 0, recursive=False)
severity_samples += get_image_paths_and_labels(severity_1_dir, 1, recursive=False)
severity_samples += get_image_paths_and_labels(severity_2_dir, 2, recursive=False)
severity_samples += get_image_paths_and_labels(severity_3_dir, 3, recursive=False)
severity_samples += get_image_paths_and_labels(severity_4_dir, 4, recursive=False)

if len(severity_samples) == 0:
    raise ValueError("No severity images found in the specified severity directories. Please verify the paths.")

random.shuffle(severity_samples)
split_idx = int(0.8 * len(severity_samples))
severity_train_samples = severity_samples[:split_idx]
severity_val_samples = severity_samples[split_idx:]

severity_train_dataset = CustomImageDataset(severity_train_samples, transform=severity_train_transform)
severity_val_dataset = CustomImageDataset(severity_val_samples, transform=severity_test_transform)

knee_train_loader = DataLoader(severity_train_dataset, batch_size=32, shuffle=True, num_workers=4)
knee_val_loader   = DataLoader(severity_val_dataset, batch_size=32, shuffle=False, num_workers=4)

#############################################
# 7. Training Function (with optional Scheduler)
#############################################
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, model_save_path, scheduler=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                history["test_loss"].append(epoch_loss)
                history["test_acc"].append(epoch_acc.item())
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), model_save_path)
                    print(f'-- Best model saved with accuracy: {best_acc:.4f}')
                    
        if scheduler is not None:
            scheduler.step()
    
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Test Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history

#############################################
# 8. Grad-CAM for Explainability and X-ray Heatmaps
#############################################
def grad_cam(model, image_tensor, target_class, target_layer):
    model.eval()
    gradients = []
    activations = []
    
    def save_gradient(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())
    
    handle = target_layer.register_forward_hook(lambda m, i, o: activations.append(o.detach()))
    if hasattr(target_layer, 'register_full_backward_hook'):
        handle_grad = target_layer.register_full_backward_hook(save_gradient)
    else:
        handle_grad = target_layer.register_backward_hook(save_gradient)
    
    output = model(image_tensor)
    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()
    
    grad = gradients[0]
    act = activations[0]
    weights = torch.mean(grad, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * act, dim=1).squeeze()
    cam = torch.clamp(cam, min=0)
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()
    cam = cam.cpu().numpy()
    
    handle.remove()
    handle_grad.remove()
    return cam

def explain_severity_prediction(image_path, model, device, transform):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        confidence = confidence.item()
        pred_class = pred_class.item()

    # For EfficientNet V2-S, use the last block of features as the target layer.
    target_layer = model.base_model.features[-1]
    heatmap = grad_cam(model, input_tensor, pred_class, target_layer)

    # This function returns the predicted severity class, its confidence, and the heatmap.
    return pred_class, confidence, heatmap

#############################################
# Helper Function to Overlay Heatmap
#############################################
def overlay_heatmap(original_image, heatmap, intensity=0.4, colormap=cv2.COLORMAP_JET):
    # Convert the heatmap to 0-255 scale
    heatmap_uint8 = np.uint8(255 * heatmap)
    # Get the original image's dimensions (width, height)
    orig_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    height, width, _ = orig_cv.shape
    # Resize the heatmap to match the original image size
    heatmap_resized = cv2.resize(heatmap_uint8, (width, height))
    # Apply the color map to the resized heatmap
    heatmap_color = cv2.applyColorMap(heatmap_resized, colormap)
    # Superimpose the heatmap on the original image
    superimposed_image = cv2.addWeighted(orig_cv, 1 - intensity, heatmap_color, intensity, 0)
    return superimposed_image




#############################################
# 9. Inference Pipeline
#############################################

def predict_disease_and_treatment(image_path, device, disease_model, severity_model, transform):
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Predict disease from multi-disease model
    if disease_model is not None:
        disease_model.eval()
        with torch.no_grad():
            outputs = disease_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, dim=1)
            disease_class = pred.item()
            disease_conf = confidence.item()
    else:
        disease_class = 0
        disease_conf = 1.0
    disease_map = {0: 'Knee_OA', 1: 'Rheumatoid_Arthritis', 2: 'Bone_Fracture', 3: 'No_disease'}
    disease_name = disease_map.get(disease_class, "Unknown")
    
    if disease_name == 'Knee_OA':
        severity_model.eval()
        with torch.no_grad():
            sev_out = severity_model(input_tensor)
            sev_probs = torch.softmax(sev_out, dim=1)
            sev_conf, sev_pred = torch.max(sev_probs, dim=1)
            severity_level_num = sev_pred.item()
            sev_confidence = sev_conf.item()
        # Map numeric severity level to text
        severity_map_text = {0: 'Normal', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Very Severe'}
        severity_level = severity_map_text.get(severity_level_num, "Unknown")
        treatment = treatment_suggestions_pdf.get(severity_level_num, "No treatment recommendation available.")
        
        # Optionally, compute gradCAM if you want to show heatmap when severity is between 1 and 4.
        # (You can add an if-condition here if you wish to show heatmap only for certain severity ranges.)
        pred_class, gradcam_conf, heatmap = explain_severity_prediction(image_path, severity_model, device, transform)
        # Overlay the heatmap on the original image
        superimposed = overlay_heatmap(original_image, heatmap)
        # Save the heatmap image to the static folder
        heatmap_filename = f"heatmap_{os.path.basename(image_path)}"
        heatmap_path = os.path.join('static', 'uploads', heatmap_filename)
        cv2.imwrite(heatmap_path, superimposed)
        
        result = {
            'disease_name': disease_name,
            'disease_confidence': disease_conf,
            'severity_level': severity_level,
            'severity_confidence': sev_confidence,
            'treatment': treatment,
            'scan_date': time.strftime('%Y-%m-%d %H:%M', time.localtime()),
            'id': 1,
            'gradcam_image': heatmap_filename  # new key with heatmap filename
        }
    else:
        result = {
            'disease_name': disease_name,
            'disease_confidence': disease_conf,
            'scan_date': time.strftime('%Y-%m-%d %H:%M', time.localtime()),
            'id': 1
        }
    return result

#############################################
# 10. Flask Integration
#############################################
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change in production

# Paths to save models
DISEASE_MODEL_PATH = 'disease_classifier_efficientnetb3.pth'
# Align severity path with training save
SEVERITY_MODEL_PATH = 'improved_knee_oa_severity_classifier.pth'

# Load models for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

disease_model = DiseaseClassifier(num_classes=4).to(device)
severity_model = ImprovedKneeOASeverityClassifier(num_classes=5).to(device)

# Try loading weights
def load_model(model, path, name):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded {name} model.")
    else:
        print(f"{name} model not found at {path}.")

load_model(disease_model, DISEASE_MODEL_PATH, "multi-disease classifier")

load_model(severity_model, SEVERITY_MODEL_PATH, "severity classifier")

# Define routes for the web app
@app.route('/')
def home():
    return render_template('index.html')



from forms import LoginForm  # if you put the form in a file named forms.py

# Dashboard page
@app.route('/dashboard')
def dashboard():
    # Retrieve past scan results or show an empty list
    scan_results = session.get('scan_history', [])
    return render_template('dashboard.html', scan_results=scan_results)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # Dummy login logic (replace with your own authentication)
        flash("Logged in successfully.", "success")
        return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)

from forms import RegisterForm

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        # (Add your user registration logic here)
        flash("Registered successfully. Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

from flask import session
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash("No selected file", "danger")
            return redirect(request.url)
        if file:
            upload_folder = os.path.join('static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            filepath = os.path.join(upload_folder, file.filename)
            file.save(filepath)
            prediction = predict_disease_and_treatment(filepath, device, disease_model, severity_model, severity_test_transform)
            session['prediction'] = prediction  # optionally store in session
            return redirect(url_for('results', image_filename=file.filename))
    return render_template('upload.html')

@app.route('/results')
def results():
    image_filename = request.args.get('image_filename', None)
    prediction = session.get('prediction', None)
    if not prediction:
        prediction = {
            'disease_name': 'N/A',
            'disease_confidence': 0,
            'severity_level': 'N/A',
            'treatment': 'N/A',
            'scan_date': time.strftime('%Y-%m-%d %H:%M', time.localtime()),
            'id': 1
        }
    return render_template('results.html', result=prediction, image_path=image_filename)


#############################################
# Main Execution Block
#############################################
if __name__ == '__main__':
    # To train the models, uncomment the block below and run this script.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    # Train Multi-Disease Classifier
    print("Training Multi-Disease Classifier...")
    disease_model = DiseaseClassifier(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(disease_model.parameters(), lr=0.001)
    dataloaders_multi = {'train': multi_train_loader, 'test': multi_test_loader}
    disease_model, history_multi = train_model(disease_model, dataloaders_multi, criterion, optimizer,
                                                 num_epochs=5, device=device,
                                                 model_save_path=DISEASE_MODEL_PATH)
    print("Multi-Disease Classifier Training History:")
    print(history_multi)
   
     # ---- Train Knee OA Severity Classifier (Improved) ----
    print("\nTraining Knee OA Severity Classifier with EfficientNet V2 (Improved)...")
    severity_model = ImprovedKneeOASeverityClassifier(num_classes=5).to(device)
    criterion_sev = nn.CrossEntropyLoss()
    optimizer_sev = optim.Adam(severity_model.parameters(), lr=0.0005)
    scheduler_sev = optim.lr_scheduler.StepLR(optimizer_sev, step_size=3, gamma=0.1)
    dataloaders_sev = {'train': knee_train_loader, 'test': knee_val_loader}
    severity_model, history_sev = train_model(severity_model, dataloaders_sev, criterion_sev, optimizer_sev,
                                              num_epochs=5, device=device,
                                              model_save_path='improved_knee_oa_severity_classifier.pth',
                                              scheduler=scheduler_sev)
    
    print("\nKnee OA Severity Classifier Training History:")
    print(history_sev)
     '''
@app.context_processor
def inject_current_user():
    class DummyUser:
        is_authenticated = False
    return dict(current_user=DummyUser())

    # Disable the reloader to prevent double execution
app.run(debug=True, use_reloader=False)

