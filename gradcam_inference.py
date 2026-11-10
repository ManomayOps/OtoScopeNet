import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import EdgeOtoscopeNet

MODEL_PATH = 'edge_otoscope_net_best.pth'
TEST_DIR = './dataset/test'
CLASS_NAMES = ['Acute Otitis Media', 'Chronic Otitis Media', 'Earwax', 'Myringosclerosis', 'Normal']
NORMAL_ALPHA = 0.7
GRADCAM_ALPHA = 0.5

def get_target_layer(model):
    return model.head[0]

def get_random_image_from_class(class_dir):
    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        return None
    return os.path.join(class_dir, random.choice(image_files))

def create_overlay_image(normal_img, gradcam_img, normal_alpha=0.7, gradcam_alpha=0.5):
    normal_img = np.array(normal_img) / 255.0 if normal_img.max() > 1 else normal_img
    gradcam_img = np.array(gradcam_img) / 255.0 if gradcam_img.max() > 1 else gradcam_img
    overlay = np.clip(normal_alpha * normal_img + gradcam_alpha * gradcam_img, 0, 1)
    return overlay

def analyze_classes_with_gradcam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_NAMES)
    model = EdgeOtoscopeNet(num_classes=num_classes)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cam = GradCAM(model=model, target_layers=[get_target_layer(model)])

    fig, axes = plt.subplots(len(CLASS_NAMES), 4, figsize=(20, 4 * len(CLASS_NAMES)))
    fig.suptitle('GradCAM Analysis for All Classes', fontsize=18, fontweight='bold', y=0.98)

    axes[0, 0].set_title('Class Information', fontsize=14, fontweight='bold', pad=20)
    axes[0, 1].set_title('Original Image', fontsize=14, fontweight='bold', pad=20)
    axes[0, 2].set_title('GradCAM Heatmap', fontsize=14, fontweight='bold', pad=20)
    axes[0, 3].set_title('Overlay Visualization', fontsize=14, fontweight='bold', pad=20)

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(TEST_DIR, class_name)
        image_path = get_random_image_from_class(class_dir)

        if image_path is None:
            for col in range(4):
                axes[class_idx, col].text(0.5, 0.5, 'No Images Found', ha='center', va='center', transform=axes[class_idx, col].transAxes)
                axes[class_idx, col].axis('off')
            continue

        pil_img = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, pred_idx = torch.max(probabilities, 0)

        predicted_class = CLASS_NAMES[pred_idx.item()]
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        pil_img_resized = pil_img.resize((224, 224))
        rgb_img = np.array(pil_img_resized) / 255.0

        gradcam_visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        overlay_img = create_overlay_image(rgb_img, gradcam_visualization, NORMAL_ALPHA, GRADCAM_ALPHA)

        axes[class_idx, 0].axis('off')
        info_text = f"TRUE CLASS:\n{class_name}\n\nPREDICTED CLASS:\n{predicted_class}"
        text_color = 'green' if class_name == predicted_class else 'red'
        
        axes[class_idx, 0].text(0.05, 0.95, info_text, transform=axes[class_idx, 0].transAxes, 
                                fontsize=11, fontweight='bold', verticalalignment='top', color=text_color, 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))

        axes[class_idx, 1].imshow(rgb_img)
        axes[class_idx, 1].axis('off')
        axes[class_idx, 2].imshow(gradcam_visualization)
        axes[class_idx, 2].axis('off')
        axes[class_idx, 3].imshow(overlay_img)
        axes[class_idx, 3].axis('off')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.07, wspace=0.15, hspace=0.3)
    plt.show()

if __name__ == '__main__':
    analyze_classes_with_gradcam()