import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from ptflops import get_model_complexity_info
from model import EdgeOtoscopeNet

def get_data_loaders(data_dir, batch_size=32):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), test_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    return train_loader, val_loader, test_loader, class_names

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    return running_loss / total_samples, correct_predictions / total_samples

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    return running_loss / total_samples, correct_predictions / total_samples

def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

if __name__ == '__main__':
    DATA_DIR = './dataset'
    MODEL_SAVE_PATH = 'edge_otoscope_net_best.pth'
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Dataset directory '{DATA_DIR}' not found. Please provide valid path.")

    train_loader, val_loader, test_loader, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)
    NUM_CLASSES = len(class_names)
    model = EdgeOtoscopeNet(num_classes=NUM_CLASSES).to(device)

    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False, print_per_layer_stat=False, verbose=False)
    print(f"Computational complexity: {(macs * 2) / 1e9:.2f} GFLOPs")
    print(f"Number of parameters: {params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0.00009)

    best_val_acc, best_val_loss, early_stopping_counter = 0.0, float('inf'), 0
    train_acc_history, val_acc_history, train_loss_history, val_loss_history = [], [], [], []

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, best_val_loss, early_stopping_counter = val_acc, val_loss, 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        elif val_acc == best_val_acc and val_loss < best_val_loss:
            best_val_loss, early_stopping_counter = val_loss, 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    test_labels, test_preds, test_probs = get_predictions(model, test_loader, device)
    print(f"\nTest Accuracy: {np.mean(test_preds == test_labels):.4f}")
    print(classification_report(test_labels, test_preds, target_names=class_names))