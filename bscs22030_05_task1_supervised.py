from torchvision import transforms 
from utils.dataset_splits import get_cifar10_subset
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim 
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random



#define the paths
home = Path.home()
root = home / "Documents" / "University" / "DL" / "SimCLR"
splits = root / "splits"
train_path = splits / "train_labeled_10percent.txt"
val_path = splits / "val.txt"
test_path = splits / "test.txt"
data_path = root / "data"

seed = 2026
lr = 0.0003
epochs = 30



random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


#DEFINE THE TRAIN AND THE TEST TRANSFORMS 
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(), 

    transforms.ToTensor(),

   transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616)
)
])

test_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616)
)
])



#Get the dataset using the defined function in dataset_splits
train_dataset = get_cifar10_subset(data_root=data_path, split_file=train_path, train=True, transform=train_transform)
val_dataset = get_cifar10_subset(data_root=data_path, split_file=val_path, train=True,transform=test_transform)
test_dataset = get_cifar10_subset(data_root=data_path, split_file=test_path, train=False, transform=test_transform)


#dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=64,shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64,shuffle=False)



#load resnet18 random init
model = resnet18(weights=None)

#change conv1
model.conv1 = nn.Conv2d(
    in_channels = 3,
    out_channels = 64, 
    kernel_size = 3, 
    stride = 1, 
    padding = 1,
    bias = False
)

#reomve max pool
model.maxpool = nn.Identity()

#change the fc layer
num_features = model.fc.in_features
model.fc = nn.Linear(
    in_features = num_features,
    out_features = 10
)



#train loop
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)  

train_losses = []
val_losses = []
val_accuracies = []

best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()  
        optimizer.step()
        running_train_loss += loss.item() * images.size(0)
        
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # Validation Phase
    model.eval()
    running_val_loss = 0.0
    correct_val_preds = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            loss = criterion(output, labels)
            
            running_val_loss += loss.item() * images.size(0)
            
            preds = output.argmax(dim=1)
            correct_val_preds += (preds == labels).sum().item()
            
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_val_acc = (correct_val_preds / len(val_loader.dataset)) * 100
    
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)
    
    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Train Loss: {epoch_train_loss:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} | "
          f"Val Acc: {epoch_val_acc:.2f}%")
    
    # Checkpoint
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), "models/supervised_model.pt")
        print(f" => New best validation checkpoint saved with accuracy: {best_val_acc:.2f}%")

print("\nTraining complete. Generating evaluation analytics...")

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color="royalblue")
plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", color="crimson")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Task 1: Supervised Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("graphs/supervised_loss.png", bbox_inches="tight")
plt.close()

# Reload state dict weights to reflect the optimized checkpoint parameters
model.load_state_dict(torch.load("models/supervised_model.pt"))
model.eval()

all_test_preds = []
all_test_labels = []
correct_test_preds = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        
        preds = output.argmax(dim=1)
        correct_test_preds += (preds == labels).sum().item()
        
        # Collect raw values back to host CPU for scikit-learn analytics
        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

final_test_acc = (correct_test_preds / len(test_loader.dataset)) * 100
print(f"\nFinal Test Set Execution Performance Accuracy: {final_test_acc:.2f}%")

cm = confusion_matrix(all_test_labels, all_test_preds)

# CIFAR-10 Standard Class Names for clarity
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=cifar10_classes, 
    yticklabels=cifar10_classes
)
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.title(f"Task 1: Supervised Confusion Matrix (Test Acc: {final_test_acc:.2f}%)")
plt.savefig("results/supervised_confusion_matrix.png", bbox_inches="tight")
plt.close()

