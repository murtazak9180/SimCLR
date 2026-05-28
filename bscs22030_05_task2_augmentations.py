from torchvision import transforms 
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

seed = 2026

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2,1.0)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4,contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616))
]
)



class TwoViewTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)

        return view1, view2


def unnormalize_to_numpy(tensor):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    # Move channel dim from [C, H, W] to [H, W, C] for matplotlib
    np_img = tensor.numpy().transpose((1, 2, 0))
    # Reverse the normalization math: (Tensor * std) + mean
    np_img = (np_img * std) + mean
    # Clip values to ensure they stay strictly inside the valid [0, 1] color bounds
    np_img = np.clip(np_img, 0, 1)
    return np_img

if __name__ == '__main__':
    base_dataset = datasets.CIFAR10(root="data", train=True, download=False)
    
    # Randomly select 10 unique image indices
    random_indices = random.sample(range(len(base_dataset)), 10)
    
    fig, axes = plt.subplots(10, 3, figsize=(9, 20))
    
    axes[0, 0].set_title("Original Image", fontsize=12, pad=10)
    axes[0, 1].set_title("Augmented View 1", fontsize=12, pad=10)
    axes[0, 2].set_title("Augmented View 2", fontsize=12, pad=10)
    two_view = TwoViewTransform(transform)
    
    for row_idx, img_idx in enumerate(random_indices):
        # Fetch raw PIL image and its class label
        original_pil, _ = base_dataset[img_idx]
        view1_tensor, view2_tensor = two_view(original_pil)
        
        view1_img = unnormalize_to_numpy(view1_tensor)
        view2_img = unnormalize_to_numpy(view2_tensor)
        
        axes[row_idx, 0].imshow(original_pil)
        axes[row_idx, 1].imshow(view1_img)
        axes[row_idx, 2].imshow(view2_img)
        
        for col_idx in range(3):
            axes[row_idx, col_idx].axis('off')
            
    plt.tight_layout()
    plt.savefig("results/augmentation_examples.png", bbox_inches="tight", dpi=150)
    plt.close()
    
    print("Success! Generated results/augmentation_examples.png showing 10 sample pipelines.")
