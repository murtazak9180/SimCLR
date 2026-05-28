from torchvision.models import resnet18
import torch.nn as nn
from torchvision import datasets, transforms
import torch
from bscs22030_05_task2_augmentations import transform, TwoViewTransform
import torch.nn.functional as F
from pathlib import Path



#define the paths
home = Path.home()
root = home / "Documents" / "University" / "DL" / "SimCLR"
splits = root / "splits"
train_path = splits / "train_labeled_10percent.txt"
val_path = splits / "val.txt"
test_path = splits / "test.txt"
data_path = root / "data"


N = 64


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
#get the 512 dimensional embedding
model.fc = nn.Identity()

#load an image
#get an augmentated pair 

#append to the batch. 

#repeat for N images say 64. 

base_dataset = datasets.CIFAR10(root="data", train=True, download=False)
two_view = TwoViewTransform(transform)

batch_a = []
batch_b = []
model.eval()
for i in range(N):
    img, label = base_dataset[i]
    view1, view2 = two_view(img)
    view1, view2 = view1.unsqueeze(0), view2.unsqueeze(0)
    emb1 = model(view1)
    emb2 = model(view2)

    
    batch_a.append(emb1)
    batch_b.append(emb2)



# cosine similarity between positive pairs
same_similarity = 0.0

for i in range(N):
    sim = F.cosine_similarity(batch_a[i], batch_b[i], dim=1)
    same_similarity += sim.item()

same_similarity /= N


# cosine similarity between negative pairs
diff_similarity = 0.0
num_combs = 0

for i in range(N):
    for j in range(N):
        if i != j:
            sim = F.cosine_similarity(batch_a[i], batch_b[j], dim=1)
            diff_similarity += sim.item()
            num_combs += 1

diff_similarity /= num_combs


print(f"Same: {same_similarity:.4f} | Different: {diff_similarity:.4f}")





