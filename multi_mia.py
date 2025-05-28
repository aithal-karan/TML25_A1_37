import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision.transforms import Normalize
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# === Dataset Definitions ===
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return id_, img, label

    def __len__(self):
        return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]
transform = Normalize(mean=mean, std=std)

# === Load model ===
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 44)
model.load_state_dict(torch.load("01_MIA.pt", map_location=device))
model.to(device)
model.eval()

# === Load and filter public data ===
public_data: MembershipDataset = torch.load("pub.pt")
public_data.transform = transform
valid_indices = [
    i for i in range(len(public_data))
    if public_data.labels[i] is not None and
       public_data.imgs[i] is not None and
       public_data.membership[i] is not None
]

class FilteredPublicDataset(MembershipDataset):
    def __init__(self, dataset, indices):
        super().__init__(dataset.transform)
        self.ids = [dataset.ids[i] for i in indices]
        self.imgs = [dataset.imgs[i] for i in indices]
        self.labels = [dataset.labels[i] for i in indices]
        self.membership = [dataset.membership[i] for i in indices]

public_data = FilteredPublicDataset(public_data, valid_indices)
public_loader = DataLoader(public_data, batch_size=64, shuffle=False)

# === Extract features from public data ===
public_features = []
public_membership = []

with torch.no_grad():
    for ids, imgs, labels, is_member in tqdm(public_loader, desc="Public"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, labels, reduction='none')
        confidence = probs.max(dim=1).values
        entropy = (-probs * torch.log(probs + 1e-8)).sum(dim=1)
        margin = probs.topk(2, dim=1).values
        margin = (margin[:, 0] - margin[:, 1])
        features = torch.stack([
            loss.cpu(),
            confidence.cpu(),
            entropy.cpu(),
            margin.cpu()
        ], dim=1)
        public_features.append(features)
        public_membership.extend(is_member.cpu().numpy())

X_public = torch.cat(public_features).numpy()
y_public = np.array(public_membership)

# === Train attack model ===
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

#clf = LogisticRegression()
clf.fit(X_public, y_public)

# === Load and filter private data ===
private_data: TaskDataset = torch.load("priv_out.pt")
private_data.transform = transform
valid_indices = [
    i for i in range(len(private_data))
    if private_data.labels[i] is not None and
       private_data.imgs[i] is not None
]

class FilteredPrivateDataset(TaskDataset):
    def __init__(self, dataset, indices):
        super().__init__(dataset.transform)
        self.ids = [dataset.ids[i] for i in indices]
        self.imgs = [dataset.imgs[i] for i in indices]
        self.labels = [dataset.labels[i] for i in indices]

private_data = FilteredPrivateDataset(private_data, valid_indices)
private_loader = DataLoader(private_data, batch_size=64, shuffle=False)

# === Extract features from private data ===
private_features = []
private_ids = []

with torch.no_grad():
    for ids, imgs, labels in tqdm(private_loader, desc="Private"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, labels, reduction='none')
        confidence = probs.max(dim=1).values
        entropy = (-probs * torch.log(probs + 1e-8)).sum(dim=1)
        margin = probs.topk(2, dim=1).values
        margin = (margin[:, 0] - margin[:, 1])
        features = torch.stack([
            loss.cpu(),
            confidence.cpu(),
            entropy.cpu(),
            margin.cpu()
        ], dim=1)
        private_features.append(features)
        private_ids.extend([int(i) if isinstance(i, torch.Tensor) else i for i in ids])

X_private = torch.cat(private_features).numpy()
membership_scores = clf.predict_proba(X_private)[:, 1]

# === Save to test.csv ===
df = pd.DataFrame({
    "ids": private_ids,
    "score": membership_scores
})
df.to_csv("test_multi_mia.csv", index=False)
print("test_multi_mia.csv saved")
