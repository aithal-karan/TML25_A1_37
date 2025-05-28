import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision.transforms import Normalize
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
transform = Normalize(mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889])

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

# === Feature extraction ===
def extract_features(loader, with_membership=False):
    all_features, all_membership, all_ids = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            if with_membership:
                ids, imgs, labels, is_member = batch
            else:
                ids, imgs, labels = batch

            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)

            loss = F.cross_entropy(logits, labels, reduction='none')
            confidence = probs.max(dim=1).values
            entropy = (-probs * torch.log(probs + 1e-8)).sum(dim=1)
            margin = probs.topk(2, dim=1).values
            margin = (margin[:, 0] - margin[:, 1])
            logit_std = logits.std(dim=1)
            logit_max = logits.max(dim=1).values

            features = torch.stack([
                loss.cpu(),
                confidence.cpu(),
                entropy.cpu(),
                margin.cpu(),
                logit_std.cpu(),
                logit_max.cpu()
            ], dim=1)

            all_features.append(features)
            all_ids.extend([int(i) if isinstance(i, torch.Tensor) else i for i in ids])
            if with_membership:
                all_membership.extend(is_member.cpu().numpy())

    X = torch.cat(all_features).numpy()
    return (X, all_ids, all_membership) if with_membership else (X, all_ids)

# === Public (Shadow) Data
X_pub, pub_ids, y_pub = extract_features(public_loader, with_membership=True)

# === Normalize for KDE
scaler_kde = StandardScaler()
X_pub_scaled = scaler_kde.fit_transform(X_pub)

X_mem = X_pub_scaled[np.array(y_pub) == 1]
X_non = X_pub_scaled[np.array(y_pub) == 0]

kde_mem = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(X_mem)
kde_non = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(X_non)

# === Train RF (no calibration)
clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
clf.fit(X_pub, y_pub)

# === Private (Target) Data
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

X_priv, priv_ids = extract_features(private_loader)
X_priv_scaled = scaler_kde.transform(X_priv)

# === KDE Scores
log_p_mem = kde_mem.score_samples(X_priv_scaled)
log_p_non = kde_non.score_samples(X_priv_scaled)
kde_score = log_p_mem - log_p_non

# === RF Scores
rf_score = clf.predict_proba(X_priv)[:, 1]

# === Normalize and Ensemble
norm = MinMaxScaler()
rf_score_norm = norm.fit_transform(rf_score.reshape(-1, 1)).flatten()
kde_score_norm = norm.fit_transform(kde_score.reshape(-1, 1)).flatten()

final_score = (rf_score_norm + kde_score_norm) / 2

# === Save
df = pd.DataFrame({
    "ids": priv_ids,
    "score": final_score
})
df.to_csv("test_kde_hybrid.csv", index=False)
print("Saved hybrid ensemble predictions to `test_kde_hybrid.csv`")

import pandas as pd
from scipy.stats import rankdata

df = pd.read_csv("test_kde_hybrid.csv")  
raw_scores = df["score"].values
stretched_score = rankdata(raw_scores, method="average") / len(raw_scores)

df["score"] = stretched_score
df.to_csv("test_kde_hybrid.csv", index=False)
print("scores saved to `test_kde_hybrid.csv`")

