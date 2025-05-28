# lira_kde.py
# LiRA-inspired KDE Membership Inference Attack (Multivariate, Optimized)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet18
from torchvision.transforms import Normalize
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import random
import os

# === Configuration ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_SHADOWS = 5
EPOCHS = 25
TRAIN_FRAC = 0.8
normalize = Normalize(mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889])

# === Dataset Class ===
class MembershipDataset(Dataset):
    def __init__(self):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.membership = []

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if img is not None and label is not None:
            img = normalize(img)
        return img, label, self.membership[index]

    def __len__(self):
        return len(self.ids)

# === Load and shuffle pub.pt ===
pub = torch.load("pub.pt")
all_indices = list(range(len(pub)))
random.shuffle(all_indices)
samples_per_model = len(all_indices) // NUM_SHADOWS

features, labels = [], []

# === Shadow model training ===
def train_shadow_model(dataset):
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 44)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for x, y, _ in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()

# === Feature extraction (all 6 features) ===
def extract_features(model, loader):
    all_feat, all_labels = [], []
    with torch.no_grad():
        for x, y, m in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, y, reduction='none').cpu()
            confidence = probs.max(dim=1)[0].cpu()
            entropy = (-probs * torch.log(probs + 1e-8)).sum(dim=1).cpu()
            margin = (probs.topk(2, dim=1).values[:, 0] - probs.topk(2, dim=1).values[:, 1]).cpu()
            logit_std = logits.std(dim=1).cpu()
            logit_max = logits.max(dim=1)[0].cpu()
            features = torch.stack([loss, confidence, entropy, margin, logit_std, logit_max], dim=1)
            all_feat.append(features)
            all_labels.extend(m)
    return torch.cat(all_feat).numpy(), np.array(all_labels)

# === Train shadow models and collect member/non-member data ===
X_mem, X_non = [], []
for i in range(NUM_SHADOWS):
    idx = all_indices[i * samples_per_model:(i + 1) * samples_per_model]
    mid = int(len(idx) * TRAIN_FRAC)
    train_idx, test_idx = idx[:mid], idx[mid:]

    dataset = Subset(pub, train_idx + test_idx)
    for j, k in zip(train_idx, test_idx):
        pub.membership[j] = 1
        pub.membership[k] = 0

    model = train_shadow_model(Subset(pub, train_idx))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    feats, memb = extract_features(model, loader)
    X_mem.extend(feats[memb == 1])
    X_non.extend(feats[memb == 0])

X_mem, X_non = np.array(X_mem), np.array(X_non)
scaler = StandardScaler()
X_mem_scaled = scaler.fit_transform(X_mem)
X_non_scaled = scaler.transform(X_non)

# === KDE Grid Search for Bandwidth ===
def best_kde(X):
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': np.logspace(-1, 1, 20)}, cv=3)
    grid.fit(X)
    return grid.best_estimator_

kde_mem = best_kde(X_mem_scaled)
kde_non = best_kde(X_non_scaled)

# === Load and preprocess priv_out.pt ===
priv = torch.load("priv_out.pt")
valid_idx = [i for i in range(len(priv)) if priv.imgs[i] is not None and priv.labels[i] is not None]

class FilteredPriv(Dataset):
    def __init__(self, base, idx):
        self.ids = [base.ids[i] for i in idx]
        self.imgs = [base.imgs[i] for i in idx]
        self.labels = [base.labels[i] for i in idx]

    def __getitem__(self, index):
        return self.ids[index], normalize(self.imgs[index]), self.labels[index]

    def __len__(self):
        return len(self.ids)

priv_filtered = FilteredPriv(priv, valid_idx)
loader = DataLoader(priv_filtered, batch_size=BATCH_SIZE, shuffle=False)

model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 44)
model.load_state_dict(torch.load("01_MIA.pt", map_location=DEVICE))
model.to(DEVICE).eval()

final_ids, features = [], []
with torch.no_grad():
    for ids, x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, y, reduction='none').cpu()
        confidence = probs.max(dim=1)[0].cpu()
        entropy = (-probs * torch.log(probs + 1e-8)).sum(dim=1).cpu()
        margin = (probs.topk(2, dim=1).values[:, 0] - probs.topk(2, dim=1).values[:, 1]).cpu()
        logit_std = logits.std(dim=1).cpu()
        logit_max = logits.max(dim=1)[0].cpu()
        batch_features = torch.stack([loss, confidence, entropy, margin, logit_std, logit_max], dim=1)
        features.append(batch_features)
        final_ids.extend([int(i) for i in ids])

X_priv = torch.cat(features).numpy()
X_priv_scaled = scaler.transform(X_priv)

log_p_mem = kde_mem.score_samples(X_priv_scaled)
log_p_non = kde_non.score_samples(X_priv_scaled)
llr_scores = log_p_mem - log_p_non

# === Normalize & Save
stretched_scores = MinMaxScaler().fit_transform(llr_scores.reshape(-1, 1)).flatten()
df = pd.DataFrame({"ids": final_ids, "score": stretched_scores})
df.to_csv("test_lira_kde.csv", index=False)
print("LiRA KDE scores saved to test_lira_kde.csv")
