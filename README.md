# TML25_A1_37
TML25_A1_37 Assignment 1 Membership Inference Attack

## Membership Inference Attack — TML Assignment 1 (SS 2025)

This repository contains our implementation for the Membership Inference Attack (MIA) task. The objective is to determine whether a given data point was part of the training data used to train a ResNet-18 model (`01_MIA.pt`).

Our attack relies on extracting confidence-based features from the model outputs and training a classifier to distinguish between members and non-members.

---

## Objective

Perform a membership inference attack on a black-box ResNet18 model using a public dataset with known membership labels and infer membership for an unseen private dataset.

---

## Methods Implemented

We implemented and compared 3 methods:

## Dependencies

pip install torch torchvision pandas numpy tqdm scikit-learn


##  `multi_mia.py` (Code 1)
- Extracts 4 features from model predictions:
  - Cross-entropy loss
  - Softmax confidence
  - Entropy
  - Margin (top-1 - top-2 confidence)
- Trains a `RandomForestClassifier` on `pub.pt`
- Applies inference to `priv_out.pt`

## `kde_hybrid.py` (Code 2)
- Extends Code 1 with 2 additional features:
  - Logit standard deviation
  - Max logit value
- Applies Kernel Density Estimation (KDE) for likelihood ratio-style scoring
- Combines RF and KDE scores using a hybrid ensemble
- Applies quantile stretching for improved score separation

## `lira_kde.py` (LiRA-style KDE Attack) (Code 3)
- **Shadow Model Training**: We trained 5 shadow models on disjoint subsets of `pub.pt`, with an 80/20 split for member/non-member partitioning.
- **Feature Extraction**: From each model, we extracted:
  - Cross-entropy loss
  - Confidence
  - Entropy
  - Margin (top-1 vs. top-2 probability)
  - Logit standard deviation
  - Maximum logit value
- **Density Modeling**: Using `GridSearchCV`, we fit separate **Kernel Density Estimators (KDEs)** on standardized features for member and non-member groups.
- **Score Computation**: We computed **log-likelihood ratios** for each private sample as:
                                                                `log(P_in(x) / P_out(x))`
         where `P_in` and `P_out` represent the probability densities for member and non-member samples, respectively.

- **Calibration**: The scores were normalized using MinMax scaling and saved as the final prediction file.

---

## Final Evaluation Scores

| Model           | TPR@FPR=0.05    | AUC        |
|-----------------|-----------------|------------|
| `multi_mia.py`  | **0.0707**      | **0.6440** |
| `kde_hybrid.py` | **0.0707**      | **0.6407** |
| `lira_kde.py`   | **0.0323**      | **0.5121** |  

Both models performed equally well in terms of TPR@FPR and AUC.

---

## Important Files

| File                    | Description                                               |
|-------------------------|-----------------------------------------------------------|
| `multi_mia.py`          | Baseline MIA using 4 handcrafted features + RF classifier |
| `kde_hybrid.py`         | Hybrid KDE + RF-based MIA with quantile stretching        |
| `01_MIA.pt`             | Pretrained ResNet18 model (target)                        |
| `pub.pt`                | Public dataset with membership labels                     |
| `priv_out.pt`           | Private dataset (membership unknown)                      |
| `lira_kde.py`           | Multivariate KDE-based shadow attack (LiRA-inspired)      |
| `test_multi_mia.csv`    | Output from multi MIA attack                              |
| `test_lira_kde.csv`     | Output from final LiRA KDE attack                         |
| `test_kde_hybrid.csv`   | Output from KDE Hybrid attack                             |

---

##  How to Run

Ensure the model and datasets are in the same directory. Then run:

```bash
python multi_mia.py
python kde_hybrid.py
python lira_kde.py
