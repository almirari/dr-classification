# Lightweight Diabetic Retinopathy Grading via Heterogeneous Multi-Teacher Knowledge Distillation

Grading diabetic retinopathy (DR) severity from retinal fundus photographs with a **1.5 M-parameter MobileNetV3-Small student** distilled from larger **nominal (cross-entropy)** and **ordinal (CORAL / CORN / regression)** teachers.

The central question of this repo: *the DR grades (No DR → Proliferative DR) are both categories and an ordered scale. Can a small model inherit both views from teachers trained with different loss families, and how should those views be combined?*

- **Dataset:** APTOS 2019 Blindness Detection (5 classes, 3,662 labelled images)
- **Student:** MobileNetV3-Small (≈ 1.52 M params, ≈ 5.9 MB)
- **Teachers:** ResNet-50 (23.5 M) and EfficientNet-B3 (10.7 M)
- **Protocol:** 3 seeds (`33, 81, 5`) × 100 epochs per experiment, model selection by validation QWK, results reported on a held-out test split
- **Primary metric:** Quadratic Weighted Kappa (QWK), reported with Accuracy, MAE, RMSE/MSE and macro/weighted F1 for every run. One-vs-rest AUC-ROC is computed only in the dual-head notebooks.

---

## Table of contents

1. [Headline results](#1-headline-results)
2. [Repository layout](#2-repository-layout)
3. [Data pipeline](#3-data-pipeline)
4. [Models](#4-models)
5. [Methods and formulas](#5-methods-and-formulas)
   - [5.1 Evaluation metrics](#51-evaluation-metrics)
   - [5.2 Output heads and losses](#52-output-heads-and-losses)
   - [5.3 Ordinal outputs → class probabilities](#53-ordinal-outputs--class-probabilities)
   - [5.4 Single-teacher KD](#54-single-teacher-kd)
   - [5.5 Heterogeneous multi-teacher KD (CA-MKD hybrid)](#55-heterogeneous-multi-teacher-kd-ca-mkd-hybrid)
   - [5.6 Dual-head student with fusion gate (DP-KD)](#56-dual-head-student-with-fusion-gate-dp-kd)
6. [Training protocol](#6-training-protocol)
7. [Full results](#7-full-results)
8. [Observations](#8-observations)
9. [Reproducing the experiments](#9-reproducing-the-experiments)
10. [Known caveats](#10-known-caveats)
11. [References](#11-references)

---

## 1. Headline results

Test split (367 images), mean over 3 seeds. Standard deviations are in [Section 7](#7-full-results).

| Model | Teacher(s) | QWK ↑ | Acc (%) ↑ | MAE ↓ |
|---|---|---:|---:|---:|
| Standalone CE | none | 0.8590 | 77.66 | 0.2916 |
| Standalone CORAL | none | 0.8708 | 76.75 | 0.2834 |
| Single-teacher KD (CE) | ResNet-50-CE | 0.8847 | 83.29 | 0.2252 |
| Single-teacher KD (CORAL) | ResNet-50-CORAL | 0.8867 | 81.38 | 0.2352 |
| Multi-teacher KD, nominal | ResNet-50-CE + EffNet-B3-CE | 0.8721 | 81.93 | 0.2498 |
| Multi-teacher KD, ordinal | ResNet-50-CORAL + EffNet-B3-CORAL | 0.8834 | 79.02 | 0.2579 |
| **CA-MKD-style hybrid** | **ResNet-50-CE + EffNet-B3-CORAL** | **0.8907** | **84.11** | **0.2125** |
| Dual-head DP-KD, learned gate (MSE head) | ResNet-50-CE + EffNet-B3-MSE | 0.8895 | 83.11 | 0.2216 |

For reference, the teachers are far larger and slower:

| | Params | Size | Latency |
|---|---:|---:|---:|
| ResNet-50 | 23.5 M | 90.0 MB | ≈ 70–99 ms |
| EfficientNet-B3 | 10.7 M | 41.4 MB | ≈ 58–78 ms |
| **MobileNetV3-Small student** | **1.52 M** | **5.9 MB** | **≈ 6–7 ms** |

Latency numbers come from the evaluation scripts in `05_results/`, which run every model on **CPU** with the same settings. The CPU model is not recorded, so compare them relatively, not absolutely.

> **Read the differences with care.**
> - Every comparison uses 3 seeds and a 367-image test set, and most gaps between the KD variants are smaller than the seed-to-seed standard deviation.
> - The CA-MKD hybrid row (the only bold one) and the MSE dual-head row were each picked as the best of their family *by test QWK*, so both are optimistically biased. Selecting on validation and reporting test once would be cleaner.
> - The single-teacher CORAL row uses a different KD loss from the other KD rows (no temperature, α = 0.3; see §5.4), so "single vs. multi-teacher" comparisons also change the loss.
>
> See [Known caveats](#10-known-caveats).

---

## 2. Repository layout

```
dr-classification/
├── 00_data/
│   ├── 02_stratifiedsplit.ipynb        # stratified 80/10/10 split + crop + Ben Graham preprocessing
│   ├── 03_dataaugmentation.ipynb       # offline augmentation (5 aug + 1 MixUp per training image)
│   └── aptos_2019_splitted/            # train/val/test CSVs (train_split.csv is the augmented 20,503-row version incl. fractional MixUp labels)
├── 01_baseline/                        # standalone MobileNetV3-Small: CE, CORAL, CORN, Niu-style
│   ├── standalone_{ce,coral,corn,niu}_mobilenetv3-small.ipynb
│   └── models/                         # checkpoints, dashboards
├── 02_hypertuning/                     # staged search (lr/wd → batch/dropout → T/α/β)
│   ├── hyperparametersearch.ipynb
│   └── checkpoints/phase_*.json
├── 03_kd_single/                       # single-teacher KD from ResNet-50 (CE and CORAL)
│   ├── kd_ce_resnet50_mobilenetv3-small.ipynb
│   ├── kd_coral_resnet50_mobilenetv3-small.ipynb
│   └── kd_masterfile_hybrid_fixed_mse.ipynb
├── 04_kd_multi/
│   ├── 01_mkd/                         # multi-teacher KD, single-head student
│   │   ├── mkd_homogenous_resnet50_efficientnetb3_coral.ipynb
│   │   ├── mkd_hoomogenous_resnet50_efficientnetb3_ce.ipynb
│   │   ├── mkd_hybrid_resnet50_efficientnetb3_coral.ipynb   # CE + CORAL teachers (best QWK)
│   │   ├── mkd_hybrid_resnet50_efficientnetb3_corn.ipynb    # CE + CORN teachers
│   │   └── mkd_hybrid_resnet50_resnet50_coral.ipynb
│   ├── 02_dualheaded_hybrid/           # two-head student, probability-space fusion
│   │   ├── {learned,fixed,relative}_{coral,corn,mse}.ipynb
│   └── 03_dualheaded_logit_feature/    # dual-head + feature distillation (β > 0)
│       └── {coral,corn}.ipynb
├── 05_results/                         # eval scripts, per-run text reports, CSVs, summaries
│   ├── eval_{baseline,singlekd,multikd}.py
│   ├── *_results.csv, *_summary.txt, *_test_results.txt
│   └── results.ipynb
└── 06_analysis/
    └── statisticalanalysis.ipynb       # paired t-test, Wilcoxon, Cohen's d
```

The notebooks were written for **Kaggle** (GPU, `/kaggle/input/...`, `/kaggle/working/`). The scripts in `05_results/` run locally against the saved checkpoints.

---

## 3. Data pipeline

### 3.1 Split

APTOS 2019 `train.csv` (3,662 images) is split **stratified by diagnosis** with `random_state=55`:

| Split | Fraction | Images (original) |
|---|---:|---:|
| Train | 80 % | 2,929 |
| Validation | 10 % | 366 |
| Test | 10 % | 367 |

Validation and test are never augmented.

### 3.2 Preprocessing

Each image is cropped to its retina (largest contour above intensity 10, kept only if it covers more than 10 % of both dimensions), resized to $224\times224$ and enhanced with **Ben Graham's local-contrast normalisation**:

$$
I' = 4\,I \;-\; 4\,(G_\sigma * I) \;+\; 128,\qquad \sigma = \frac{\text{size}}{30}
$$

where $G_\sigma$ is a Gaussian blur and `cv2.addWeighted` clips the result to $[0,255]$. In the code `size` is the 224 px side length, so $\sigma\approx7.5$ px. As far as I recall, Graham's published code uses $\sigma=\text{radius}/30$, which would be about half that, so the blur here is roughly twice as wide relative to the image.

### 3.3 Offline augmentation (training split only)

Every training image yields **7 images**: the original, 5 augmented copies and 1 MixUp image, so 2,929 → **20,503** training images.

- **Spatial/colour augmentations (Albumentations):** horizontal/vertical flip ($p=0.5$), rotation up to 360° ($p=0.9$), shift ±5 % and scale ±10 % ($p=0.5$), hue/saturation/value jitter ($p=0.7$), brightness/contrast ±0.2 ($p=0.7$), grid distortion and optical distortion ($p=0.4$ each).
- **MixUp** ($\alpha=0.4$) with a random partner that is never the image itself:

$$
\lambda \sim \mathrm{Beta}(\alpha,\alpha),\qquad
\tilde x = \lambda x_a + (1-\lambda)\,x_b,\qquad
\tilde y = \lambda\, y_a + (1-\lambda)\,y_b
$$

Augmentation is seeded deterministically per image, so the augmented set is reproducible.

**Label truncation (important).** The MixUp rows are written to the CSV with fractional labels, and the training notebooks read them with `int(...)` / `astype(int)`, which truncates towards zero (for example 2.7 → 2). Recomputing from `00_data/aptos_2019_splitted/train_split.csv` gives exactly the class counts in §3.4 after truncation, so this is what the models were trained on:

- 1,985 of the 2,929 MixUp rows have a fractional label.
- 44.9 % of MixUp images are trained with a label different from their own source image's label (32.0 % lowered, 12.9 % raised).
- Of the 236 MixUp images whose source image is Proliferative DR, 216 are labelled below Proliferative DR, so only 20 MixUp images carry that label.

### 3.4 Class distribution after augmentation (train, as the loaders see it)

| Class | Name | Images | Share |
|---:|---|---:|---:|
| 0 | No DR | 10,157 | 49.5 % |
| 1 | Mild | 2,494 | 12.2 % |
| 2 | Moderate | 5,279 | 25.7 % |
| 3 | Severe | 1,137 | 5.5 % |
| 4 | Proliferative DR | 1,436 | 7.0 % |

These counts are after the truncation described in §3.3. The 2,929 original images contribute 1,444 / 296 / 799 / 154 / 236 images per class (each appears 6 times with its 5 augmentations), and the MixUp images add 1,493 / 718 / 485 / 213 / 20.

The imbalance is handled with a `WeightedRandomSampler` using weights $w_c = 1/n_c$ (inverse class frequency, sampling with replacement). Inputs are normalised with ImageNet mean/std.

---

## 4. Models

| Role | Backbone | Pooled feature dim | Params |
|---|---|---:|---:|
| Student | MobileNetV3-Small (ImageNet-pretrained) | 1024 | ≈ 1.52 M |
| Teacher A | ResNet-50 (ImageNet-pretrained) | 2048 | ≈ 23.5 M |
| Teacher B | EfficientNet-B3 (ImageNet-pretrained) | 1536 | ≈ 10.7 M |

All networks are trained at $224\times224$. Teachers are fine-tuned with the same optimiser settings as the students and are **frozen** during distillation.

The student's final classifier layer is replaced by one of:

| Head | Output | Used by |
|---|---|---|
| Nominal | $K=5$ logits | CE baseline, single/multi-teacher KD, dual-head (nominal branch) |
| CORAL | $K-1=4$ logits, shared weight vector + per-threshold bias | CORAL baselines/teachers, dual-head |
| CORN | $K-1=4$ conditional logits, independent weights | CORN baseline/teachers, dual-head |
| Niu-style | $K-1$ independent linear heads | Niu-style baseline |
| Regression | 1 scalar score in class-index space | MSE branch of the dual-head model |

---

## 5. Methods and formulas

Notation: $K=5$ classes, input $x$, label $y\in\{0,\dots,4\}$, student logits $z^s$, teacher logits $z^t$, temperature $T$, softmax $\sigma_{\text{sm}}$, sigmoid $\sigma$.

### 5.1 Evaluation metrics

**Quadratic Weighted Kappa** (model-selection metric):

$$
\kappa = 1 - \frac{\sum_{i,j} w_{ij}\,O_{ij}}{\sum_{i,j} w_{ij}\,E_{ij}},\qquad
w_{ij} = \frac{(i-j)^2}{(K-1)^2}
$$

$O$ is the observed confusion matrix and $E$ the matrix expected by chance from the marginals. QWK penalises a prediction by the *squared grade distance*, which is why it suits an ordered scale.

**Error metrics** on predicted grade $\hat y$:

$$
\mathrm{MAE}=\frac1N\sum_n|y_n-\hat y_n|,\qquad
\mathrm{MSE}=\frac1N\sum_n(y_n-\hat y_n)^2,\qquad
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}
$$

**F1** is reported both macro-averaged (each class counts equally, so it is sensitive to the rare Severe/Proliferative classes) and support-weighted:

$$
\mathrm{F1}_c = \frac{2\,P_c R_c}{P_c+R_c},\qquad
\mathrm{F1}_{\text{macro}}=\frac1K\sum_c \mathrm{F1}_c,\qquad
\mathrm{F1}_{\text{weighted}}=\sum_c \frac{n_c}{N}\mathrm{F1}_c
$$

**AUC-ROC** is one-vs-rest, macro-averaged, computed from the (row-normalised) class probabilities.

### 5.2 Output heads and losses

**Cross-entropy (nominal):**

$$
\mathcal L_{\text{CE}} = -\log \sigma_{\text{sm}}(z)_y
$$

**CORAL** (Cao et al., 2019). Rank-consistent: one shared weight vector $w$ and $K-1$ biases $b_k$,

$$
P(Y>k\mid x)=\sigma\!\big(w^\top h(x)+b_k\big),\qquad k=0,\dots,K-2
$$

$$
\mathcal L_{\text{CORAL}}=\frac{1}{N(K-1)}\sum_{n}\sum_{k=0}^{K-2}
\mathrm{BCE}\!\Big(\sigma(z_{n,k}),\;\mathbb 1[y_n>k]\Big)
$$

Prediction: $\hat y=\sum_k \mathbb 1\big[\sigma(z_k)>0.5\big]$.

**CORN** (Shi, Cao & Raschka, 2021). Each node models a *conditional* probability, with independent weights per node:

$$
\sigma(z_k)=P(Y>k\mid Y>k-1)
$$

Node $k$ is trained only on samples that reached it ($y>k-1$; all samples for $k=0$):

$$
\mathcal L_{\text{CORN}}=\frac{1}{\sum_k |S_k|}\sum_{k=0}^{K-2}\sum_{n\in S_k}
\mathrm{BCE}\!\Big(\sigma(z_{n,k}),\;\mathbb 1[y_n>k]\Big),\qquad
S_k=\{n:\;y_n>k-1\}
$$

Unconditional cumulative probabilities are recovered by the chain rule:

$$
P(Y>k)=\prod_{j\le k}\sigma(z_j)
$$

> **Implementation note.** CORN logits are *conditional*, so class probabilities must be built from `cumprod(sigmoid(z))`. Using `sigmoid(z)` directly (the CORAL conversion) gives non-monotonic cumulatives and wrong class probabilities.

**Niu-style baseline** (after Niu et al., 2016). $K-1$ independent binary heads on a shared backbone. The monotonicity penalty below is this repo's own addition and, as far as I recall, not part of the original method, so treat this as an OR-CNN-style baseline rather than a faithful reproduction:

$$
\mathcal L_{\text{Niu}}=\mathcal L_{\text{BCE}}+\lambda_{\text{mono}}\sum_{k\ge1}
\mathrm{ReLU}\big(\sigma(z_k)-\sigma(z_{k-1})\big),\qquad \lambda_{\text{mono}}=0.5
$$

**Regression (MSE) head** (dual-head model only). A scalar score $r$ initialised at the middle of the scale ($(K-1)/2$):

$$
\mathcal L_{\text{MSE}}=(r-y)^2,\qquad \hat y=\mathrm{clamp}\big(\mathrm{round}(r),0,K-1\big)
$$

### 5.3 Ordinal outputs → class probabilities

To fuse ordinal and nominal views, every ordinal output is mapped onto the $K$-class simplex.

**CORAL / CORN.** With $P(Y>-1)=1$ and $P(Y>K-1)=0$:

$$
p_k = P(Y>k-1)-P(Y>k),\qquad
\tilde p_k=\frac{\max(p_k,\epsilon)}{\sum_j \max(p_j,\epsilon)},\quad \epsilon=10^{-8}
$$

where $P(Y>k)=\sigma(z_k)$ for CORAL and $\prod_{j\le k}\sigma(z_j)$ for CORN.

**Regression.** Distance-based softmax with sharpness $\tau=0.5$:

$$
p_k=\frac{\exp\!\big(-(r-k)^2/\tau\big)}{\sum_j\exp\!\big(-(r-j)^2/\tau\big)}
$$

### 5.4 Single-teacher KD

The two single-teacher notebooks use **different** loss forms.

**CE variant** (`kd_ce_*`): Hinton-style distillation with temperature $T=10$ and $\alpha=0.5$. The per-sample distillation term is clamped at 10 before averaging:

$$
\mathcal L=(1-\alpha)\,\mathrm{CE}(z^s,y)+\alpha\;\overline{\min\!\Big(T^2\,\mathrm{KL}\big(\sigma_{\text{sm}}(z^t/T)\,\|\,\sigma_{\text{sm}}(z^s/T)\big),\;10\Big)}
$$

The $T^2$ factor keeps gradient magnitudes comparable across temperatures.

**CORAL variant** (`kd_coral_*`): distillation in the cumulative-logit space, **without temperature**, with $\alpha=0.3$:

$$
\mathcal L=(1-\alpha)\,\mathcal L_{\text{CORAL}}(z^s,y)+\alpha\,\mathrm{BCE}\big(\sigma(z^s),\,\sigma(z^t)\big)
$$

The notebook declares `TEMPERATURE = 10` and `BETA = 1.0`, but its loss uses neither. The multi-teacher notebooks (§5.5, §5.6) use $T=10$, $\alpha=0.5$.

**Where the clamp applies.** The per-sample clamp at 10 appears only in the single-teacher CE-KD notebook and in the homogeneous nominal multi-teacher notebook (`mkd_hoomogenous_..._ce`). The hybrid CA-MKD, homogeneous CORAL, single-teacher CORAL and dual-head losses (§5.5, §5.6) have no clamp, so their KD terms are not capped. Whether the clamp is ever active at $T=10$ is not logged.

**The stage-3 grid tuned a different loss.** In `hyperparametersearch.ipynb` the CORAL-KD loss is $(1-\alpha)\,\beta\,\mathcal L_{\text{CORAL}}+\alpha\,T^2\,\mathrm{KL}$ with the KL taken in class space at temperature $T$, so $\beta$ weights the hard term. The final CORAL-KD notebook instead uses a BCE on cumulative logits with no temperature and no $\beta$. The search therefore ranked $(\alpha,\beta)$ for a loss the final notebook does not use, and $\alpha=0.3$ was carried over without being re-tuned for it.

**Hyper-parameter search** (`02_hypertuning`, validation QWK, staged):

| Stage | Searched | Best found |
|---|---|---|
| 0a | learning rate × weight decay | lr = 1e-3, wd = 1e-4 |
| 0b | batch size × dropout | bs = 64, dropout = 0.4 ranked first |
| 1 | temperature $T\in\{3,5,7,10\}$ | $T=10$, the upper edge of the grid, so larger values were not tested |
| 2 | CE-KD $\alpha\in\{0.3,0.5,0.7,0.9\}$ | $\alpha=0.5$ |
| 3 | CORAL-KD $(\alpha,\beta)$ grid | $\alpha=0.3,\ \beta=1.0$ ranked first; the final CORAL-KD notebook uses $\alpha=0.3$ and does not use $\beta$ in its loss |

### 5.5 Heterogeneous multi-teacher KD (CA-MKD hybrid)

A **CE teacher** (softmax output) and an **ordinal teacher** (cumulative output) speak different probabilistic languages. The hybrid recipe (adapted from CA-MKD, Zhang et al., 2022) unifies them and weights each teacher *per sample* by how well it agrees with the ground truth.

> **Fidelity to CA-MKD.** This is a CA-MKD-*style* method, not a reproduction. The paper weights each teacher's prediction by how close it is to the one-hot label, which this recipe shares, but it also adds an intermediate-layer feature loss with its own separately computed weights. Here the weights are $c/\sum c$ with $c=p_y$, computed on tempered distributions after aggregating the teachers' outputs into one soft target. I have not checked the paper's exact weight function or whether it distils each teacher separately, so check Zhang et al. before comparing against their reported numbers. The logit-only recipe here has no feature term. Read every "CA-MKD" label in this README as "CA-MKD-style".

**Step 1 — unify onto the simplex at the same temperature.**

$$
p^{\text{CE}}=\sigma_{\text{sm}}\!\big(z^{\text{CE}}/T\big),\qquad
p^{\text{ord}}=\sigma_{\text{sm}}\!\Big(\log\tilde p^{\text{ord}}_{\text{raw}}\,/\,T\Big)
$$

where $\tilde p^{\text{ord}}_{\text{raw}}$ is the ordinal teacher's class distribution from §5.3. Tempering in log-space gives the ordinal teacher the same notion of "softness" as the CE teacher at a given $T$.

**Step 2 — per-sample confidence.** With $y$ one-hot, each teacher's cross-entropy against the label is turned into a confidence:

$$
c^{(m)}=\exp\!\big(-\mathrm{CE}(p^{(m)},y)\big)=p^{(m)}_y,\qquad
w^{(m)}=\frac{c^{(m)}}{c^{\text{CE}}+c^{\text{ord}}}
$$

**Step 3 — aggregate soft target and student loss.**

$$
p^{\text{agg}}=w^{\text{CE}}p^{\text{CE}}+w^{\text{ord}}p^{\text{ord}}
$$

$$
\boxed{\;\mathcal L=(1-\alpha)\,\mathrm{CE}(z^s,y)+\alpha\,T^2\,
\mathrm{KL}\!\Big(p^{\text{agg}}\;\Big\|\;\sigma_{\text{sm}}(z^s/T)\Big)\;}
\qquad T=10,\;\alpha=0.5
$$

**Note on the confidences at $T=10$ (single-head CA-MKD).** Both confidences are computed from *tempered* class distributions, which are close to uniform at $T=10$. The values $c^{(m)}=p^{(m)}_y$ then vary little between samples and are similar for the two teachers, so the weights $w^{(m)}$ are probably close to 0.5 and the per-sample weighting may do little. This is not measured, because the weights are not logged. The dual-head case differs (§5.6).

**Clamp.** The boxed loss has no per-sample clamp, and the hybrid notebooks do not apply one. The homogeneous nominal notebook (`mkd_hoomogenous_..._ce`) does clamp its per-sample KD term at 10, so its KD term is not directly comparable with the hybrid ones (see "Where the clamp applies" in §5.4).

The student keeps a **single nominal head**, so inference cost is identical to the baseline.

Variants in `04_kd_multi/01_mkd/`:

| Notebook | Teachers |
|---|---|
| `mkd_hoomogenous_..._ce` | ResNet-50-CE + EffNet-B3-CE (both nominal) |
| `mkd_homogenous_..._coral` | ResNet-50-CORAL + EffNet-B3-CORAL (both ordinal) |
| `mkd_hybrid_resnet50_efficientnetb3_coral` | ResNet-50-CE + EffNet-B3-CORAL |
| `mkd_hybrid_resnet50_efficientnetb3_corn` | ResNet-50-CE + EffNet-B3-CORN |
| `mkd_hybrid_resnet50_resnet50_coral` | ResNet-50-CE + ResNet-50-CORAL |

### 5.6 Dual-head student with fusion gate (DP-KD)

Instead of collapsing both teachers into one head, the student gets **two heads on a shared backbone**: a nominal head distilled from the CE teacher and an ordinal head distilled from the ordinal teacher. The two heads' class distributions are fused at inference.

```
                         ┌─► nominal head ──► p_nom ─┐
image ─► MobileNetV3 ─► h│                           ├─► w·p_nom + (1−w)·p_ord ─► argmax
                         └─► ordinal head ──► p_ord ─┘
```

#### Distillation losses (per sample $n$)

**Nominal branch** (teacher: ResNet-50-CE):

$$
\ell^{\text{nom}}_{\text{KD}}=T^2\,\mathrm{KL}\!\Big(\sigma_{\text{sm}}(z^{t,\text{CE}}/T)\,\Big\|\,\sigma_{\text{sm}}(z^{s,\text{nom}}/T)\Big),\qquad
c^{\text{nom}}=\mathrm{sg}\Big[\exp\!\big(-\mathrm{CE}(p^{\text{CE}},y)\big)\Big]
$$

**Ordinal branch, CORAL/CORN** (teacher: EfficientNet-B3-ordinal). Distillation happens in the teacher's *native binary space*. The KD term sums over **all** $K-1$ nodes with no masking (for CORN this includes nodes a sample would not reach during conditional training). The "active node" mask $a_{n,k}$ (all ones for CORAL; $\mathbb 1[y_n>k-1]$ for CORN) is used only inside the confidence below:

$$
\ell^{\text{ord}}_{\text{KD}}=T^2\sum_{k}\mathrm{BCE}\!\Big(\sigma(z^{s}_k/T),\;\sigma(z^{t}_k/T)\Big)
$$

$$
c^{\text{ord}}=\mathrm{sg}\Bigg[\exp\!\Bigg(-\frac{\sum_k a_k\,\mathrm{BCE}\big(z^t_k/T,\;\mathbb 1[y>k]\big)}{\sum_k a_k}\Bigg)\Bigg]
$$

**Ordinal branch, MSE** (teacher regresses a score $r^t$):

$$
\ell^{\text{ord}}_{\text{KD}}=s\,(r^s-r^t)^2,\quad s=10,\qquad
c^{\text{ord}}=\mathrm{sg}\big[\exp(-(r^t-y)^2)\big]
$$

**Per-sample confidence weighting** (same idea as §5.5; $\mathrm{sg}$ = stop-gradient):

$$
w^{\text{nom}}_n=\frac{c^{\text{nom}}_n}{c^{\text{nom}}_n+c^{\text{ord}}_n},\quad
w^{\text{ord}}_n=1-w^{\text{nom}}_n,\qquad
\mathcal L_{\text{KD}}=\frac1N\sum_n\Big(w^{\text{nom}}_n\,\ell^{\text{nom}}_{\text{KD}}+w^{\text{ord}}_n\,\ell^{\text{ord}}_{\text{KD}}\Big)
$$

**How the confidences are computed differs by branch.** For the nominal branch and the CORAL/CORN branch they come from $T$-softened teacher outputs ($T=10$). For the MSE branch, $c^{\text{ord}}=\exp(-(r^t-y)^2)$ uses the raw score with no temperature.

A back-of-envelope estimate follows, using assumed logit magnitudes and not measured values:

- **CORAL/CORN.** With $z/T$ small, each BCE term is near $\log 2$, so $c^{\text{ord}}$ is about 0.5–0.6 and nearly constant. $c^{\text{nom}}=p^{\text{CE}}_y$ at $T=10$ is plausibly 0.2–0.35. That would put $w^{\text{nom}}$ around 0.3–0.4, tilted towards the ordinal branch and almost constant across samples.
- **MSE.** $c^{\text{ord}}$ varies strongly with the teacher's error (about 0.78 at half a grade, 0.37 at one grade, 0.02 at two), so the weighting is genuinely sample-dependent, and it combines an untempered ordinal confidence with a tempered nominal one.

Log the weights to settle this.

#### Hard-label and fusion supervision

$$
\mathcal L_{\text{hard}}=\tfrac12\,\mathrm{CE}(z^{s,\text{nom}},y)+\tfrac12\,\mathcal L_{\text{ord}}(z^{s,\text{ord}},y)
$$

$$
p_{\text{final}}=\mathrm{normalise}\big(w\,p_{\text{nom}}+(1-w)\,p_{\text{ord}}\big),\qquad
\mathcal L_{\text{fuse}}=-\log p_{\text{final},\,y}
$$

#### Total objective

$$
\boxed{\;\mathcal L=(1-\alpha)\,\mathcal L_{\text{hard}}+\alpha\,\mathcal L_{\text{KD}}+\lambda_f\,\mathcal L_{\text{fuse}}+\beta\,\mathcal L_{\text{feat}}\;}
\qquad T=10,\ \alpha=0.5,\ \lambda_f=0.1
$$

$\lambda_f=0.1$ is the default of the `fusion_weight` argument of `dp_kd_loss`; it is not in the config block.

$\mathcal L_{\text{feat}}$ is optional feature distillation, active only when `BETA_FEAT > 0` (`03_dualheaded_logit_feature/`). Two linear projectors map the student feature into each teacher's feature space and a cosine distance is weighted by the same confidences:

$$
\mathcal L_{\text{feat}}=\frac1N\sum_n\Big[w^{\text{nom}}_n\big(1-\cos(P_{\text{CE}}h_n,\,f^{\text{CE}}_n)\big)+w^{\text{ord}}_n\big(1-\cos(P_{\text{ord}}h_n,\,f^{\text{ord}}_n)\big)\Big]
$$

The projectors are used in training only and are dropped at inference.

#### Fusion gate modes (`GATE_MODE`)

$w$ is the weight on the **nominal** branch. The notebooks implement four ways to set it:

| Mode | Definition | Trainable? |
|---|---|---|
| `learned` | $w=\sigma(g)$, $g$ a single scalar initialised at 0 (so $w_0=0.5$) | yes, gradient only via $\lambda_f\mathcal L_{\text{fuse}}$ |
| `fixed` | $w=0.5$ (hard-coded; `FIXED_W_NOM`) | no |
| `loss_relative`, `REL_FORM='ratio'` | $w_b=\dfrac{L_{\text{ord}}}{L_{\text{nom}}+L_{\text{ord}}}$ | no |
| `loss_relative`, `REL_FORM='sigmoid'` | $w_b=\sigma\big(K\,(L_{\text{ord}}-L_{\text{nom}})\big)$, `REL_K`=1 | no |

For `loss_relative`, $L_{\text{nom}}$ and $L_{\text{ord}}$ are the **true-class negative log-likelihoods** of each branch on the batch, $L=-\frac1B\sum_n\log p_{n,y_n}$. Both live on the same $K$-class simplex, so they share a scale. The branch that fits the labels *worse* gets the *smaller* weight. No gradient flows through $w_b$.

Labels do not exist at validation or test time, so the evaluation weight is an **exponential moving average** of the training batch weights:

$$
\bar w\leftarrow 0.99\,\bar w+0.01\,w_b,\qquad \bar w_0=0.5
$$

In `learned` mode the gate is **one global scalar**: it is the same for every image and does not adapt per sample.

#### Probability normalisation (`PROB_NORM`)

Applied identically to both branches before fusion and before the fuse loss:

| Option | Operation |
|---|---|
| `none` | unchanged (original behaviour) |
| `simplex` | clamp to $\epsilon$ and renormalise; almost a no-op because softmax and CORAL/CORN class probabilities already sum to 1 |
| `tempered` | $p\leftarrow\sigma_{\text{sm}}(\log p\,/\,T_p)$ with the **same** $T_p$ (`TEMPER_T`, default 2) for both branches; this actually changes distribution sharpness |

---

## 6. Training protocol

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr = 1e-3, weight decay = 1e-4 (teachers and students) |
| Schedule | Cosine annealing, `T_max = 100` |
| Epochs | 100 |
| Batch size | 32 |
| Sampler | `WeightedRandomSampler`, inverse class frequency |
| Precision | Mixed precision (AMP) when a GPU is available |
| Seeds | 33, 81, 5 (Python, NumPy, PyTorch; deterministic cuDNN, benchmark off) |
| Model selection | Best **validation QWK** epoch, restored before testing |
| Resume | A `*_resume.pth` checkpoint (model, optimiser, scheduler, scaler, history) is written every epoch, so Kaggle sessions can be continued |
| KD | CE-based losses: $T=10$, $\alpha=0.5$. Single-teacher CORAL KD: $\alpha=0.3$, no temperature (§5.4) |

A notebook trains its teachers only if their checkpoint file is missing, and otherwise loads it. Kaggle sessions start fresh, so the ResNet-50-CE teacher was retrained in several notebooks with different outcomes (§7.5). The dual-head notebooks load whatever teacher checkpoints were in the working directory, and which training run produced them is not recorded.

Reported mean ± std uses the population standard deviation (`np.std`, `ddof=0`) over the three seeds.

---

## 7. Full results

All numbers are on the **test split** unless stated otherwise. Mean ± std over seeds 33, 81, 5. The ± values are population standard deviations (`ddof=0`). With $n=3$, multiply them by $\sqrt{3/2}\approx1.22$ to get the sample standard deviation.

### 7.1 Baselines (standalone MobileNetV3-Small)

| Method | QWK | Acc (%) | MAE | RMSE | Macro F1 |
|---|---:|---:|---:|---:|---:|
| CE | 0.8590 ± 0.0150 | 77.66 ± 2.57 | 0.2916 ± 0.0292 | 0.6698 ± 0.0312 | 0.6157 ± 0.0400 |
| CORAL | 0.8708 ± 0.0049 | 76.75 ± 0.51 | 0.2834 ± 0.0097 | 0.6433 ± 0.0175 | 0.6068 ± 0.0106 |
| CORN | 0.8633 ± 0.0039 | 78.11 ± 3.24 | 0.2834 ± 0.0231 | 0.6657 ± 0.0070 | 0.6318 ± 0.0455 |
| Niu-style | 0.8605 ± 0.0032 | 77.93 ± 1.02 | 0.2879 ± 0.0056 | 0.6759 ± 0.0091 | 0.6246 ± 0.0126 |

### 7.2 Single-teacher KD (teacher: ResNet-50)

| Method | QWK | Acc (%) | MAE | Macro F1 |
|---|---:|---:|---:|---:|
| KD-CE | 0.8847 ± 0.0041 | 83.29 ± 0.34 | 0.2252 ± 0.0056 | 0.6896 ± 0.0096 |
| KD-CORAL | 0.8867 ± 0.0064 | 81.38 ± 0.84 | 0.2352 ± 0.0112 | 0.6397 ± 0.0062 |

### 7.3 Multi-teacher KD, single-head student

| Method | Teachers | QWK | Acc (%) | MAE | RMSE | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| Homogeneous nominal | R50-CE + EB3-CE | 0.8721 ± 0.0121 | 81.93 ± 1.00 | 0.2498 ± 0.0180 | 0.6434 ± 0.0341 | 0.6744 ± 0.0185 |
| Homogeneous ordinal | R50-CORAL + EB3-CORAL | 0.8834 ± 0.0138 | 79.02 ± 1.18 | 0.2579 ± 0.0114 | 0.6143 ± 0.0222 | 0.6321 ± 0.0286 |
| **Hybrid (CA-MKD-style)** | **R50-CE + EB3-CORAL** | **0.8907 ± 0.0040** | **84.11 ± 0.90** | **0.2125 ± 0.0102** | **0.5873 ± 0.0136** | **0.7022 ± 0.0113** |
| Hybrid (CA-MKD-style) | R50-CE + R50-CORAL | 0.8800 ± 0.0061 | 81.20 ± 2.79 | 0.2452 ± 0.0257 | 0.6234 ± 0.0116 | 0.6683 ± 0.0410 |
| Hybrid (CA-MKD-style) | R50-CE + EB3-**CORN** | 0.8645 ± 0.0102 | 80.74 ± 1.03 | 0.2607 | n/a | n/a |

The CORN row is taken from the notebook's printed output (per-seed QWK 0.8681 / 0.8748 / 0.8506); the RMSE and F1 columns were not recorded in a summary file.

### 7.4 Dual-head student, probability-space fusion

Logit-only distillation ($\beta=0$), teachers R50-CE + EB3-(CORAL | CORN | MSE). Metrics are computed from the **fused** distribution.

| Ordinal head | Gate | QWK | Acc (%) | MAE | Macro F1 | AUC |
|---|---|---:|---:|---:|---:|---:|
| CORAL | learned | 0.8824 ± 0.0145 | 80.47 ± 2.68 | 0.2470 ± 0.0312 | 0.6487 ± 0.0304 | 0.9319 ± 0.0021 |
| CORAL | fixed 0.5 | 0.8788 ± 0.0038 | 81.47 ± 1.02 | 0.2407 ± 0.0046 | 0.6618 ± 0.0150 | 0.9350 ± 0.0011 |
| CORAL | relative (ratio) | 0.8865 ± 0.0038 | 82.20 ± 0.51 | 0.2334 ± 0.0013 | 0.6697 ± 0.0170 | 0.9333 ± 0.0056 |
| CORN | learned | 0.8838 ± 0.0097 | 82.92 ± 1.30 | 0.2298 ± 0.0156 | 0.6779 ± 0.0203 | 0.9342 ± 0.0035 |
| CORN | fixed 0.5 | 0.8811 ± 0.0048 | 81.93 ± 1.30 | 0.2389 ± 0.0161 | 0.6628 ± 0.0256 | 0.9399 ± 0.0042 |
| CORN | relative (ratio) | 0.8811 ± 0.0054 | 83.11 ± 0.80 | 0.2280 ± 0.0078 | 0.6835 ± 0.0058 | 0.9303 ± 0.0073 |
| MSE | learned | 0.8895 ± 0.0106 | 83.11 ± 1.46 | 0.2216 ± 0.0167 | 0.6857 ± 0.0082 | 0.9300 ± 0.0058 |
| MSE | fixed 0.5 | 0.8871 ± 0.0010 | 82.38 ± 1.30 | 0.2289 ± 0.0097 | 0.6816 ± 0.0122 | 0.9321 ± 0.0011 |

Not yet run or without saved outputs: `relative` with `REL_FORM='sigmoid'`, `relative` for MSE, `PROB_NORM='tempered'`, and the feature-distillation notebooks (`03_dualheaded_logit_feature/`).

### 7.5 Teachers (one trained model per row, test split)

Teachers are **not identical across experiment families**: each notebook trained its own copy unless a checkpoint was already present. The same architecture and loss therefore appears with different weights. Validation QWK comes from the training log of the notebook named in the second column. Test metrics come from `05_results/` (the notebook-to-row mapping follows `eval_multikd.py`; the single-teacher rows follow the scenario names).

| Teacher | Notebook family | Best val QWK | Test QWK | Acc (%) | MAE | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| ResNet-50-CE | single-teacher KD | 0.9093 | 0.8783 | 81.47 | 0.2425 | 0.6643 |
| ResNet-50-CE | homogeneous nominal | 0.9037 | 0.8796 | 84.47 | 0.2207 | 0.7113 |
| ResNet-50-CE | hybrid R50 + EB3 (CORAL) | 0.9057 | 0.8708 | 81.47 | 0.2452 | 0.6373 |
| ResNet-50-CE | hybrid R50 + R50 | 0.9073 | 0.8822 | 83.11 | 0.2262 | 0.6710 |
| ResNet-50-CE | hybrid R50 + EB3 (CORN) | 0.9027 | n/a | n/a | n/a | n/a |
| EfficientNet-B3-CE | homogeneous nominal | 0.9270 | 0.8807 | 82.56 | 0.2371 | 0.6704 |
| ResNet-50-CORAL | single-teacher KD | 0.9094 | 0.8818 | 78.47 | 0.2589 | 0.5968 |
| ResNet-50-CORAL | homogeneous ordinal; hybrid R50 + R50 (same test row) | 0.9024 | 0.8678 | 77.38 | 0.2779 | 0.5971 |
| EfficientNet-B3-CORAL | homogeneous ordinal; hybrid R50 + EB3 (same test row) | 0.9155 | 0.8984 | 84.20 | 0.2044 | 0.6797 |
| EfficientNet-B3-CORN | hybrid R50 + EB3 (CORN) | 0.9112 | n/a | n/a | n/a | n/a |

Points to note:

- **Five different ResNet-50-CE models** (four with test scores) have validation QWK between 0.9027 and 0.9093. The four with test scores have test QWK between 0.8708 and 0.8822. The fifth, from the CORN hybrid notebook, has no test score.
- **Two different ResNet-50-CORAL models.** The one used for single-teacher KD (test QWK 0.8818) is much stronger than the one used in the multi-teacher runs (0.8678).
- **No test score exists for the CORN and MSE teachers.** The dual-head CORN and MSE runs load EfficientNet-B3-CORN / -MSE and ResNet-50-CE checkpoints whose training run is not recorded.
- **Validation and test rankings disagree** in places (EfficientNet-B3-CE has the highest validation QWK, 0.9270, but a test QWK of 0.8807).

### 7.6 Per-class behaviour of the best single-head model

CA-MKD hybrid (R50-CE + EB3-CORAL), test F1 per class, mean ± std over seeds:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| No DR | 0.961 ± 0.009 | 0.991 ± 0.003 | 0.976 ± 0.006 |
| Mild | 0.663 ± 0.014 | 0.730 ± 0.066 | 0.694 ± 0.038 |
| Moderate | 0.808 ± 0.014 | 0.783 ± 0.013 | 0.795 ± 0.010 |
| Severe | 0.400 ± 0.017 | 0.561 ± 0.025 | 0.467 ± 0.018 |
| Proliferative DR | 0.837 ± 0.042 | 0.444 ± 0.031 | 0.579 ± 0.026 |

Severe and Proliferative DR remain the hard classes. Proliferative DR has high precision but low recall.

### 7.7 Statistical analysis

`06_analysis/statisticalanalysis.ipynb` compares CA-MKD hybrid (R50 + EB3) against the standalone CE baseline and the single-teacher KD-CE model with a **paired t-test**, a **Wilcoxon signed-rank test** and **Cohen's d**.

With $n=3$ pairs the smallest attainable two-sided Wilcoxon $p$-value is $0.25$, so the Wilcoxon test can never reach conventional significance here. Cohen's d is the more informative number, and it should still be read as indicative only. The paired t-test has df = 2 (two-sided critical $t=4.30$ at the 5 % level), so it has little power and one seed can decide it. Treat all three tests as descriptive.

---

## 8. Observations

1. **Distillation helps a lot over training the small model alone.** Standalone CE reaches QWK 0.8590. Distilling from ResNet-50 lifts it to 0.8847 and accuracy from 77.7 % to 83.3 %.
2. **One hybrid pairing gives the best single-head result.** CA-MKD with ResNet-50-CE and EfficientNet-B3-CORAL has the highest mean QWK (0.8907), accuracy (84.11 %), and the lowest MAE and RMSE. Its QWK std (0.0040) is small, but standard deviations from 3 seeds are too noisy to rank. 0.0040 against 0.0061 for the next single-head variant, or against 0.0038 for the CORAL dual-head variants, is not a meaningful difference.
3. **Heterogeneous teachers are not shown to be better in general.** Only one of the three hybrid pairings beats the homogeneous pairs. R50-CE + R50-CORAL (0.8800) and R50-CE + EB3-CORN (0.8645) both score below homogeneous ordinal distillation (0.8834). A second teacher also did not beat the best single-teacher student in either homogeneous case (nominal 0.8721 vs. KD-CE 0.8847; ordinal 0.8834 vs. KD-CORAL 0.8867). The hybrid's margin over the best single-teacher model is 0.0040 QWK, smaller than the single-teacher stds (0.0041 and 0.0064).
4. **Teacher strength and heterogeneity are confounded.** The winning pair contains EfficientNet-B3-CORAL, the strongest teacher by test QWK (0.8984). The CORN hybrid is lower (0.8645), but there is no test score for its teacher, and its validation QWK (0.9112) is close to the CORAL teacher's (0.9155). The CORN → class-probability conversion was checked in the code. In `mkd_hybrid_resnet50_efficientnetb3_corn` the teacher is converted with `corn_logits_to_probs`, and the helper named `coral_logits_to_probs` in that notebook also uses `cumprod(sigmoid(z))`, so a wrong (CORAL-style) conversion is ruled out as the cause of 0.8645. The dual-head CORN notebooks (`{learned,fixed,relative}_corn` and `03_dualheaded_logit_feature/corn`) use `cumprod` in `ord_cum_probs` for CORN as well, so the same check passes there. The ResNet-50-CORAL comparison is also not like for like: the 0.8867 single-teacher student was distilled from a stronger ResNet-50-CORAL model (test QWK 0.8818) than the one in the R50-CE + R50-CORAL hybrid (0.8678), which gave 0.8800. Teachers are retrained per notebook (§7.5), and EfficientNet-B3 is run at 224 px, below its native 300 px.
5. **The dual-head variants land in the same range as single-head CA-MKD without clearly beating it.** Their QWK values (0.879 – 0.890) lie within the seed noise of one another. The MSE head with a learned gate comes closest (0.8895).
6. **A learned gate drifts towards the nominal branch, but not for the reported checkpoint.** In the CORN run the nominal weight starts near 0.46 and rises to about 0.95 by epoch 100. The checkpoint actually evaluated is chosen by validation QWK, where the gate is lower: 78.4 % (seed 33, epoch 42), 73.5 % (seed 81) and 51.7 % (seed 5, epoch 17).
7. **Fixed 0.5 and loss-relative gates are about as accurate as the learned gate, and their QWK may vary less across seeds.** Accuracy shows no consistent difference between fixed and learned: fixed is 1.0 point higher for CORAL (81.47 vs. 80.47) and lower for CORN (81.93 vs. 82.92) and MSE (82.38 vs. 83.11), all within noise. Seed-to-seed std of QWK is 0.0010 – 0.0054 for fixed/relative against 0.0097 – 0.0145 for learned, although with 3 seeds that spread comparison is itself weak.
8. **Errors look mostly adjacent-grade, but no confusion matrix was saved.** The repo stores per-class precision and recall (§7.6) but no confusion matrices, so which grade pairs are confused is not documented here. One bound that does follow from the tables: for the CA-MKD hybrid, MAE 0.2125 over a misclassification rate of 15.9 % means misclassified images are on average about 1.34 grades off. Since every error is at least 1 grade, that forces roughly two-thirds or more of the errors to be exactly one grade off (this uses means over seeds, so it is approximate). The per-class pattern (§7.6) is that Proliferative DR has high precision but low recall and Severe has low precision, so some Proliferative images are being predicted as lower grades. Where they go is untested. Adding a confusion matrix per model would settle it.

---

## 9. Reproducing the experiments

### 9.1 Environment

- Python 3.10+
- PyTorch + torchvision, scikit-learn, NumPy, pandas, matplotlib, Pillow
- `opencv-python` and `albumentations` for the data notebooks
- `scipy` for the statistical analysis
- A CUDA GPU is strongly recommended (each experiment is 3 seeds × 100 epochs on 20,503 images)

### 9.2 Order of execution

1. **Data.** `00_data/02_stratifiedsplit.ipynb` → `00_data/03_dataaugmentation.ipynb`. Upload the result as a Kaggle dataset.
2. **Baselines.** `01_baseline/standalone_*.ipynb`.
3. **(Optional) Hyper-parameter search.** `02_hypertuning/hyperparametersearch.ipynb`.
4. **Single-teacher KD.** `03_kd_single/kd_*.ipynb`. Each notebook trains its ResNet-50 teacher only if the checkpoint is missing: `kd_ce_*` uses `teacher_resnet50_shared.pth` and `kd_coral_*` uses `teacher_resnet50_coral.pth`, which are two different models. The dual-head notebooks load differently named `hybrid_teacher_*` checkpoints (§6, §7.5).
5. **Multi-teacher KD.** `04_kd_multi/01_mkd/*.ipynb` and `04_kd_multi/02_dualheaded_hybrid/*.ipynb`.
6. **Evaluation and tables.** `05_results/eval_*.py`, then `05_results/results.ipynb`.
7. **Statistics.** `06_analysis/statisticalanalysis.ipynb`.

Set the dataset path at the top of each notebook:

```python
DATASET_DIR = '/kaggle/input/datasets/<user>/aptos-2019-224px-aug-v2'
CKPT_DIR    = '/kaggle/working/'
```

The dataset, the extracted images and `dataset.zip` are git-ignored and are not in the repository.

### 9.3 Switches in the dual-head notebooks

Each dual-head notebook is configured by a block at the top. The same code covers every row of §7.4.

```python
ORDINAL_MODE = 'corn'          # 'coral' | 'corn' | 'mse'
BETA_FEAT    = 0.0             # 0 = logit-only KD, >0 = + feature KD

GATE_MODE    = 'learned'       # 'learned' | 'fixed' | 'loss_relative'
FIXED_W_NOM  = 0.5             # used when GATE_MODE == 'fixed'
REL_FORM     = 'ratio'         # 'ratio' | 'sigmoid'   (loss_relative)
REL_K        = 1.0             # sigmoid slope
REL_EMA      = 0.99            # EMA momentum for the evaluation weight

PROB_NORM    = 'none'          # 'none' | 'simplex' | 'tempered'
TEMPER_T     = 2.0             # shared temperature for 'tempered'

MSE_TAU      = 0.5             # MSE head: score → class distribution
MSE_KD_SCALE = 10.0            # MSE head: regression-KD scale

KD_TEMPERATURE = 10.0
KD_ALPHA       = 0.5
```

Checkpoints and summaries are named after the variant, so runs do not overwrite each other, for example `dpkd_logit_corn_seed33_best.pth` for the original configuration and a suffix such as `_relratio_none` for a non-default gate or normalisation. Each run also writes `summary_dpkd_{logit|feat}_{mode}.json` with per-seed validation and test metrics and the final gate weight.

The fuse-loss weight $\lambda_f=0.1$ is not in this block. It is the default `fusion_weight=0.1` argument of `dp_kd_loss`.

Suggested next experiments, none of which are in the tables above:

| Run | Settings |
|---|---|
| Relative gate, sigmoid form | `GATE_MODE='loss_relative'`, `REL_FORM='sigmoid'` |
| Relative gate for MSE | `ORDINAL_MODE='mse'`, `GATE_MODE='loss_relative'` |
| Shared temperature | `PROB_NORM='tempered'`, `TEMPER_T=2.0`, with a fixed or relative gate |

---

## 10. Known caveats

- **Small evaluation set, few seeds.** The test split has 367 images (Severe: 19, Proliferative DR: 30), so one Severe image moves its recall by about 5 points and one Proliferative image by about 3. Seed standard deviations ignore this test-set sampling variance. Bootstrap confidence intervals on the test set would be more honest than seed std, and the ± values here are population std (multiply by about 1.22 for sample std).
- **Selection on the test set.** The highlighted CA-MKD hybrid (best of the single-head variants) and the MSE learned-gate dual-head (best of the dual-head variants) were picked by test QWK. That inflates their apparent advantage. Choose variants on validation, then report test once.
- **Single split, single dataset.** There is one stratified split and no external validation set, so generalisation to other cameras or populations is untested.
- **MixUp labels are truncated when loaded** (details in §3.3). 44.9 % of MixUp images are trained with a label that differs from their source image, and only 20 of 236 Proliferative-source MixUp images keep the Proliferative label. This may contribute to the low Proliferative recall (0.444) and low Severe precision (0.400), but that link is a hypothesis and has not been tested. Re-running with rounded or soft labels would settle it.
- **Confidence weights are not logged.** In single-head CA-MKD both confidences come from $T=10$ tempered probabilities and are probably near 0.5. In the dual-head CORAL/CORN branch they are probably near-constant but tilted towards the ordinal branch, and in the MSE branch the ordinal confidence uses the untempered score error (§5.6). These are estimates, not measurements.
- **Mixed KD formulations.** Single-teacher CORAL KD uses α = 0.3 and no temperature, while the CE variant and all multi-teacher runs use T = 10, α = 0.5. The per-sample clamp at 10 exists only in two nominal-KD notebooks (§5.4), and the stage-3 (α, β) search tuned a different loss from the final CORAL-KD one. $T=10$ is also the upper edge of the searched grid. Differences between rows therefore mix teacher count with loss form.
- **Teachers differ between experiment families.** ResNet-50-CE was retrained in five notebooks (val QWK 0.9027 – 0.9093; test QWK 0.8708 – 0.8822 across the four that have test scores) and two different ResNet-50-CORAL models exist (test QWK 0.8818 and 0.8678). Student comparisons across families therefore also compare different teacher weights. The teachers behind the dual-head runs are not identifiable from the repo (§7.5).
- **Teacher resolution.** EfficientNet-B3 is trained at 224 px, although the split notebook notes 300 px as its native size. That may handicap that teacher.
- **Niu-style baseline.** The monotonicity penalty (λ = 0.5) is this repo's addition, so the baseline is not a strict reproduction of Niu et al.
- **Ben Graham blur.** σ = 224/30 is wider than Graham's σ = radius/30 relative to the image (see §3.2).
- **Metric coverage and labels.** AUC-ROC exists only for the dual-head runs. The per-seed lines in the dual-head notebooks print both macro F1 and weighted F1 under the same name `F1-score` (the first value is macro, the second weighted).
- **Two result sources.** Baseline, single-KD and single-head multi-teacher numbers come from the `05_results/eval_*.py` scripts. Dual-head numbers come from each notebook's own test cell. Both use the same test split but are not guaranteed to be bit-identical.
- **Evaluation scripts use older names.** `eval_multikd.py` still points at names such as `mkd_ce_resnet50_efficientnetb3.ipynb` and `04_kd_multi/models/...`. Update them after the reorganisation into `01_mkd/`, `02_dualheaded_hybrid/`, `03_dualheaded_logit_feature/`.
- **Search result vs. final setting.** In stage 0b of the hyper-parameter search, batch size 64 with dropout 0.4 ranked first on validation QWK, while the experiment notebooks use batch size 32.
- **Latency** was measured on CPU on one machine (hardware not recorded), and peak-memory values are recorded as `nan`.
- **Study status.** The feature-distillation notebooks have no saved outputs, and a few gate/normalisation variants listed in §9.3 have not been run.

---

## 11. References

- **APTOS 2019 Blindness Detection**, Kaggle competition dataset (Asia Pacific Tele-Ophthalmology Society).
- Graham, B. *Kaggle Diabetic Retinopathy Detection competition report* (2015), source of the local-contrast preprocessing.
- Hinton, G., Vinyals, O., Dean, J. *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531, 2015.
- Zhang, H., Chen, D., Wang, C. *Confidence-Aware Multi-Teacher Knowledge Distillation.* ICASSP 2022. arXiv:2201.00007.
- Cao, W., Mirjalili, V., Raschka, S. *Rank consistent ordinal regression for neural networks with application to age estimation* (CORAL). arXiv:1901.07884, 2019.
- Shi, X., Cao, W., Raschka, S. *Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities* (CORN). arXiv:2111.08851, 2021.
- Niu, Z., Zhou, M., Wang, L., Gao, X., Hua, G. *Ordinal Regression with Multiple Output CNN for Age Estimation.* CVPR 2016.
- Zhang, H., Cissé, M., Dauphin, Y., Lopez-Paz, D. *mixup: Beyond Empirical Risk Minimization.* ICLR 2018.
- Howard, A. et al. *Searching for MobileNetV3.* ICCV 2019.
- Tan, M., Le, Q. *EfficientNet: Rethinking Model Scaling for CNNs.* ICML 2019.
- He, K., Zhang, X., Ren, S., Sun, J. *Deep Residual Learning for Image Recognition.* CVPR 2016.

---

## License and citation

No license file is currently included in the repository. Add one before reuse by others, and add a citation entry here once the thesis or paper is published.
