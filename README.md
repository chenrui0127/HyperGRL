# ANGLE

This repository contains the official implementation of **HyperGRL**.

---


## ⚙️ Dependencies

| Package | Version |
|----------|----------|
| Python | 3.10 |
| torch | 2.4.1 |
| torch-geometric | 2.6.1 |
| torch_sparse | 0.6.18 |
| torch_scatter | 2.1.2 |
| torch_cluster | 1.6.3 |
| torch_spline_conv | 1.2.2 |
| numpy | 2.1.2 |
| scipy | 1.15.3 |
| scikit-learn | 1.6.1 |
| pandas | 2.3.2 |
| matplotlib | 3.10.5 |
| tqdm | 4.67.1 |
| ogb | 1.3.6 |
| umap-learn | 0.5.9.post2 |


---

## 🚀 Run

Example command to train on **Cora**:

```bash
python flow-on-hypersphere.py   --dataset Cora   --k 1   --alpha 1.5   --dataset_dir [path_to_datasets]
```

You can modify the arguments to experiment with other datasets such as `CiteSeer`, `PubMed`, or `Photo`.

---

## 📁 Project Structure

```
ANGLE/
├── run.py                   # Main training script
├── transforms.py            # Graph data augmentation
├── README.md                # Project description
└── results/                 # Output logs and results
```
