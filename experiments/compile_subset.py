# Cara jalankan: python experiments/compile_subset.py
# Kompilasi hasil subset experiment -> tabel perbandingan PaDiM vs KNN

import os, json
base = "output/experiments/subset"
dirs = sorted(os.listdir(base))

print(f"{'Folder':<15} {'N':>4} {'Aug':>5} {'PaDiM-AUROC':>12} {'KNN-AUROC':>10}", end="")
print(f" {'Gap':>8} {'PaDiM-Pixel':>12} {'KNN-Pixel':>10}", end="")
print(f" {'PaDiM-PRO':>10} {'KNN-PRO':>8}")
print("-" * 95)

for d in dirs:
    mpath = os.path.join(base, d, "metrics.json")
    if not os.path.exists(mpath):
        continue
    m = json.load(open(mpath))
    p, k = m["padim"], m["knn"]
    n, aug = p["n_train"], d.endswith("_aug")
    gap = p["auroc"] - k["auroc"]
    print(f"{d:<15} {n:>4} {str(aug):>5} {p['auroc']:>12.4f} {k['auroc']:>10.4f}", end="")
    print(f" {gap:>8.4f} {p['pixel_auroc']:>12.4f} {k['pixel_auroc']:>10.4f}", end="")
    print(f" {p['pro_score']:>10.4f} {k['pro_score']:>8.4f}")
