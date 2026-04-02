# Zakine2023_PRX — Results & Tested Parameters Summary

整理自 `Zakine2023_PRX/` 內所有輸出與目錄結構（約 **75 個子目錄**、**740+ 張圖**）。參數多由檔名或目錄名推得（例如 `upward_L10_N800_D0.001_dtau2.0e-02_*.png` → L=10, Ncopy=800, D=0.001, dtau=0.02）。

---

## 根目錄檔案

| 檔案 | 說明 |
|------|------|
| `Zakine2023_PRX.pdf` | 論文 PDF |
| `legendre_transform_viz.py` | Legendre 變換視覺化（\(L(v)=\sup_\theta[v\theta-H(\theta)]\)，slider 調 \(v\)） |
| `maier_stein_phase_portrait.py` | Maier–Stein 2D 非平衡相圖（向量場 + 異宿軌道 / Forward–Backward MAP 概念路徑） |

---

## Model A（非守恆，modified Ginzburg–Landau + κ）

- **路徑**：`modelA/`
- **端點**：均勻（homogeneous）\(\rho_1,\rho_2\)，來自 cubic roots（有 \(\kappa\)）。

### 1D：L10 forward / backward

| 目錄 | 推得參數 | 備註 |
|------|----------|------|
| `modelA/L10_forward` | L=10, N=800, D=0.001, kappa=0.26, dtau=2e-2 | forward 路徑 |
| `modelA/L10_backward` | 同上 | backward 路徑 |

### 1D：L2 與 L2 GL code

| 目錄 | 說明 |
|------|------|
| `modelA/L2_GL_code` | L=2 相關 |
| `modelA/L2_modelA` | L=2 Model A |

### 2D / 1D（fix_gridsize_h05）

網格間距固定 \(h=0.5\) 的測試，多為 2D（Lx×Ly）或 1D（L×1）：

| 目錄 | 推得/註解 |
|------|-----------|
| `fix_gridsize_h05/L20_run1_det` | L=20, deterministic |
| `fix_gridsize_h05/L25by25_run1_seed45` | 25×25, seed 45 |
| `fix_gridsize_h05/L3by3_run1_det`, `L3by3_run1_seed42` | 3×3 |
| `fix_gridsize_h05/L40_run1_det` | L=40 |
| `fix_gridsize_h05/L5by5_run1_seed42`, `_best`, `L5by5_run2_seed45_fine_t` | 5×5，多種 run / 時間解析 |
| `fix_gridsize_h05/L6by6_run1_seed45`, `L6by6_run2_seed45_fine_t` | 6×6 |
| `fix_gridsize_h05/L7by1_run1_seed42` | 7×1 條帶 |
| `fix_gridsize_h05/L7by7_run1_seed42`, `_seed45_fine_t`, run2/run3 | 7×7 |

### fix_total_box & diffusion

調整總格點數與擴散係數的對照組：

| 目錄 | 推得/註解 |
|------|-----------|
| `fix_total_box&diffusion/L10_run1_det`, `L10_run1_seed42` | L=10 |
| `fix_total_box&diffusion/L10by1_run1_det`, `L10by1_run1_seed42` | 10×1 |
| `fix_total_box&diffusion/L20_run1_seed42`, `L20by1_run1_seed42` | 20, 20×1 |
| `fix_total_box&diffusion/L3by3_run1_seed42`, `_bubble` | 3×3 |
| `fix_total_box&diffusion/L4_run1_det` | L=4 |

---

## Model B（守恆，h=1 系列）

- **路徑**：`modelB_h_1/`
- **端點**：多為非均勻（high/low 或 high/mid/low）空間分布；部分 run 有 relaxed 端點（`relaxed_states_*.png`）。

### 1D — high_mid_low_not_stable_when_relax

端點為「高/中/低」型態，鬆弛時不穩定，多組 Tmax、N、h、D、dtau：

| 目錄 | 推得參數（由檔名） |
|------|--------------------|
| `L2_T2` | L=2, N=800, D=0.001, dtau=2e-2, Tmax=2 |
| `L2_T10` | L=2, Tmax=10 |
| `L6_T2` | L=6, N=800, D=0.001, dtau=2e-2 |
| `L6_T2_coarse_t` | L=6, N=400, D=0.001, dtau=2e-2 |
| `L10_T2` | L=10, N=800, D=0.001, dtau=2e-2 |
| `L10_T2_coarse_t` | L=10, N=200, D=0.001, dtau=2e-2 |
| `L15_T2` | L=15 |
| `L15by1_run1_seed42` | 15×1, N=400, seed 42 |
| `L15by1_run2_det` ~ `run5_det` | 15×1, N=400, h=1.0 或 0.5, D=0.001–0.002, dtau=1e-2 或 1e-3 |
| `L15by1_run4_det` | N=400, h=0.3, D=0.002, dtau=1e-4 |
| `L30_T2_not_yet_converge` | L=30, N=400, h=1.0, D=0.001, dtau=2e-2（註：尚未收斂） |
| `L2by1_run1_seed42` | L=2, N=400, h=1.0, D=0.01, dtau=2e-2 |

### 1D — high_low（端點左高右低 ↔ 左低右高）

| 目錄 | 推得參數（由檔名） |
|------|--------------------|
| `L20by1_run1_det_relax` | L=20, N=400, h=0.3, D=0.2, dtau=1e-4；有 `relaxed_states_Ly20_Lx1.png` |
| `L20by1_run2_seed42_relax` | 同上系列，seed 42 |
| `L20by1_run3_det_relax_shift_T_0.5` | h=0.3, D=0.2, dtau=5e-4（T 相關 shift） |
| `L20by1_run4_det_relax_shift_T_5` | N=2000, h=0.3, D=0.2, dtau=5e-1 等 |
| `L20by1_run5_det_relax_shift_T_5` | N=2000, h=0.3, D=0.5, dtau=5e-1 |
| `L20by1_run6_det_relax_small_shift_T_5` | 同上系列，small shift |
| `L20by1_run7_det_relax_T_5` | N=2000, h=0.3, D=0.5, dtau=5e-1；有 relaxed_states |
| `L30by1_run1_det_relax` | L=30×1 |
| `L80by1_run1_det_relax`, `run2_det_relax` | L=80×1 |

### 2D — droplet_strip

| 目錄 | 推得/註解 |
|------|-----------|
| `2D/droplet_strip/L10by10_run1_seed45` | 10×10 |
| `2D/droplet_strip/L25by25_run1_seed45`, `run2_seed45` | 25×25 |

### 2D — high_low

| 目錄 | 推得/註解 |
|------|-----------|
| `2D/high_low/L10by10_run1_seed45` | 10×10 |
| `2D/high_low/L10by2_run1_det_wrong?`, run2, run3 | 10×2，註記 wrong?（可能端點或設定有誤） |

### 2D — high_mid_low

| 目錄 | 推得/註解 |
|------|-----------|
| `2D/high_mid_low/L2by2_run1_seed45` | 2×2 |
| `2D/high_mid_low/L4by4_run1_seed45` | 4×4 |
| `2D/high_mid_low/L10by10_run1_seed45` | 10×10 |
| `2D/high_mid_low/L30by30_run1_seed45` | 30×30 |

---

## 參數範圍總覽（由檔名與目錄推得）

| 參數 | Model A | Model B (1D/2D) |
|------|---------|------------------|
| **L (1D)** | 2, 4, 10, 20, 40 | 2, 6, 10, 15, 20, 30, 80 |
| **Lx×Ly (2D)** | 3×3, 5×5, 6×6, 7×7, 25×25 等 | 2×2, 4×4, 10×2, 10×10, 25×25, 30×30 |
| **Ncopy** | 200, 400, 800 | 200, 400, 800, 2000 |
| **Tmax** | 2, 10 等 | 0.5, 2, 5, 10 |
| **D** | 0.001 | 0.001, 0.002, 0.01, 0.2, 0.5 |
| **kappa** | 0.26 | —（Model B 無 κ） |
| **h** | 1（部分 0.5） | 0.3, 0.5, 1.0 |
| **dtau** | 2e-2 | 1e-4, 1e-3, 1e-2, 2e-2, 5e-1 |
| **seed** | 42, 45 | 42, 45 |
| **端點類型** | 均勻（cubic roots） | 非均勻（high/low, high_mid_low）、部分 relaxed |

---

## 輸出類型

- **PNG**：`upward_L*_N*_D*_..._0000XXXXX.png` — 某虛時間 iteration 的 snapshot（熱圖 \(\rho/\theta\)、對稱破缺投影、\(L/H\) 等）。
- **GIF**：`boxes_L*.gif` — 各 box 的 \(\rho\) 隨 path 的動畫。
- **Relaxed states**：`relaxed_states_Ly*_Lx*.png` — 僅部分 Model B 1D run（如 L20, L80）有。
- **2D 模擬動畫**：`2d_sim_Lx*_Ly*.gif` — 出現在部分 2D 目錄。

---

## 備註

1. **Model A** 的「L10_forward / L10_backward」與 **modelA/fix_*** 下的 run 是論文 Zakine2023 PRX 風格的主要對照。
2. **Model B** 的「high_mid_low」與「high_low」對應不同端點設定；`_relax` 表示端點來自鬆弛。
3. 目錄名中的 `_not_yet_converge`、`_wrong?` 表示該 run 尚未收斂或可能有設定錯誤，解讀時需留意。
4. 精確參數以各 run 當時的 script / 日誌為準；此文件僅依檔名與目錄結構整理。

若要補「某目錄對應的完整參數表」或「只列已收斂的 run」，可以指定目錄或條件再細化。
