# KH2018（$\rho,m$）遷移：`active_modelB_1_5D.py` 應該怎麼改

需求：把目前的「單場 Model B」實作（狀態：`rho`，共軛：`theta`）改成你在 `KH2018_Hamiltonian_MAM_migration.md` 裡寫的「雙場 Hamiltonian」（狀態：`rho,m`，共軛：`p_rho,p_m`）。

約定與重要提醒（務必先讀）：

1. `KH2018_Hamiltonian_MAM_migration.md` 給的是 **1D（沿 y）**，所以你要新增 `∂_y`（梯度）算子；`apply_lap_2d()` 在 `Lx=1` 時可視為 `∂_y^2`。
2. `active_modelB_1_5D.py` 現在做的預放鬆（relaxation）是為 Model B 的自由能（裡有 `-D*Lap(rho) - rho + rho^3`），**不等同**你 KH2018 的 deterministic drift。你可以先把程式跑起來驗證數值框架，但若你追求物理一致性，後續仍需把 relaxation/初末狀態也對齊到 KH2018。
3. 下文的「後版 code」是**照 KH 的泛函導數逐項替換**，但 evolve loop 的 `U/V` upwind banded 更新法本質上依賴你目前單場版本的符號慣例。下面我用與你目前單場寫法一致的 reaction 組合（`reaction_U = dH_dq - dH_dp`, `reaction_V = dH_dq + dH_dp`）來推進。你必須用 `p_rho=p_m=0` 的 deterministic limit 做一次一致性驗證。

你要求的格式：以下每一步都提供「修改前 code」與「修改後 code（建議）」；此文件只做**指導**，不會直接改你的程式。

---

## Step 1) 新增 1D 梯度算子 `apply_grad_y`

### 修改前（目前只有 Laplacian）

```python
def apply_lap_2d(field):
    # 完全捨棄有限差分矩陣，改用絕對精準的傅立葉頻域計算 Laplacian
    # field 的形狀為 (N_current, Ly, Lx)
    field_k = sp_fft.fft2(field, axes=(1, 2))
    
    # 乘上我們早就準備好的 2D 頻域 k^2 矩陣
    # k2_2d 的形狀是 (Ly, Lx)，Numpy 會自動漂亮地廣播 (Broadcast) 到所有 Ncopy 身上
    lap_k = -k2_2d * field_k
    
    # 轉回實數空間
    return sp_fft.ifft2(lap_k, axes=(1, 2)).real
```

### 修改後（新增 `apply_grad_y`；沿 y 微分）

把它放在 `apply_lap_2d` 下面（用到你已有的 `KY`）：

```python
def apply_grad_y(field):
    """
    ∂_y field (1D along y), via FFT.
    field shape: (N_current, Ly, Lx)
    returns: same shape (real part)
    """
    field_k = sp_fft.fft2(field, axes=(1, 2))
    grad_k = 1j * KY * field_k          # ∂_y ↔ i*k_y
    return sp_fft.ifft2(grad_k, axes=(1, 2)).real
```

---

## Step 2) 替換 `Hamiltonian()` / `Lagrangian()` 成 KH2018 雙場版本

### 修改前（單場 Model B Hamiltonian / Lagrangian）

```python
def Hamiltonian(h, rho, theta):
	# rho, theta: (Ncopy, Ly*Lx) 以配合 Lagrangian
	rho_3d = rho.reshape(Ncopy, Ly, Lx)
	theta_3d = theta.reshape(Ncopy, Ly, Lx)
	lap_rho = apply_lap_2d(rho_3d)
	lap_theta = apply_lap_2d(theta_3d)
	det_part = apply_lap_2d(-D * lap_rho - rho_3d + rho_3d**3)
	noise_part = -0.5 * aa * theta_3d * lap_theta
	H = np.sum(det_part * theta_3d + noise_part, axis=(1, 2))
	return H

def Lagrangian(h, ds, rho, theta):
	rhoDot      = (np.roll(rho,-1,axis=0) - np.roll(rho,1,axis=0))/(2*ds)
	rhoDot[0,:] = (np.roll(rho,-1,axis=0) - rho)[0,:] /(ds)
	rhoDot[Ncopy-1,:] = (-np.roll(rho,1,axis=0) + rho)[Ncopy-1,:]/(ds)
	Ham = Hamiltonian(h, rho, theta)
	L =  np.sum( rhoDot * theta, axis=1 ) - Ham
	return L
```

### 修改後（建議新增新函數名，避免把舊版刪掉太快）

新增以下兩個函數（依 `KH2018_Hamiltonian_MAM_migration.md` 的 1D 公式；這裡不含你舊 code 的 `aa`）：

```python
def Hamiltonian_KH_1D(Pe, rho, m, p_rho, p_m):
    """
    KH Hamiltonian (1D along y).
    Inputs are flattened in space: (Ncopy, Ly*Lx) like your current Hamiltonian().
    Returns H per path slice: shape (Ncopy,)
    """
    rho3 = rho.reshape(Ncopy, Ly, Lx)
    m3   = m.reshape(Ncopy, Ly, Lx)
    pr3  = p_rho.reshape(Ncopy, Ly, Lx)
    pm3  = p_m.reshape(Ncopy, Ly, Lx)

    drho = apply_grad_y(rho3)
    dm   = apply_grad_y(m3)
    gpr  = apply_grad_y(pr3)
    gpm  = apply_grad_y(pm3)

    C11 = rho3 * (1.0 - rho3)
    C12 = m3   * (1.0 - rho3)
    C22 = C11

    # Drift terms
    term1 = gpr * (drho - Pe * m3 * (1.0 - rho3))
    term2 = gpm * (dm   - Pe * rho3 * (1.0 - rho3))
    term_react = -2.0 * pm3 * m3

    # Conserved-noise quadratic form: (∂p)^T C (∂p) in 1D
    term_quad = C11 * (gpr**2) + 2.0 * C12 * (gpr * gpm) + C22 * (gpm**2)

    # Nonconserved piece
    term_noncons = rho3 * (pm3**2)

    H_density = term1 + term2 + term_react + term_quad + term_noncons
    return np.sum(H_density, axis=(1, 2))


def Lagrangian_KH_1D(Pe, ds, rho, m, p_rho, p_m):
    """
    L = sum( rhoDot*p_rho + mDot*p_m ) - H
    All inputs: (Ncopy, Ly*Lx)
    Returns L per slice: shape (Ncopy,)
    """
    rhoDot = (np.roll(rho, -1, axis=0) - np.roll(rho, 1, axis=0)) / (2*ds)
    mDot   = (np.roll(m,   -1, axis=0) - np.roll(m,   1, axis=0)) / (2*ds)

    rhoDot[0, :]     = (np.roll(rho, -1, axis=0) - rho)[0, :] / ds
    rhoDot[Ncopy-1,:]= (-np.roll(rho, 1, axis=0) + rho)[Ncopy-1, :] / ds
    mDot[0, :]       = (np.roll(m, -1, axis=0) - m)[0, :] / ds
    mDot[Ncopy-1,:]  = (-np.roll(m, 1, axis=0) + m)[Ncopy-1, :] / ds

    Ham = Hamiltonian_KH_1D(Pe, rho, m, p_rho, p_m)
    kin = np.sum(rhoDot * p_rho + mDot * p_m, axis=1)
    return kin - Ham
```

說明：

- 你原程式裡 `H` / `L` 的參數 `h` 沒在 KH Hamiltonian 的公式中出現，因此這裡用 `Pe` 與四個場直接算。你可以照你想保留的介面再把 `h` 接回來，但不需要。
- 如果你希望把 KH 公式裡的噪聲强度 `ε` 映射到你程式的 `aa`，要在 `term_quad` 與 `term_noncons` 之前乘上一個統一縮放係數（這點你需要你自己的物理標定；KH 的文字版目前沒有顯式 ε）。

---

## Step 3) 在參數段新增 `Pe`（並決定 `aa` 是否還要保留）

### 修改前（目前只有 Model B 的噪聲幅度 `aa`）

```python
aa = 2.   #noise amplitude
...
D     = 1
...
```

### 修改後（新增 `Pe`；並把 `aa` 用不用取決於你是否做噪聲縮放）

例如：

```python
Pe = 1.0   # TODO: pick your Péclet-like parameter (must match KH paper's convention)

# 你舊的 aa 現在在 KH Hamiltonian 裡可能不會用到
aa = 2.   # TODO: keep only if you decide to map epsilon -> aa
```

---

## Step 4) 陣列建立：新增 `m, p_rho, p_m`，以及對應的 `U/V` 組

### 修改前（單場）

```python
rho   = np.zeros((Ncopy,Ly,Lx), dtype=complex)

theta = np.zeros((Ncopy,Ly,Lx), dtype=complex)

U = rho + theta
V = rho - theta
```

以及初始化前後使用 `theta`：

```python
theta[0] = 0.0 + 0j
theta[Ncopy-1] = 0.0 + 0j
U = rho + theta
V = rho - theta
```

### 修改後（雙場）

新增 4 個場（狀態兩個、共軛兩個）：

```python
rho   = np.zeros((Ncopy, Ly, Lx), dtype=complex)
m     = np.zeros((Ncopy, Ly, Lx), dtype=complex)

p_rho = np.zeros((Ncopy, Ly, Lx), dtype=complex)
p_m   = np.zeros((Ncopy, Ly, Lx), dtype=complex)

# 兩套 U/V（每個 (q,p) 一套）
U_rho = rho + p_rho
V_rho = rho - p_rho

U_m   = m + p_m
V_m   = m - p_m
```

同時要新增（或重用）Fourier buffer 與更新用的 reaction arrays。例如，你原本只有：

```python
U_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)
V_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)
U2_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)
V2_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)
reaction_U = np.zeros((Ncopy,Ly,Lx), dtype=complex)
reaction_V = np.zeros((Ncopy,Ly,Lx), dtype=complex)
```

建議改成「至少」：

```python
U_rho_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)
V_rho_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)
U_m_Fourier   = np.zeros((Ncopy,Ly,Lx), dtype=complex)
V_m_Fourier   = np.zeros((Ncopy,Ly,Lx), dtype=complex)

U_rho_new_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)
V_rho_new_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)
U_m_new_Fourier   = np.zeros((Ncopy,Ly,Lx), dtype=complex)
V_m_new_Fourier   = np.zeros((Ncopy,Ly,Lx), dtype=complex)

reaction_U_rho = np.zeros((Ncopy,Ly,Lx), dtype=complex)
reaction_V_rho = np.zeros((Ncopy,Ly,Lx), dtype=complex)
reaction_U_m   = np.zeros((Ncopy,Ly,Lx), dtype=complex)
reaction_V_m   = np.zeros((Ncopy,Ly,Lx), dtype=complex)
```

---

## Step 5) 初始化與邊界條件：設定 `m`、`p_rho`、`p_m`

### 修改前（單場）

```python
theta = np.zeros((Ncopy,Ly,Lx), dtype=complex)
...
if os.path.exists(resume_file):
    data = np.load(resume_file)
    rho = data['rho']
    theta = data['theta']
    ...
else:
    start_iter = 0

U = rho + theta
V = rho - theta
```

在每次更新後都強制端點：

```python
theta[0] = 0.0 + 0j
theta[Ncopy-1] = 0.0 + 0j
```

### 修改後（雙場）

1) checkpoint resume：把 `theta` 改成 3 個新鍵值（`m,p_rho,p_m`）  
2) 未 resume：初始化策略要決定（建議先最簡單先跑通）

建議最簡單可行版本：

```python
# 未 resume 時：
start_iter = 0

U_rho = rho + p_rho
V_rho = rho - p_rho
U_m   = m + p_m
V_m   = m - p_m
```

端點（每次 V/U 更新後也要再強制一次）：

```python
p_rho[0] = 0.0 + 0j
p_rho[Ncopy-1] = 0.0 + 0j

p_m[0] = 0.0 + 0j
p_m[Ncopy-1] = 0.0 + 0j
```

關於 `m` 的端點值 `m1,m2`：

- 你目前程式的 `rho1,rho2` 是用 Model B equilibria（`rho` 在 -1 與 +1 左右）。在 KH2018 版本裡 `m` 的 deterministic drift 是 `-2m`，所以最自然的端點是 `m1=m2=0`（或非常小）。
- 因此你可以先做：`m[0]=m1=0`、`m[Ncopy-1]=m2=0`，然後讓 evolve loop 自然生成非零 `m`。

在「每次更新後」強制：

```python
m[0] = 0.0 + 0j
m[Ncopy-1] = 0.0 + 0j
```

初始化路徑中間 `m[j]`：

- 建議先全 0：`m[j,:,:]=0`
- 若你發現 saddle 不動（或 p 永遠為 0），再嘗試給 `m` 或 `p_m` 很小的隨機初值作為起始種子。

---

## Step 6) 主要修改：evolve loop 的 reaction / 更新方程

這一步是最大的變更點：你原本在每個迭代裡計算 `dH_drho` 和 `dH_dtheta`（單場），現在要計算四個泛函導數（雙場）：

- `dH_drho` = δH/δρ
- `dH_dm`   = δH/δm
- `dH_dprho`= δH/δp_rho
- `dH_dpm`  = δH/δp_m

然後用與你單場一致的組合去做 reaction：

- `reaction_U_* = dH_dq - dH_dp`
- `reaction_V_* = dH_dq + dH_dp`

### 修改前（單場：計算 dH 與 reaction）

```python
# ================= UPDATE U =================
lap_rho   = apply_lap_2d(rho)
lap_theta = apply_lap_2d(theta)
# Model B: dH_drho = -D*lap(lap_theta) - (1-3rho^2)*lap_theta
dH_drho   = -D * apply_lap_2d(lap_theta) - (1 - 3*rho**2) * lap_theta
# Model B: dH_dtheta = lap(-D*lap_rho - rho + rho^3) - aa*lap_theta
dH_dtheta = apply_lap_2d(-D * lap_rho - rho + rho**3) - aa * lap_theta
reaction_U = dH_drho - dH_dtheta

# V step 類似...
reaction_V = dH_drho + dH_dtheta
```

### 修改後（雙場：照 KH2018 δH/δ* 計算）

把 `UPDATE U` 這段中間反應部分替換成下列邏輯（請注意：這是**示意版**；你要把它接回你現有的 FFT 更新框架；不會改你整段 banded solver 的結構）。

首先在 loop 內新增「梯度與二階導」計算：

```python
rho_prime = apply_grad_y(rho)
m_prime   = apply_grad_y(m)
pr_prime  = apply_grad_y(p_rho)
pm_prime  = apply_grad_y(p_m)

lap_pr = apply_lap_2d(p_rho)  # = ∂_y^2 p_rho (因 Lx=1)
lap_pm = apply_lap_2d(p_m)    # = ∂_y^2 p_m
```

接著用 KH 公式計算四個泛函導數：

```python
C11 = rho * (1.0 - rho)
C12 = m   * (1.0 - rho)
C22 = C11

# δH/δp_rho = -∂y[ rho' - Pe m(1-rho) + 2(C11 pr' + C12 pm') ]
br_pr = rho_prime - Pe * m * (1.0 - rho) + 2.0 * (C11 * pr_prime + C12 * pm_prime)
dH_dprho = -apply_grad_y(br_pr)

# δH/δp_m = -∂y[ m' - Pe rho(1-rho) + 2(C12 pr' + C22 pm') ] -2m + 2 rho p_m
br_pm = m_prime - Pe * rho * (1.0 - rho) + 2.0 * (C12 * pr_prime + C22 * pm_prime)
dH_dpm = -apply_grad_y(br_pm) - 2.0*m + 2.0*rho*p_m

# δH/δρ
dH_drho = (-lap_pr
           + Pe*m*pr_prime
           - Pe*(1.0-2.0*rho)*pm_prime
           + (1.0-2.0*rho)*(pr_prime**2 + pm_prime**2)
           - 2.0*m*pr_prime*pm_prime
           + (p_m**2))

# δH/δm
dH_dm = (-lap_pm
         - Pe*(1.0-rho)*pr_prime
         - 2.0*p_m
         + 2.0*(1.0-rho)*pr_prime*pm_prime)
```

最後得到 reaction（照你原本單場的 pattern）：

```python
reaction_U_rho = dH_drho - dH_dprho
reaction_V_rho = dH_drho + dH_dprho

reaction_U_m   = dH_dm - dH_dpm
reaction_V_m   = dH_dm + dH_dpm
```

### 修改後（用同樣 banded solver 更新 U_rho/U_m）

你原本 `UPDATE U` 的更新流程：

1. `U_Fourier = fft2(U)`
2. `reaction_U_Fourier = fft2(reaction_U)`
3. `RHS_U_Fourier = U_Fourier + dtau*reaction_U_Fourier`
4. `RHS_U_Fourier[Ncopy-1] = -V_Fourier[Ncopy-1] + 2*rho2k`
5. solve banded for each spatial mode to get `U_new`
6. `U = ifft2(U_new)`
7. `rho = 0.5*(U+V)`, `theta=0.5*(U-V)`
8. set boundary and recompute U,V

雙場你要對 `U_rho` 和 `U_m` 各做一次（但共享同一套 `A_banded`/`k4_flat/k2_flat`）。

示意：

```python
# ---- UPDATE U_rho ----
U_rho_Fourier = sp_fft.fft2(U_rho, axes=(1,2))
reaction_U_rho_Fourier = sp_fft.fft2(reaction_U_rho, axes=(1,2))

RHS_U_rho_Fourier = U_rho_Fourier + dtau * reaction_U_rho_Fourier
RHS_U_rho_Fourier[Ncopy-1,:,:] = -V_rho_Fourier[Ncopy-1,:,:] + 2.0 * rho2k

# 逐 col solve_banded -> U_rho_new_Fourier
# U_rho = ifft2(U_rho_new_Fourier)


# ---- UPDATE U_m ----
U_m_Fourier = sp_fft.fft2(U_m, axes=(1,2))
reaction_U_m_Fourier = sp_fft.fft2(reaction_U_m, axes=(1,2))

RHS_U_m_Fourier = U_m_Fourier + dtau * reaction_U_m_Fourier
RHS_U_m_Fourier[Ncopy-1,:,:] = -V_m_Fourier[Ncopy-1,:,:] + 2.0 * m2k

# 逐 col solve_banded -> U_m_new_Fourier
# U_m = ifft2(U_m_new_Fourier)


# ---- reconstruct (same as your single-field) ----
rho   = 0.5 * (U_rho + V_rho)
p_rho = 0.5 * (U_rho - V_rho)

m     = 0.5 * (U_m + V_m)
p_m   = 0.5 * (U_m - V_m)
```

然後再強制端點並重建 U/V：

```python
rho[0] = rho1
rho[Ncopy-1] = rho2

m[0] = m1   # often 0
m[Ncopy-1] = m2   # often 0

p_rho[0] = 0.0 + 0j
p_rho[Ncopy-1] = 0.0 + 0j

p_m[0] = 0.0 + 0j
p_m[Ncopy-1] = 0.0 + 0j

U_rho = rho + p_rho
V_rho = rho - p_rho
U_m   = m   + p_m
V_m   = m   - p_m
```

### 修改後（UPDATE V_rho / UPDATE V_m）

`UPDATE V` 的邏輯與單場幾乎完全一樣，只是把 reaction_U/dH_dtheta 換成 reaction_V_* 與各自 U/V 變數即可：

```python
# reaction_V_rho computed above
# reaction_V_m computed above
RHS_V_rho_Fourier = V_rho_Fourier + dtau * reaction_V_rho_Fourier
RHS_V_rho_Fourier[0,:,:] = -U_rho_Fourier[0,:,:] + 2.0 * rho1k

RHS_V_m_Fourier = V_m_Fourier + dtau * reaction_V_m_Fourier
RHS_V_m_Fourier[0,:,:] = -U_m_Fourier[0,:,:] + 2.0 * m1k
```

最後 reconstruct：

```python
rho   = 0.5 * (U_rho + V_rho)
p_rho = 0.5 * (U_rho - V_rho)
m     = 0.5 * (U_m + V_m)
p_m   = 0.5 * (U_m - V_m)

# 再次強制端點（同 Step 5 的 block）
```

---

## Step 7) 收斂檢查與 Plot：把 `Hamiltonian()` / `Lagrangian()` 換成 KH 版本

### 修改前（單場）

```python
rho_1d_check = rho.reshape(Ncopy, -1)
theta_1d_check = theta.reshape(Ncopy, -1)
H_path = Hamiltonian(h, rho_1d_check, theta_1d_check).real
...
Lag = Lagrangian(h, dnu, rho_1d, theta_1d).real
...
Hamiltonian(h,rho_1d, theta_1d)
```

### 修改後（雙場）

把 `theta` 換成 `p_rho,p_m`，把 `Hamiltonian/Lagrangian` 換成 `Hamiltonian_KH_1D/Lagrangian_KH_1D`：

```python
rho_1d_check   = rho.reshape(Ncopy, -1)
m_1d_check     = m.reshape(Ncopy, -1)
pr_1d_check    = p_rho.reshape(Ncopy, -1)
pm_1d_check    = p_m.reshape(Ncopy, -1)

H_path = Hamiltonian_KH_1D(Pe, rho_1d_check, m_1d_check, pr_1d_check, pm_1d_check).real
...
Lag = Lagrangian_KH_1D(Pe, dnu, rho_1d, m_1d, pr_1d, pm_1d).real
```

Plot 部分：

- 你原本只畫 `rho` 與 `theta`，現在共軛有兩個。你可以先最簡單：把 `theta` subplot 換成 `p_rho`（先跑通），`p_m` 可先不畫或改成第二張。
- 若你要維持 4 subplots，可以替換成：`rho`, `p_rho`, `L`, `H`，並把 `p_m` 先忽略或在圖上另存。

---

## Step 8) checkpoint 與動畫：保存載入也要改鍵名

### 修改前（單場保存）

```python
np.savez_compressed('checkpoints/checkpoint.npz',
    rho=rho, theta=theta, iteration=i, Lx=Lx, Ly=Ly, h=h, Ncopy=Ncopy,
    Tmax=Tmax, aa=aa, D=D, dtau=dtau, upward=upward, iterations=i, plotStep=plotStep)
```

### 修改後（雙場保存）

```python
np.savez_compressed('checkpoints/checkpoint.npz',
    rho=rho, m=m, p_rho=p_rho, p_m=p_m,
    iteration=i, Lx=Lx, Ly=Ly, h=h, Ncopy=Ncopy, Tmax=Tmax,
    Pe=Pe, D=D, dtau=dtau, upward=upward, iterations=i, plotStep=plotStep)
```

同時你 resume block：

- 原本 `data['theta']` 要換成 `data['p_rho']` 與 `data['p_m']`
- `data['rho']` 與 `data['m']` 要一致

動畫 `animate_2d_heatmap(rho.real, theta.real, ...)` 也要換成你想顯示的量（例如顯示 `p_rho` 或 `p_m`）。

---

## Step 9)（必做）數值一致性測試：在程式裡加一次臨時檢查

在你完成 evolve loop 的替換後，建議做下列檢查（不用改文件，只在本地暫時 print）：

1. 設定 `p_rho=0` 且 `p_m=0`、同時固定 `m=0` 或使用你的初始化值  
2. 檢查 `dH_dprho` 與 `dH_dpm` 是否只依賴 `rho,m` 與它們的空間導數，且 deterministic drift 的符號符合你希望的 KH 方程
3. 檢查 `δH/δp_rho` 的結果在數值上是否等於 `∂_y^2 rho - Pe ∂_y[m(1-ρ)]`（忽略離散誤差）

這能快速抓出 reaction 組合或 `∂_y` 符號是否寫反。

---

## 你接下來要我做什麼（可選）

如果你希望我把 Step 6 的 `UPDATE U/V` 區塊也直接寫成「能貼進程式的完整替換版 code（含變數名稱對齊）」並減少你手工拼裝，我可以再根據你想採用的實作選項做：

1. 你要用兩套 `U_Fourier/V_Fourier` buffers 還是重用一套 buffers？
2. 你希望畫圖 subplot 顯示 `p_rho` 還是 `p_m`？
3. `m1,m2` 你想先固定為 0 還是從 deterministic 的 end state 推？（預設先 0）

你回覆這三點後，我可以把 Step 6/7 的 code block 寫到更接近「直接貼上即可運行」的程度。

