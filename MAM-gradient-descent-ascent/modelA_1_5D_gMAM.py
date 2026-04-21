import math
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy.linalg import solve_banded
start_time = time.time()
PI = math.pi
##############################
""" DEFINE FUNCTIONS """
###############################
def Hamiltonian(h, rho, theta): #return an array of size Ncopy, axis=1 sums the colums
	L_spatial = rho.shape[1]
	H = np.sum( (D*(np.roll(rho,-1, axis=1) +  np.roll(rho,1, axis=1) -2* rho)/h**2 + rho -rho**3 + kappa*np.outer(np.mean(rho**2,axis=1), np.ones(L_spatial)))*theta + 0.5*aa*theta**2     ,  axis=1)
	return H

def Lagrangian(h, ds, rho, theta): #return an array of size Ncopy, axis=1 sums the colums
	#rhoDot = (np.roll(rho,-1,axis=0) - rho)/ds
	#L = h*np.sum(np.power(theta.real,2),axis=1)
	rhoDot      = (np.roll(rho,-1,axis=0) - np.roll(rho,1,axis=0))/(2*ds)
	rhoDot[0,:] = (np.roll(rho,-1,axis=0) - rho)[0,:] /(ds)
	rhoDot[Ncopy-1,:] = (-np.roll(rho,1,axis=0) + rho)[Ncopy-1,:]/(ds)
	Ham = Hamiltonian(h, rho, theta)
	L =  np.sum( rhoDot * theta, axis=1 ) - Ham
	return L
from scipy.interpolate import interp1d
import scipy.integrate as integrate

def reparameterize_system(rho, theta):
    """
    gMAM 核心：使用統一的物理狀態弧長，對 Model A 的系統進行重分佈
    """
    N_t = rho.shape[0]
    
    # 1. 只有物理狀態 (rho) 有資格定義「相空間距離」
    diff_rho = np.diff(rho.real, axis=0)
    step_lengths = np.sqrt(np.sum(diff_rho**2, axis=(1, 2)))
    
    # 2. 計算統一的歸一化弧長 s
    l_cum = np.zeros(N_t)
    l_cum[1:] = np.cumsum(step_lengths)
    
    # 防呆：避免路徑未初始化導致除以 0
    if l_cum[-1] == 0:
        return rho, theta
        
    s_old = l_cum / l_cum[-1]
    s_new = np.linspace(0, 1, N_t)
    
    # 3. 建立插值器 (強制取實部以消除 FFT 微小虛部)
    rho_flat = rho.real.reshape(N_t, -1)
    theta_flat = theta.real.reshape(N_t, -1)

    rho_new = interp1d(s_old, rho_flat, axis=0, kind='linear')(s_new).reshape(rho.shape)
    theta_new = interp1d(s_old, theta_flat, axis=0, kind='linear')(s_new).reshape(theta.shape)
    
    # 加回 0j 以配合後續的 FFT 運算型態
    return rho_new + 0j, theta_new + 0j
def Geometric_Action_Density(rho_1d, theta_1d, Ncopy):
    """
    gMAM 專用：計算歸一化弧長上的幾何作功密度，並加上物理過濾
    """
    d_alpha = 1.0 / Ncopy
    
    # 相對於歸一化弧長 s 的導數 (rho')
    rhoPrime = (np.roll(rho_1d, -1, axis=0) - np.roll(rho_1d, 1, axis=0)) / (2 * d_alpha)
    rhoPrime[0, :] = (np.roll(rho_1d, -1, axis=0) - rho_1d)[0, :] / d_alpha
    rhoPrime[Ncopy-1, :] = (-np.roll(rho_1d, 1, axis=0) + rho_1d)[Ncopy-1, :] / d_alpha
    
    # kin_geom_raw = theta * rho' (theta 即為共軛動量 p_rho)
    kin_geom_raw = np.sum(rhoPrime * theta_1d, axis=1)
    
    # 物理過濾：消除 Relaxation tail 造成的負值雜訊
    kin_geom_clean = np.maximum(kin_geom_raw, 0.0)
    return kin_geom_clean
def compute_lambda_gmam(rho, dH_dtheta, H_path, ds):
	"""
	依據 H=0 條件，解一元二次方程式計算每個 slice 的 lambda (拉格朗日乘子)。
	輸入參數:
		- rho: (Ncopy, Ly, Lx) 當前的密度場
		- dH_dtheta: (Ncopy, Ly, Lx) 迴圈中算出的偏微分項 (漂移項 V)
		- H_path: (Ncopy,) 當前路徑每個點的 Hamiltonian 數值
		- ds: float 幾何弧長的間距 (通常是 1.0 / (Ncopy - 1))
	"""
	N_t = rho.shape[0]

	# 1. 計算 rho' (對 ds 的中心差分，邊界使用前向/後向差分)
	rho_prime = np.zeros_like(rho)
	rho_prime[1:-1] = (rho[2:] - rho[:-2]) / (2.0 * ds)
	rho_prime[0]    = (rho[1] - rho[0]) / ds
	rho_prime[-1]   = (rho[-1] - rho[-2]) / ds

	# 2. 計算 A = ||rho'||^2 (在空間上加總)
	A = np.sum(rho_prime.real**2, axis=(1, 2))

	# 3. 計算 B = <rho', dH_dtheta> (在空間上加總)
	B = np.sum(rho_prime.real * dH_dtheta.real, axis=(1, 2))

	# 4. C = H
	C = H_path.real

	# 5. 求解 lambda = (B + sqrt(max(0, B^2 - 4AC))) / 2A
	lambdas = np.zeros(N_t)

	for i in range(1, N_t - 1):
		if A[i] < 1e-12: # 防呆：如果該點完全沒形變，避免除以零
			lambdas[i] = 0.0
		else:
			discriminant = B[i]**2 - 4.0 * A[i] * C[i]
			# 強制把數值誤差導致的微小負數判別式歸零
			discriminant = max(0.0, discriminant)
			lambdas[i] = (B[i] + np.sqrt(discriminant)) / (2.0 * A[i])

	lambdas = np.clip(lambdas, 1e-2, 100.0)
	# 邊界條件 (穩態不需要時間演化，lambda = 0)
	lambdas[0] = 0.0
	lambdas[-1] = 0.0
    
	return lambdas
################################
""" Start MAM """
##############################

Lx = 1
Ly = 50
h  = 1
Ncopy = 2000 
Tmax = 600.


s = np.linspace(0,Tmax,Ncopy)
ds  = s[1]-s[0]
dnu = s[1]-s[0]
aa = 2.   #noise amplitude


upward = True # choose if path from -1 to +1 (upward), or the opposite
previous_data = True
reverse = False
threshold = 1000

D     = 5
kappa = 0.1

solRho = np.sort(np.roots([-1, kappa, 1, 0])) # rho-, rhos, rho+ 

dtau = 0.02

iterations = 40000
plotStep   = 200
resume_file = "checkpoints/checkpoint_local.npz"
initial_guess = "checkpoints/checkpoint_gMAM2.npz"
r = dtau/dnu
print('conditions: Ly,Lx,Ncopy =', Ly,Lx,Ncopy, 'h =', h, 'D =', D, 'kappa =', kappa, 'dtau =', dtau, 'iterations =', iterations, 'Tmax =', Tmax)

# === IMEX 穩定化（對應 modified 的 k4、gamma）===
gamma = 0.5   # 與 modified 相同，壓制高頻
# 2D Fourier 波數（與 fft2(..., axes=(1,2)) 對應）
kx = 2 * PI * np.fft.fftfreq(Lx, d=h)
ky = 2 * PI * np.fft.fftfreq(Ly, d=h)
# 若 Lx 或 Ly 為 1，確保為 1D 且長度正確
if np.isscalar(kx) or kx.size == 1:
    kx = np.atleast_1d(kx)
if np.isscalar(ky) or ky.size == 1:
    ky = np.atleast_1d(ky)
KY, KX = np.meshgrid(ky, kx, indexing='ij')   # shape (Ly, Lx)
k2_2d = KX**2 + KY**2
k4_2d = k2_2d**2
k4_flat = k4_2d.ravel()   # shape (Ly*Lx,) 與 U_Fourier.reshape(Ncopy,-1) 的 column 對應
k2_flat = k2_2d.ravel()   # [新增] 準備 k2 的 flat 陣列

############# PATH-TIME UPWIND MATRICES（per-mode + IMEX）##########
N_space = Ly * Lx
A_banded = np.zeros((N_space, 3, Ncopy))
B_banded = np.zeros((N_space, 3, Ncopy))

for col in range(N_space):
	stab = dtau * gamma * (D * k4_flat[col] + 2.0 * k2_flat[col])
	# A: second order + stab on diagonal
	A_banded[col, 2, :] = 1. + 3*r/2. + stab
	A_banded[col, 2, Ncopy-2] = 1. + r + stab
	A_banded[col, 2, Ncopy-1] = 1. + stab

	# 2. First upper diagonal
	A_banded[col, 1, 1:] = -2*r
	A_banded[col, 1, Ncopy-1] = -r

	# 3. Second upper diagonal
	A_banded[col, 0, 2:] = r/2.

	# ================= B 矩陣 (Lower Triangular) =================
	# scipy 規定下三角帶狀矩陣：row 0 是主對角線，row 1 是一階下方，row 2 是二階下方

	# 1. Main diagonal
	B_banded[col, 0, :] = 1. + 3*r/2. + stab
	B_banded[col, 0, 1] = 1. + r + stab
	B_banded[col, 0, 0] = 1.0 + stab # (邊界不加 stab)

	# 2. First lower diagonal
	B_banded[col, 1, :-1] = -2*r
	B_banded[col, 1, 0] = -r

	# 3. Second lower diagonal
	B_banded[col, 2, :-2] = r/2.

############# 2D SPARSE SOLVER SETUP ##########
def build_rect_2d_laplacian(Nx, Ny, h):
	# ----------------------------------------
	# 1. 定義一個通用的 helper 函數來產生 1D 矩陣
	# ----------------------------------------
	def make_1d_diff_matrix(N):
		# 建立 1D 差分矩陣: 對角線 -2, 左右 1
		vals = [np.ones(N), -2*np.ones(N), np.ones(N)]
		offsets = [-1, 0, 1]
		D = sparse.diags(vals, offsets, shape=(N, N)).tolil()
		
		# 週期性邊界 (Periodic BC)
		D[0, N-1] = 1
		D[N-1, 0] = 1
		return sparse.csr_matrix(D)

	# ----------------------------------------
	# 2. 分別建立 X 和 Y 的 1D 算子
	# ----------------------------------------
	Dx = make_1d_diff_matrix(Nx) # 大小 Nx * Nx
	Dy = make_1d_diff_matrix(Ny) # 大小 Ny * Ny

	Ix = sparse.identity(Nx)     # 大小 Nx * Nx
	Iy = sparse.identity(Ny)     # 大小 Ny * Ny

	# ----------------------------------------
	# 3. 利用 Kronecker Product 組合 (核心改動)
	# ----------------------------------------
	# 注意順序：
	# row-major (C-style) flatten 時，Y 是快速變化的索引 (Fast index)，X 是慢速 (Slow index)
	# 所以 X 方向的微分算子要跟 Iy 做 tensor product (作用在大的 block 上)
	# Y 方向的微分算子要跟 Ix 做 tensor product (作用在每個 block 內部)

	# 擴散算子 = (d^2/dx^2) + (d^2/dy^2)
	if Nx == 1:
		# 僅 y 方向：Lap = d^2/dy^2 = Dy/h^2（與 1D 鏈 Ly 點相同）
		return Dy / (h**2)
	if Ny == 1:
		# 僅 x 方向：Lap = d^2/dx^2 = Dx/h^2
		return Dx / (h**2)
	else:
		Lap2D = sparse.kron(Dx, Iy) + sparse.kron(Ix, Dy)
		return Lap2D / (h**2)

# 1. 建立長方形的 2D Laplacian (大小是 (Lx*Ly) x (Lx*Ly))
Lap_Matrix = build_rect_2d_laplacian(Lx, Ly, h)

# 2. 建立隱式算子
I_total = sparse.identity(Lx * Ly, format='csr')
Operator_Matrix = I_total - dtau * D * Lap_Matrix

# 3. 預先分解
solve_2D = factorized(Operator_Matrix)


########## ARRAY CREATION

rho   = np.zeros((Ncopy,Ly,Lx), dtype=complex)

theta = np.zeros((Ncopy,Ly,Lx), dtype=complex)

U = np.zeros((Ncopy,Ly,Lx), dtype=complex)
U_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)

V = np.zeros((Ncopy,Ly,Lx), dtype=complex)
V_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)

U2_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)
V2_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)

reaction_U = np.zeros((Ncopy,Ly,Lx), dtype=complex)
reaction_V = np.zeros((Ncopy,Ly,Lx), dtype=complex)

reaction_U_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)
reaction_V_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)

######  Boundary conditions
if(upward==True):
	rho1 = solRho[0] * np.ones((Ly,Lx), dtype=complex)
	rho1k = np.fft.fft2(rho1)
	rho2 = solRho[2] * np.ones((Ly,Lx), dtype=complex)
	rho2k = np.fft.fft2(rho2)
else:
	rho1 = solRho[2] * np.ones((Ly,Lx), dtype=complex)
	rho1k = np.fft.fft2(rho1)
	rho2 = solRho[0] * np.ones((Ly,Lx), dtype=complex)
	rho2k = np.fft.fft2(rho2)

extract_ratio = 1.0

######  Initial guess
if previous_data == True:
	if os.path.exists(initial_guess):
		data = np.load(initial_guess)
		rho_old = data['rho']
		theta_old = data['theta']
		T_old = data['Tmax']
		Ncopy_old = data['Ncopy']
		Ly_old = data['Ly']
		Lx_old = data['Lx']

		# extract rho and theta from 0 to 0.9*T_old
		rho_old = rho_old[:int(extract_ratio*Ncopy_old),:,:]
		theta_old = theta_old[:int(extract_ratio*Ncopy_old),:,:]
		# rho_old = rho_old[int((1-extract_ratio)*Ncopy_old+1):,:,:]
		# theta_old = theta_old[int((1-extract_ratio)*Ncopy_old+1):,:,:]
		Ncopy_old = int(extract_ratio*Ncopy_old)
		T_old = T_old * extract_ratio

		# shift rho and theta by Ly//2
		# rho_old = np.roll(rho_old, Ly//2, axis=1)
		# theta_old = np.roll(theta_old, Ly//2, axis=1)
		import scipy.ndimage as ndimage
		zoom_factors = (Ncopy / Ncopy_old, Ly / Ly_old, Lx / Lx_old)
		rho = ndimage.zoom(rho_old.real, zoom_factors, order=1).astype(complex)
		theta = ndimage.zoom(theta_old.real, zoom_factors, order=1).astype(complex)
		if reverse == True:
			# reverse rho in time
			rho = rho[::-1,:,:]
			theta = theta[::-1,:,:]
		rho[0,:,:]       = rho1
		rho[Ncopy-1,:,:] = rho2
		print("rho shape", rho.shape)
		print("theta shape", theta.shape)
else:
	# deterministically set the initial condition
	amp = 0.8
	y_coords = np.arange(Ly)
	x_coords = np.arange(Lx)
	Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

	# randomly set the initial condition
	noise_amp = 0.1  # noise amplitude
	np.random.seed(42 if Lx == 1 else None)

	max_radius = np.sqrt((Ly/2)**2 + (Lx/2)**2) 
	if upward == True:
		for j in range(1,Ncopy-1):
			tt = float(j)/Ncopy
			linear = rho1*(1-tt) + tt*rho2
			# bubble growth in 1D chain (smooth initial condition)
			# current_radius = tt * max_radius
			# dist = np.sqrt((Y - Ly/2)**2 + (X - Lx/2)**2)
			# w = np.sqrt(D) / h  
			# if w < 1.0: w = 1.0 
			# profile = 0.5 * (solRho[0] - solRho[2]) * np.tanh((dist - current_radius) / w) + 0.5 * (solRho[0] + solRho[2])
			# rho[j, :, :] = profile + 0j
			# deterministic initial condition
			bump = amp * np.square(np.sin(PI * Y / Ly)) * np.power(np.sin(PI * tt), 2)
			# random initial condition
			# bump = noise_amp*np.random.normal(0, 1, size=(Ly,Lx))*np.sin(PI*tt)
			rho[j,:,:] = linear + bump
	else: # downward
		for j in range(1,Ncopy-1):
			tt = float(j)/Ncopy
			linear = rho1*(1-tt) + tt*rho2
			# bubble growth in 1D chain (smooth initial condition)
			# current_radius = tt * max_radius
			# dist = np.sqrt((Y - Ly/2)**2 + (X - Lx/2)**2)
			# w = np.sqrt(D) / h  
			# if w < 1.0: w = 1.0 
			# profile = 0.5 * (solRho[2] - solRho[0]) * np.tanh((dist - current_radius) / w) + 0.5 * (solRho[2] + solRho[0])
			# rho[j, :, :] = profile + 0j
			# bump = amp * np.square(np.sin(PI * Y / Ly)) * np.power(np.sin(PI * tt), 2)
			rho[j, :, :] = linear #+ bump
U = rho + theta
V = rho - theta


if os.path.exists(resume_file):
	data = np.load(resume_file)
	rho = data['rho']
	theta = data['theta']
	iteration = data['iteration']
	Lx = data['Lx']
	Ly = data['Ly']
	h = data['h']
	Ncopy = data['Ncopy']
	Tmax = data['Tmax']
	aa = data['aa']
	D = data['D']
	kappa = data['kappa']
	dtau = data['dtau']
	upward = data['upward']
	end_iterations = data['iterations']
	plotStep = data['plotStep']
	start_iter = end_iterations + 1
else:
	start_iter = 0
###### Evolve Loop (GDA Algorithm 1: path-time upwind + full reaction) ######
# Apply 2D Laplacian to each path slice: (Ncopy, Ly*Lx)
def apply_lap_2d(field):
	# field (Ncopy, Ly, Lx) -> lap (Ncopy, Ly, Lx)
	flat = field.reshape(Ncopy, -1)
	lap_flat = (Lap_Matrix.dot(flat.T)).T
	return lap_flat.reshape(Ncopy, Ly, Lx)
start_time = time.time()
print("start_iter", start_iter)
print("iterations", iterations)
def compute_physical_action(rho, theta, D, kappa, ds):
    """
    只積分爬坡段 (Uphill) 的 Action S。
    ds 為幾何網格間距 (通常是 1.0 / (N_t - 1))。
    """
    N_t = rho.shape[0]
    # 1. 空間梯度與漂移力
    # 注意：這裡的 apply_lap_2d 需對應你之前的定義
    lap_rho = apply_lap_2d(rho)
    mean_rho2 = np.mean(rho**2, axis=(1, 2), keepdims=True)
    drift_V = D * lap_rho + rho - rho**3 + kappa * mean_rho2

    # 2. 幾何切線 (中心差分)
    rho_prime = np.zeros_like(rho)
    rho_prime[1:-1] = (rho[2:] - rho[:-2]) / (2.0 * ds)
    rho_prime[0] = (rho[1] - rho[0]) / ds
    rho_prime[-1] = (rho[-1] - rho[-2]) / ds

    # 3. 幾何 Action 密度 L = <theta, rho'>
    L_geom = np.sum(theta.real * rho_prime.real, axis=(1, 2))
    
    # 4. 判定爬坡段：內積 <rho', V> < 0
    inner_prod = np.sum(rho_prime.real * drift_V.real, axis=(1, 2))
    uphill_mask = inner_prod < 0
    
    # 5. 積分：只採計爬坡段
    # 注意：ds 這裡代表幾何步長
    S_uphill = np.sum(L_geom[uphill_mask]) * ds
    
    return S_uphill, L_geom, uphill_mask
for i in range(start_iter, iterations+1):
	U[:] = rho + theta
	V[:] = rho - theta
	lap_rho   = apply_lap_2d(rho)
	mean_rho2_global  = np.mean(rho**2, axis=(1,2), keepdims=True)
	dH_dtheta = D * lap_rho + rho - rho**3 + kappa * mean_rho2_global + aa * theta
	
	rho_1d_for_H = rho.reshape(Ncopy, -1)
	theta_1d_for_H = theta.reshape(Ncopy, -1)
	H_current = Hamiltonian(h, rho_1d_for_H, theta_1d_for_H).real # 取得形狀為 (Ncopy,) 的 H 陣列
	
	# [新增] 2. 計算動態拉格朗日乘子
	ds = 1.0 / (Ncopy - 1)
	lambdas = compute_lambda_gmam(rho, dH_dtheta, H_current, ds)
	# ================= UPDATE U =================
	# 1. 完整 reaction（含空間擴散 D*Lap）
	lap_rho   = apply_lap_2d(rho)
	lap_theta = apply_lap_2d(theta)
	mean_theta_global = np.mean(theta, axis=(1,2), keepdims=True)
	mean_rho2_global  = np.mean(rho**2, axis=(1,2), keepdims=True)

	dH_drho   = D * lap_theta + theta - 3*rho**2*theta + kappa * 2 * rho * mean_theta_global
	dH_dtheta = D * lap_rho + rho - rho**3 + kappa * mean_rho2_global + aa * theta
	reaction_U = dH_drho - dH_dtheta
	# 2. 空間 FFT2
	U_Fourier[:] = np.fft.fft2(U, axes=(1, 2))
	V_Fourier[:] = np.fft.fft2(V, axes=(1, 2))
	reaction_U_Fourier[:] = np.fft.fft2(reaction_U, axes=(1, 2))

	# 3. 路徑時間邊界條件 (t = T)
	# U_new[Ncopy-1] = -V[Ncopy-1] + 2*rho2  => 在 Fourier 裡做
	RHS_U_Fourier = U_Fourier + dtau * reaction_U_Fourier
	RHS_U_Fourier[Ncopy-1, :, :] = -V_Fourier[Ncopy-1, :, :] + 2.0 * rho2k
	# 4. 對每個空間 mode 解 A @ U_new = RHS (path-time)
	N_space = Ly * Lx
	RHS_flat = RHS_U_Fourier.reshape(Ncopy, -1)
	U_Fourier_flat = U_Fourier.reshape(Ncopy, -1)
	U_new_flat = np.zeros_like(RHS_flat)
	r_vec = (dtau / ds) * lambdas
	for col in range(N_space):
		stab_term_scalar = dtau * gamma * (D * k4_flat[col] + 2.0 * k2_flat[col])
		stab_term_array  = stab_term_scalar * U_Fourier_flat[:, col]
		
		# [修正] 動態更新 A_banded (嚴格對齊 j 索引)
		
		# Row 2 (主對角線 i=j) -> 使用 r_vec[j]
		A_banded[col, 2, :] = 1. + 1.5 * r_vec + stab_term_scalar
		A_banded[col, 2, Ncopy-2] = 1. + r_vec[Ncopy-2] + stab_term_scalar
		A_banded[col, 2, Ncopy-1] = 1. + stab_term_scalar

		# Row 1 (一階上對角線 i=j-1) -> 必須使用 r_vec[j-1]，所以 Python 切片為 [:-1]
		A_banded[col, 1, 1:] = -2.0 * r_vec[:-1]
		A_banded[col, 1, Ncopy-1] = -r_vec[Ncopy-2]

		# Row 0 (二階上對角線 i=j-2) -> 必須使用 r_vec[j-2]，所以 Python 切片為 [:-2]
		A_banded[col, 0, 2:] = 0.5 * r_vec[:-2]

		RHS_col = RHS_flat[:, col] + stab_term_array
		U_new_flat[:, col] = solve_banded((0, 2), A_banded[col], RHS_col)
	U2_Fourier[:] = U_new_flat.reshape(Ncopy, Ly, Lx)

	U[:] = np.fft.ifft2(U2_Fourier, axes=(1, 2)).real
	rho = 0.5 * (U + V)
	theta = 0.5 * (U - V)
	rho[0] = rho1
	rho[Ncopy-1] = rho2
	theta[0] = 0.0          # <--- 絕對不能漏掉
	theta[Ncopy-1] = 0.0
	U = rho + theta
	V = rho - theta
	# ================= UPDATE V =================
	lap_rho   = apply_lap_2d(rho)
	lap_theta = apply_lap_2d(theta)
	mean_theta_global = np.mean(theta, axis=(1,2), keepdims=True)
	mean_rho2_global  = np.mean(rho**2, axis=(1,2), keepdims=True)

	dH_drho   = D * lap_theta + (1 - 3*rho**2)*theta + kappa * 2 * rho * mean_theta_global
	dH_dtheta = D * lap_rho + rho - rho**3 + kappa * mean_rho2_global + aa * theta
	reaction_V = dH_drho + dH_dtheta

	reaction_V_Fourier[:] = np.fft.fft2(reaction_V, axes=(1, 2))
	U_Fourier[:] = np.fft.fft2(U, axes=(1, 2))
	V_Fourier[:] = np.fft.fft2(V, axes=(1, 2))

	# 邊界條件 (t = 0): V_new[0] = -U[0] + 2*rho1
	RHS_V_Fourier = V_Fourier + dtau * reaction_V_Fourier
	RHS_V_Fourier[0, :, :] = -U_Fourier[0, :, :] + 2.0 * rho1k

	RHS_V_flat = RHS_V_Fourier.reshape(Ncopy, -1)
	V_Fourier_flat = V_Fourier.reshape(Ncopy, -1)
	V_new_flat = np.zeros_like(RHS_V_flat)
	for col in range(N_space):
		stab_term_scalar = dtau * gamma * (D * k4_flat[col] + 2.0 * k2_flat[col])
		stab_term_array  = stab_term_scalar * V_Fourier_flat[:, col]
		
		# [修正] 動態更新 B_banded (嚴格對齊 j 索引)
		
		# Row 0 (主對角線 i=j) -> 使用 r_vec[j]
		B_banded[col, 0, :] = 1. + 1.5 * r_vec + stab_term_scalar
		B_banded[col, 0, 1] = 1. + r_vec[1] + stab_term_scalar
		B_banded[col, 0, 0] = 1.0 + stab_term_scalar 

		# Row 1 (一階下對角線 i=j+1) -> 必須使用 r_vec[j+1]，所以 Python 切片為 [1:]
		B_banded[col, 1, :-1] = -2.0 * r_vec[1:]
		B_banded[col, 1, 0] = -r_vec[1]

		# Row 2 (二階下對角線 i=j+2) -> 必須使用 r_vec[j+2]，所以 Python 切片為 [2:]
		B_banded[col, 2, :-2] = 0.5 * r_vec[2:]

		RHS_col = RHS_V_flat[:, col] + stab_term_array
		V_new_flat[:, col] = solve_banded((2, 0), B_banded[col], RHS_col)
	V2_Fourier[:] = V_new_flat.reshape(Ncopy, Ly, Lx)

	V[:] = np.fft.ifft2(V2_Fourier, axes=(1, 2)).real
	rho = 0.5 * (U + V)
	theta = 0.5 * (U - V)
	rho[0] = rho1
	rho[Ncopy-1] = rho2
	theta[0] = 0.0          # <--- 絕對不能漏掉
	theta[Ncopy-1] = 0.0
	if i % 10 == 0:
		rho, theta = reparameterize_system(rho, theta)
    
	# [極度重要] 重分佈後，必須同步更新 U 和 V 給下一次迭代使用
	U = rho + theta
	V = rho - theta
	rho_1d = rho.reshape(Ncopy, -1)
	theta_1d = theta.reshape(Ncopy, -1)
	Lag_geom = Geometric_Action_Density(rho_1d.real, theta_1d.real, Ncopy)
    
	# 使用梯形法則精確積分 (橫軸為 0~1 的 s_points)
	s_points = np.linspace(0, 1.0, Ncopy)
	ds_geom = 1.0 / (Ncopy - 1)
	S_phys, L_geom, uphill_mask = compute_physical_action(rho, theta, D, kappa, ds_geom)
	actionS = S_phys * h * aa
	# if Lag[-1] > threshold or Lag[-1] < -threshold:
	# 	print(f"iter {i} Lagrangian is too large: {Lag[-1]}")
	# 	break
	if i > start_iter:
		# 指標一：數值穩定度 (路徑是否不再移動)
		max_diff_rho = np.max(np.abs(rho - old_rho_for_check))

		# 指標二：物理精確度 (Hamiltonian 是否平坦)
		# 1. 取得壓平的 rho 和 theta 來算整條路徑的 H
		rho_1d_check = rho.reshape(Ncopy, -1)
		theta_1d_check = theta.reshape(Ncopy, -1)
		H_path = Hamiltonian(h, rho_1d_check, theta_1d_check).real*h*aa

		# 2. 計算 H 的標準差 (跳過頭尾兩點，避免邊界差分帶來的微小數值震盪)
		H_std = np.std(H_path[2:-10])

		# 判定條件：路徑幾乎不動 ( < 1e-5 ) 且 H 非常平坦 ( 標準差 < 1e-3 )
		if max_diff_rho < 5e-5 and H_std < 0.0002:
			print(f"\n✨ 恭喜！系統在第 {i} 步已完美收斂抵達 Instanton 鞍點！")
			print(f"最大數值誤差 dRho: {max_diff_rho:.2e} | 物理守恆 H 標準差: {H_std:.2e}")
			break  # 安全跳出，完美存檔！
			
	# 紀錄當前狀態，留給下一步比對
	old_rho_for_check = rho.copy()
	##################################################
	'''------------ THIRD: PLOT DATA ---------'''
	##################################################	
	    
	if( i%plotStep == 0):
		print('Lag_geom[-1]:', Lag_geom[-1])
		if i > start_iter:
			print('max_diff_rho, H_std:', max_diff_rho, H_std)
		# Flatten space to (Ncopy, Ly*Lx) for Lagrangian/Hamiltonian and plots
		rho_1d = rho.reshape(Ncopy, -1)
		theta_1d = theta.reshape(Ncopy, -1)
		plt.gcf()
		if os.path.exists(f'local.txt'):
			fig = plt.figure(figsize=(10,10),layout='constrained')
		else:
			fig = plt.figure(figsize=(14,14),constrained_layout=True)
		# 4 subplots: 1. rho, 2. theta, 3. Lagrangian, 4. Hamiltonian 2 rows 2 columns
		ax0 = fig.add_subplot(222)
		ax1 = fig.add_subplot(221)
		ax2 = fig.add_subplot(223)
		ax3 = fig.add_subplot(224)
		Lag_geom = Geometric_Action_Density(rho_1d.real, theta_1d.real, Ncopy)
    
		# 使用梯形法則精確積分 (橫軸為 0~1 的 s_points)
		s_points = np.linspace(0, 1.0, Ncopy)
		s_edges = np.linspace(0, 1.0, Ncopy + 1)
		S_phys, L_geom, uphill_mask = compute_physical_action(rho, theta, D, kappa, ds_geom)
		actionS = S_phys * h * aa
		
		fig.suptitle(r'$N_t=$'+str(int(Ncopy))+r', $L=$'+str(Ly)+r', $\Delta \tau=$'+str("%.1e"%dtau)+r', $D=$'+str("%.1e"%D) +r', $T_\mathrm{max}=$'+str("%.1f"%Tmax)+"\n"+"$\kappa =$"+str(kappa)+", $S=$"+str("%.6f"%(actionS))+', Time '+ str(i*dtau), fontsize=20)
		### [NEW] PLOT: Symmetry Breaking Projection (Mean vs Std)
		mean_rho = np.mean(rho_1d.real, axis=1) 
		std_rho  = np.std(rho_1d.real, axis=1)  	

		im = ax0.scatter(std_rho, mean_rho, c=np.linspace(0,1,Ncopy), cmap='viridis', s=15, zorder=10) 
		ax0.plot(std_rho, mean_rho, color='darkblue', linewidth=2, label='Instanton Path')
		ax0.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Homogeneous (Symmetric)')

		ax0.set_xlabel(r'Inhomogeneity $\sigma_\rho$ (Std Dev)', fontsize=15)
		ax0.set_ylabel(r'Mean Density $\bar{\rho}$', fontsize=15)
		ax0.set_title(r'Symmetry Breaking Projection', fontsize=16)
        
		ax0.grid(True, linestyle=':', alpha=0.6)
		ax0.set_xlim(left=-0.05, right=max(std_rho.max()*1.2, 0.5)) 
		ax0.set_ylim(solRho[0]-0.2, solRho[2]+0.2)
		ax0.legend(loc='best', fontsize=12)
		## colorbar
		cbar = fig.colorbar(im, ax=ax0, shrink=0.8)
		cbar.set_label(r'$\sigma$', fontsize=20)

		### PLOT rho
		rho_map = rho_1d.real.T  # Shape: (Ly*Lx, Ncopy)
		n_space = rho_1d.shape[1]  # Ly*Lx
		x_edges = np.arange(n_space + 1) - 0.5
		im = ax1.pcolormesh(s_edges, x_edges, rho_map, 
                           cmap='coolwarm', 
                           shading='flat',
                           vmin=-1.5, vmax=1.5)
		cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
		cbar.set_label(r'$\rho$', fontsize=20)
		ax1.tick_params(axis='both', which='major', labelsize=15)
		ax1.set_yticks([0, n_space-1])
		ax1.set_xlabel(r'$\sigma$', fontsize=20)
		ax1.set_ylabel('Space index (0 to L-1)', fontsize=20)
    
		### PLOT theta
		theta_map = theta_1d.real.T  # Shape: (Ly*Lx, Ncopy)
		im2 = ax2.pcolormesh(s_edges, x_edges, theta_map, 
							cmap='bwr', 
							shading='flat',
							vmin=-0.5, vmax=0.5)
		cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
		cbar2.set_label(r'$\theta$', fontsize=20)
		ax2.tick_params(axis='both', which='major', labelsize=15)
		ax2.set_yticks([0, n_space-1])
		ax2.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
		ax2.set_xlabel(r'$\sigma$', fontsize=20)
		ax2.set_ylabel('Space index (0 to L-1)', fontsize=20)
		ax2.grid(True, linestyle='-')

		### PLOT Lagrangian and Hamiltonian
		ax3.set_aspect('auto')
		ax3.plot(s_points, Lag_geom*h*aa, label=r'$L(\rho,\dot\rho)$', color='black'  ) 
		ax3.plot(s_points, Hamiltonian(h,rho_1d, theta_1d).real*h*aa, label=r'$H(\rho,\theta)$', color='brown' ) 
		plt.legend(loc='best', fontsize=14)
		ax3.tick_params(axis='both', which='major', labelsize=15)
		ax3.set_xlabel(r'$\sigma$', fontsize=20)
		ax3.set_ylabel(r'$L$', fontsize=20)
		ax3.set_xlim(0,1)
		
		if(upward==True):
			plt.savefig('upward_L'+str(int(Ly))+'_N'+str(int(Ncopy))+'h'+str(float(h))+'_D'+str(D)+'_kappa'+str(kappa)+'_dtau'+str("%.1e"%dtau)+'_'+"%09d" % (i,)+'.png', format='png', bbox_inches='tight')
			end_time = time.time()
			print(f"Time taken: {end_time - start_time} seconds", "iteration", i)
		else:
			plt.savefig('downward_L'+str(int(Ly))+'_N'+str(int(Ncopy))+'h'+str(float(h))+'_D'+str(D)+'_kappa'+str(kappa)+'_dtau'+str("%.1e"%dtau)+'_'+"%09d" % (i,)+'.png', format='png', bbox_inches='tight')
			end_time = time.time()
			print(f"Time taken: {end_time - start_time} seconds", "iteration", i)
		plt.clf()
		plt.close()
		np.savez_compressed('checkpoints/checkpoint_local.npz', rho=rho, theta=theta, iteration=i, Lx=Lx, Ly=Ly, h=h, Ncopy=Ncopy, Tmax=Tmax, aa=aa, D=D, kappa=kappa, dtau=dtau, upward=upward, iterations=i, plotStep=plotStep)
# Animate for any L
animate_any_boxes(rho_1d.real, theta_1d.real, filename=f"boxes_L{int(Ly)}.mp4", dnu=dnu, skip=4, fps=25)












