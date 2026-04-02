from numpy import loadtxt
from matplotlib import cm
import math
import cmath
import time
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

from itertools import count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy import fft as sp_fft
from scipy.linalg import solve_triangular, solve_banded
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)



import time
start_time = time.time()

PI = math.pi

def normL2(h, array): #norm L2, spacing h for integration
	norm = math.sqrt( h * np.sum( np.square( array.real ) ) )
	return norm

def normL2Big(h, array): #norm L2, spacing h for integration
	norm = np.sqrt( h * np.sum( np.square( array.real ), axis=1 ) )
	return norm




##############################
""" DEFINE FUNCTIONS """
###############################



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

# Apply 2D Laplacian to each path slice: (Ncopy, Ly*Lx)
# def apply_lap_2d(field):
# 	# field (Ncopy, Ly, Lx) -> lap (Ncopy, Ly, Lx)
# 	N_current = field.shape[0]
# 	flat = field.reshape(N_current, -1)
# 	lap_flat = (Lap_Matrix.dot(flat.T)).T
# 	return lap_flat.reshape(N_current, Ly, Lx)

def apply_lap_2d(field):
    # 完全捨棄有限差分矩陣，改用絕對精準的傅立葉頻域計算 Laplacian
    # field 的形狀為 (N_current, Ly, Lx)
    field_k = sp_fft.fft2(field, axes=(1, 2))
    
    # 乘上我們早就準備好的 2D 頻域 k^2 矩陣
    # k2_2d 的形狀是 (Ly, Lx)，Numpy 會自動漂亮地廣播 (Broadcast) 到所有 Ncopy 身上
    lap_k = -k2_2d * field_k
    
    # 轉回實數空間
    return sp_fft.ifft2(lap_k, axes=(1, 2)).real

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

def Lagrangian(h, ds, rho, theta): #return an array of size Ncopy, axis=1 sums the colums
	rhoDot      = (np.roll(rho,-1,axis=0) - np.roll(rho,1,axis=0))/(2*ds)
	rhoDot[0,:] = (np.roll(rho,-1,axis=0) - rho)[0,:] /(ds)
	rhoDot[Ncopy-1,:] = (-np.roll(rho,1,axis=0) + rho)[Ncopy-1,:]/(ds)
	Ham = Hamiltonian(h, rho, theta)
	L =  np.sum( rhoDot * theta, axis=1 ) - Ham
	return L

def Lagrangian_Theo(h, rho, theta):
	# rho, theta: (Ncopy, Ly*Lx)
	rho_3d = rho.reshape(Ncopy, Ly, Lx)
	theta_3d = theta.reshape(Ncopy, Ly, Lx)
	lap_theta = apply_lap_2d(theta_3d)
	# L_theo = -0.5 * aa * theta * lap_theta 的空間積分
	L_density = -0.5 * aa * theta_3d * lap_theta
	return np.sum(L_density, axis=(1, 2))



################################
""" Start MAM """
##############################

Lx = 1
Ly = 80
h  = 0.1
Ncopy =400
Tmax = 100.
w = 40 # width of the interface
d = 20 # distance from the center of the initial state to the center of the final state

s = np.linspace(0,Tmax,Ncopy)
ds  = s[1]-s[0]
dnu = s[1]-s[0]
aa = 2.   #noise amplitude


upward = True # choose if path from -1 to +1 (upward), or the opposite
D     = 1

solRho = np.array([-1.0, 0.0, 1.0]) # rho-, rhos, rho+ 

dtau = 0.1 

iterations = 10000
plotStep   = 100

r = dtau/dnu
resume_file = "checkpoints/checkpoint.npz"
relaxed_file = 'checkpoints/relaxed.npz'
print('conditions: Ly,Lx,Ncopy =', Ly,Lx,Ncopy, 'h =', h, 'D =', D, 'dtau =', dtau, 'iterations =', iterations, 'Tmax =', Tmax)
# === IMEX 穩定化（對應 modified 的 k4、gamma）===
gamma = 2.0   # 與 modified 相同，壓制高頻
# 2D Fourier 波數（與 fft2(..., axes=(1,2)) 對應）
kx = 2 * PI * sp_fft.fftfreq(Lx, d=h)
ky = 2 * PI * sp_fft.fftfreq(Ly, d=h)
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
	A_banded[col, 2, Ncopy-1] = 1. #+ stab

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
	B_banded[col, 0, 0] = 1.0 #+ stab 

	# 2. First lower diagonal
	B_banded[col, 1, :-1] = -2*r
	B_banded[col, 1, 0] = -r

	# 3. Second lower diagonal
	B_banded[col, 2, :-2] = r/2.

# 1. 建立長方形的 2D Laplacian (大小是 (Lx*Ly) x (Lx*Ly))
# Lap_Matrix = build_rect_2d_laplacian(Lx, Ly, h)

# 2. 建立隱式算子
# I_total = sparse.identity(Lx * Ly, format='csr')
# Operator_Matrix = I_total - dtau * D * Lap_Matrix

# 3. 預先分解
# solve_2D = factorized(Operator_Matrix)


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



# if(upward==True):
#     rho1 = solRho[0]*np.ones((Ly,Lx), dtype=complex)
#     rho2 = solRho[0]*np.ones((Ly,Lx), dtype=complex)
#     # initial state: high density in the center
#     rho1[(Ly//2-w//2):(Ly//2+w//2),(Lx//2-w//2):(Lx//2+w//2)] = solRho[2] 
#     # rho1[0,0] = solRho[0]
#     rho1k = sp_fft.fft2(rho1)
    
#     # final state: high density strip in the middle
#     # rho2[Ly//2,Lx//2] = solRho[0]
#     rho2[:,(Lx//2-w//4):(Lx//2+w//4+1)] = solRho[2]
#     rho2k = sp_fft.fft2(rho2)
# else:
#     rho1 = solRho[0]*np.ones((Ly,Lx), dtype=complex)
#     rho2 = solRho[0]*np.ones((Ly,Lx), dtype=complex)
    
#     # high density strip in the middle
#     # rho1[Ly//2,Lx//2] = solRho[0]
#     rho1[:,(Lx//2-w//4):(Lx//2+w//4+1)] = solRho[2]
#     rho1k = sp_fft.fft2(rho1)
    
#     # high density in the center
#     rho2[(Ly//2-w//2):(Ly//2+w//2),(Lx//2-w//2):(Lx//2+w//2)] = solRho[2]
#     # rho2[0,0] = solRho[0]
#     rho2k = sp_fft.fft2(rho2)

if(upward==True):
    rho1 = solRho[0]*np.ones((Ly,Lx), dtype=complex)
    rho2 = solRho[0]*np.ones((Ly,Lx), dtype=complex)
    # 起點：左高右低
    rho1[(Ly//2):(Ly//2+w),Lx//2] = solRho[2] 
    rho1[0,0] = solRho[0]
    rho1k = np.fft.fft2(rho1)
    
    # 終點：左低右高
    # rho2[Ly//2,Lx//2] = solRho[0]
    rho2[(Ly//2-d):(Ly//2-d+w),Lx//2] = solRho[2]
    rho2k = np.fft.fft2(rho2)
else:
    rho1 = solRho[0]*np.ones((Ly,Lx), dtype=complex)
    rho2 = solRho[0]*np.ones((Ly,Lx), dtype=complex)
    
    # 起點：左低右高
    # rho1[Ly//2,Lx//2] = solRho[0]
    rho1[(Ly//2-d):(Ly//2-d+w),Lx//2] = solRho[2]
    rho1k = np.fft.fft2(rho1)
    
    # 終點：左高右低
    rho2[(Ly//2):(Ly//2+w),Lx//2] = solRho[2]
    rho2[0,0] = solRho[0]
    rho2k = np.fft.fft2(rho2)
# check if mass is conserved
print("mass of rho1", np.sum(rho1))
print("mass of rho2", np.sum(rho2))
# ==========================================
# [新增] 預先放鬆 (Pre-relaxation) 區塊
# 目的：讓銳利的初始猜測自然擴散成真實的物理穩態
# ==========================================
print("Relaxing Initial and Final states to exact Model B equilibria...")
dt_relax = 0.001  # 對於你設定的 h=0.3, D=0.002，這個步長非常安全
relax_steps = 20000

if os.path.exists(relaxed_file):
	data = np.load(relaxed_file)
	rho1 = data['rho1']
	rho2 = data['rho2']
else:	
	def relax_model_b_smart(rho_state, tol=1e-7, max_steps=500000):
		rho_2d = rho_state.copy().real
		target_mass = np.mean(rho_2d)
		implicit_denom = 1.0 + dt_relax * D * k4_2d
		
		print("開始尋找絕對物理穩態...")
		for step in range(1, max_steps + 1):
			rho_old = rho_2d.copy()
			
			# 1. 計算化學勢的顯式部分
			mu_explicit = -rho_2d + rho_2d**3
			mu_k = sp_fft.fft2(mu_explicit)
			
			# 2. 半隱式時間推進
			lap_mu_k = -k2_2d * mu_k
			rho_k = sp_fft.fft2(rho_2d)
			rho_k_new = (rho_k + dt_relax * lap_mu_k) / implicit_denom
			
			rho_2d = sp_fft.ifft2(rho_k_new).real
			rho_2d = rho_2d - np.mean(rho_2d) + target_mass # 質量守恆
			
			# ====================================================
			# 每 1000 步進行一次「嚴格物理穩態檢測」
			# ====================================================
			if step % 1000 == 0:
				# 條件一：計算密度的最大變化率 (dRho/dt)
				max_diff = np.max(np.abs(rho_2d - rho_old)) / dt_relax
				
				# 條件二：計算真實化學勢 mu = -D*Lap(rho) - rho + rho^3 的「空間變異數 (Variance)」
				lap_rho = sp_fft.ifft2(-k2_2d * sp_fft.fft2(rho_2d)).real
				mu_full = -D * lap_rho - rho_2d + rho_2d**3
				mu_variance = np.var(mu_full) # 如果 mu 是常數，變異數必須趨近於 0
				
				print(f"   [Step {step:5d}] dRho/dt = {max_diff:.2e} | mu 變異數 = {mu_variance:.2e}")
				
				# 當兩個指標都小於容忍值(tol)時，確認抵達絕對穩態！
				if max_diff < tol and mu_variance < tol:
					print(f"✅ 成功！系統在第 {step} 步抵達絕對物理穩態！")
					break
					
		if step == max_steps:
			print("⚠️ 警告：達到最大步數但尚未完全收斂，請考慮增加 max_steps 或檢查初始物理設定。")
			
		return rho_2d + 0j

	# 取代原本的呼叫方式：
	print("--- 放鬆起點 rho1 ---")
	rho1 = relax_model_b_smart(rho1,max_steps=1000000)
	print("--- 放鬆終點 rho2 ---")
	rho2 = relax_model_b_smart(rho2,max_steps=1000000)
	np.savez('checkpoints/relaxed.npz', rho1=rho1, rho2=rho2)

	# ==========================================
	# [新增] 視覺化：畫出平滑後的初始與終點狀態
	# 目的：檢查 Relaxation 是否成功產生物理介面 (tanh 曲線)
	# ==========================================
	fig_relax, (ax_r1, ax_r2, ax_r3) = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')

	# 1. 畫 rho1 的 2D 熱圖
	im_r1 = ax_r1.imshow(rho1.real, cmap='bwr', vmin=-1.5, vmax=1.5, origin='lower', aspect='auto')
	ax_r1.set_title("Relaxed Initial State (rho1)")
	ax_r1.set_xlabel("x index")
	ax_r1.set_ylabel("y index")
	fig_relax.colorbar(im_r1, ax=ax_r1, shrink=0.8, label=r"$\rho$")

	# 2. 畫 rho2 的 2D 熱圖
	im_r2 = ax_r2.imshow(rho2.real, cmap='bwr', vmin=-1.5, vmax=1.5, origin='lower', aspect='auto')
	ax_r2.set_title("Relaxed Final State (rho2)")
	ax_r2.set_xlabel("x index")
	ax_r2.set_ylabel("y index")
	fig_relax.colorbar(im_r2, ax=ax_r2, shrink=0.8, label=r"$\rho$")

	# 3. 畫 1D 截面圖 (切在中間的 X 軸上，觀察 y 方向的介面變化)
	mid_x = Lx // 2
	ax_r3.plot(rho1.real[:, mid_x], 'o-', label='rho1 (Start)', color='blue', markersize=4)
	ax_r3.plot(rho2.real[:, mid_x], 's-', label='rho2 (End)', color='red', markersize=4)
	ax_r3.axhline(0, color='black', linestyle='--', alpha=0.5)
	ax_r3.axhline(1, color='gray', linestyle=':', alpha=0.5)
	ax_r3.axhline(-1, color='gray', linestyle=':', alpha=0.5)
	ax_r3.set_title(f"1D Cross-section along Y (at x={mid_x})")
	ax_r3.set_xlabel("y index")
	ax_r3.set_ylabel(r"Density $\rho$")
	ax_r3.legend()

	# 儲存圖片
	relax_plot_name = f'relaxed_states_Ly{Ly}_Lx{Lx}.png'
	plt.savefig(relax_plot_name, dpi=150, facecolor='white')
	plt.close(fig_relax)
	print(f"[*] Saved relaxed states plot to: {relax_plot_name}")
	# ==========================================

rho1k = sp_fft.fft2(rho1)
rho2k = sp_fft.fft2(rho2)
reaction_U_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)
reaction_V_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)


######  INITIAL CONDITIONS 


rho[0,:,:]       = rho1
rho[Ncopy-1,:,:] = rho2
# deterministically set the initial condition
amp = 0.4
y_coords = np.arange(Ly)
x_coords = np.arange(Lx)
Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

# randomly set the initial condition
noise_amp = 0.5  # noise amplitude
np.random.seed(42 if Lx == 1 else None)

for j in range(1,Ncopy-1):
	tt = float(j)/Ncopy
	linear = rho1*(1-tt) + tt*rho2
	# deterministic initial condition
	bump = amp * np.square(np.sin(PI * Y / Ly)) * np.power(np.sin(PI * tt), 2)
	# random initial condition
	# bump = noise_amp*np.random.normal(0, 1, size=(Ly,Lx))*np.sin(PI*tt)
	bump = bump - np.mean(bump)
	rho[j,:,:] = linear + bump

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
	dtau = data['dtau']
	upward = data['upward']
	end_iterations = data['iterations']
	plotStep = data['plotStep']
	start_iter = end_iterations + 1
else:
	start_iter = 0
U = rho + theta
V = rho - theta
###### Evolve Loop (GDA Algorithm 1: path-time upwind + full reaction) ######
start_time = time.time()
print("start_iter", start_iter)
print("iterations", iterations)
for i in range(start_iter, iterations+1):

	# ================= UPDATE U =================
	# 1. 完整 reaction（含空間擴散 D*Lap）
	lap_rho   = apply_lap_2d(rho)
	lap_theta = apply_lap_2d(theta)
	# Model B: dH_drho = -D*lap(lap_theta) - (1-3*rho^2)*lap_theta
	dH_drho   = -D * apply_lap_2d(lap_theta) - (1 - 3*rho**2) * lap_theta
	# Model B: dH_dtheta = lap(-D*lap_rho - rho + rho^3) - aa*lap_theta
	dH_dtheta = apply_lap_2d(-D * lap_rho - rho + rho**3) - aa * lap_theta
	reaction_U = dH_drho - dH_dtheta

	# 2. 空間 FFT2
	U_Fourier[:] = sp_fft.fft2(U, axes=(1, 2))
	V_Fourier[:] = sp_fft.fft2(V, axes=(1, 2))
	reaction_U_Fourier[:] = sp_fft.fft2(reaction_U, axes=(1, 2))

	# 3. 路徑時間邊界條件 (t = T)
	# U_new[Ncopy-1] = -V[Ncopy-1] + 2*rho2  => 在 Fourier 裡做
	# RHS_U_Fourier = U_Fourier + dtau * reaction_U_Fourier
	# RHS_U_Fourier[Ncopy-1, :, :] = -V_Fourier[Ncopy-1, :, :] + 2.0 * rho2k
	# # 4. 對每個空間 mode 解 A @ U_new = RHS (path-time)
	# N_space = Ly * Lx
	# RHS_flat = RHS_U_Fourier.reshape(Ncopy, -1)
	# U_new_flat = solve_triangular(A_solve_upper_adapted, RHS_flat, check_finite=False)
	# U2_Fourier[:] = U_new_flat.reshape(Ncopy, Ly, Lx)
	RHS_U_Fourier = U_Fourier + dtau * reaction_U_Fourier
	RHS_U_Fourier[Ncopy-1, :, :] = -V_Fourier[Ncopy-1, :, :] + 2.0 * rho2k
	N_space = Ly * Lx
	RHS_flat = RHS_U_Fourier.reshape(Ncopy, -1)
	U_Fourier_flat = U_Fourier.reshape(Ncopy, -1)
	U_new_flat = np.zeros_like(RHS_flat)
	for col in range(N_space):
		stab_term = gamma * (D * k4_flat[col] + 2.0 * k2_flat[col]) * U_Fourier_flat[:, col]
		RHS_col = RHS_flat[:, col] + dtau * stab_term
		RHS_col[-1] = RHS_flat[-1, col]
		U_new_flat[:, col] = solve_banded((0, 2), A_banded[col], RHS_col)
	U2_Fourier[:] = U_new_flat.reshape(Ncopy, Ly, Lx)

	U[:] = sp_fft.ifft2(U2_Fourier, axes=(1, 2)).real
	rho = 0.5 * (U + V)
	theta = 0.5 * (U - V)
	# print("rho ", np.mean(rho.real))
	# print("theta: ", np.mean(theta.real))
	
	# target_mass = np.mean(rho1)
	# rho = rho - np.mean(rho, axis=(1,2), keepdims=True) + target_mass
	# theta = theta - np.mean(theta, axis=(1,2), keepdims=True)
	rho[0] = rho1
	rho[Ncopy-1] = rho2
	theta[0] = 0.0 + 0j       # 穩態起點絕對沒有雜訊
	theta[Ncopy-1] = 0.0 + 0j # 穩態終點絕對沒有雜訊
	U = rho + theta
	V = rho - theta

	# ================= UPDATE V =================
	lap_rho   = apply_lap_2d(rho)
	lap_theta = apply_lap_2d(theta)
	dH_drho   = -D * apply_lap_2d(lap_theta) - (1 - 3*rho**2) * lap_theta
	dH_dtheta = apply_lap_2d(-D * lap_rho - rho + rho**3) - aa * lap_theta
	reaction_V = dH_drho + dH_dtheta
	reaction_V_Fourier[:] = sp_fft.fft2(reaction_V, axes=(1, 2))
	U_Fourier[:] = sp_fft.fft2(U, axes=(1, 2))
	V_Fourier[:] = sp_fft.fft2(V, axes=(1, 2))

	# 邊界條件 (t = 0): V_new[0] = -U[0] + 2*rho1
	# RHS_V_Fourier = V_Fourier + dtau * reaction_V_Fourier
	# RHS_V_Fourier[0, :, :] = -U_Fourier[0, :, :] + 2.0 * rho1k

	# RHS_V_flat = RHS_V_Fourier.reshape(Ncopy, -1)
	# V_new_flat = solve_triangular(B_solve_lower_adapted, RHS_V_flat, lower=True, check_finite=False)
	# V2_Fourier[:] = V_new_flat.reshape(Ncopy, Ly, Lx)
	RHS_V_Fourier = V_Fourier + dtau * reaction_V_Fourier
	RHS_V_Fourier[0, :, :] = -U_Fourier[0, :, :] + 2.0 * rho1k

	RHS_V_flat = RHS_V_Fourier.reshape(Ncopy, -1)
	V_Fourier_flat = V_Fourier.reshape(Ncopy, -1)
	V_new_flat = np.zeros_like(RHS_V_flat)
	for col in range(N_space):
		stab_term = gamma * (D * k4_flat[col] + 2.0 * k2_flat[col]) * V_Fourier_flat[:, col]
		RHS_col = RHS_V_flat[:, col] + dtau * stab_term
		RHS_col[0] = RHS_V_flat[0, col]
		V_new_flat[:, col] = solve_banded((2, 0), B_banded[col], RHS_col)
	V2_Fourier[:] = V_new_flat.reshape(Ncopy, Ly, Lx)
	V[:] = sp_fft.ifft2(V2_Fourier, axes=(1, 2)).real
	rho = 0.5 * (U + V)
	theta = 0.5 * (U - V)
	# print("rho ", np.mean(rho.real))
	# print("theta: ", np.mean(theta.real))
	# target_mass = np.mean(rho1)
	# rho = rho - np.mean(rho, axis=(1,2), keepdims=True) + target_mass
	# theta = theta - np.mean(theta, axis=(1,2), keepdims=True)
	# print("-----------------"+str(i)+"----------------")
	rho[0] = rho1
	rho[Ncopy-1] = rho2
	theta[0] = 0.0 + 0j       # 穩態起點絕對沒有雜訊
	theta[Ncopy-1] = 0.0 + 0j # 穩態終點絕對沒有雜訊
	U = rho + theta
	V = rho - theta
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
		if max_diff_rho < 5e-5 and H_std < Hamiltonian(h,rho_1d_check, theta_1d_check).real[2:-10].max()*h*aa*1e-2:
			print(f"\n✨ 恭喜！系統在第 {i} 步已完美收斂抵達 Instanton 鞍點！")
			print(f"最大數值誤差 dRho: {max_diff_rho:.2e} | 物理守恆 H 標準差: {H_std:.2e}")
			break  # 安全跳出，完美存檔！
			
	# 紀錄當前狀態，留給下一步比對
	old_rho_for_check = rho.copy()
	##################################################
	'''------------ THIRD: PLOT DATA ---------'''
	##################################################	
	    
	if( i%plotStep == 0):
		if i > start_iter:
			print('max_diff_rho, H_std, H_max:', max_diff_rho, H_std, Hamiltonian(h,rho_1d_check, theta_1d_check).real[2:-10].max()*h*aa*1e-2)
		# Flatten space to (Ncopy, Ly*Lx) for Lagrangian/Hamiltonian and plots
		rho_1d = rho.reshape(Ncopy, -1)
		theta_1d = theta.reshape(Ncopy, -1)
		plt.gcf()
		fig = plt.figure(figsize=(10,10),layout='constrained')
		# 4 subplots: 1. rho, 2. theta, 3. Lagrangian, 4. Hamiltonian 2 rows 2 columns
		ax0 = fig.add_subplot(222)
		ax1 = fig.add_subplot(221)
		ax2 = fig.add_subplot(223)
		ax3 = fig.add_subplot(224)
		Lag = Lagrangian(h, dnu, rho_1d, theta_1d).real
		actionS = dnu* np.sum(Lag)*h*aa
		
		fig.suptitle(r'$N_\mathrm{copy}=$'+str(int(Ncopy))+r', $L=$'+str(Ly)+r', $\Delta \tau=$'+str("%.1e"%dtau)+r', $D=$'+str("%.1e"%D) +r', $T_\mathrm{max}=$'+str("%.1f"%Tmax)+r', $S=$'+str("%.6f"%(actionS))+', Time '+ str(i*dtau), fontsize=20)
		
		### VECTOR FIELD
		# Saction = dnu* np.sum(Lagrangian(h, dnu, rho, theta).real)
		# ax0.scatter(rho[:,0].real, rho[:,1].real, color='darkblue', s=2.2, zorder=15)
		# x = np.arange(solRho[0], solRho[2], 0.02)
		# y = np.arange(solRho[0], solRho[2], 0.02)

		# X, Y = np.meshgrid(x, y)
		# u = D*(2*Y-2*X)/h**2  + (X - X*X*X) + kappa*(X*X+Y*Y)/2.
		# v = D*(2*X-2*Y)/h**2  + (Y - Y*Y*Y) + kappa*(X*X+Y*Y)/2.

		# ax0.streamplot(x, y, u, v, density=1, color='grey')
		# ax0.set_xlabel(r'$\rho_1$', fontsize=20)
		# ax0.set_ylabel(r'$\rho_2$', fontsize=20)
		# ax0.set_xlim(solRho[0],solRho[2])
		# ax0.set_ylim(solRho[0],solRho[2])
		# ax0.tick_params(axis='both', which='major', labelsize=15)
		# ax0.set_aspect('equal')
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
		cbar.set_label(r'Time', fontsize=20)

		### PLOT rho
		mid_x = Lx // 2
		rho_slice = rho[:, :, mid_x].real.T
		t_edges = np.linspace(0, Tmax, Ncopy + 1)
		y_edges = np.arange(Ly + 1) - 0.5 # 現在 Y 軸是 X 空間座標
		im = ax1.pcolormesh(t_edges, y_edges, rho_slice, 
                           cmap='bwr', 
                           shading='flat',
                           vmin=-1.5, vmax=1.5)
		cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
		cbar.set_label(r'$\rho(x_{mid}, y, t)$', fontsize=20)
		ax1.tick_params(axis='both', which='major', labelsize=15)
		ax1.set_yticks([0, Ly-1])
		ax1.set_xlabel('Time', fontsize=20)
		ax1.set_ylabel(f'Y Coordinate (at x={mid_x})', fontsize=15)
		Lag = Lagrangian(h, dnu, rho_1d, theta_1d).real
		Lag_theo = Lagrangian_Theo(h, rho_1d, theta_1d).real
		actionS = dnu* np.sum(Lag)*h*aa

		### PLOT theta
		theta_slice = theta[:, :, mid_x].real.T
		im2 = ax2.pcolormesh(t_edges, y_edges, theta_slice, 
							cmap='bwr', 
							shading='flat',
							vmin=min(theta_slice.min(), -0.5), vmax=max(theta_slice.max(), 0.5))
		cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
		cbar2.set_label(r'$\theta(x_{mid}, y, t)$', fontsize=20)
		ax2.tick_params(axis='both', which='major', labelsize=15)
		ax2.set_yticks([0, Ly-1])
		ax2.set_xlabel('Time', fontsize=20)
		ax2.set_ylabel(f'Y Coordinate (at x={mid_x})', fontsize=15)

		### PLOT Lagrangian and Hamiltonian
		ax3.set_aspect('auto')
		ax3.plot(np.linspace(0,1,Ncopy), Lag*h*aa, label=r'$L(\rho,\dot\rho)$', color='black'  ) 
		ax3.plot(np.linspace(0,1,Ncopy), Hamiltonian(h,rho_1d, theta_1d).real*h*aa, label=r'$H(\rho,\theta)$', color='brown' , linestyle='-.') 
		plt.legend(loc='best', fontsize=14)
		ax3.tick_params(axis='both', which='major', labelsize=15)
		ax3.set_xlabel(r'$t/T_\mathrm{Max}$', fontsize=20)
		ax3.set_ylabel(r'$L$', fontsize=20)
		ax3.set_xlim(0,1)
		ax3.set_ylim(-0.01*max(Lag[2:-10].max()*h*aa, Hamiltonian(h,rho_1d, theta_1d).real[2:-10].max()*h*aa), max(Lag[2:-10].max()*h*aa, Hamiltonian(h,rho_1d, theta_1d).real[2:-10].max()*h*aa)*1.2)
		
		if(upward==True):
			plt.savefig('upward_Lx'+str(int(Lx))+'_Ly'+str(int(Ly))+'_N'+str(int(Ncopy))+'h'+str(float(h))+'_D'+str(D)+'_dtau'+str("%.1e"%dtau)+'_'+"%09d" % (i,)+'.png', format='png', bbox_inches='tight')
			print(f"Time taken: {time.time() - start_time} seconds", "iteration", i)
		else:
			plt.savefig('downward_Lx'+str(int(Lx))+'_Ly'+str(int(Ly))+'_N'+str(int(Ncopy))+'h'+str(float(h))+'_D'+str(D)+'_dtau'+str("%.1e"%dtau)+'_'+"%09d" % (i,)+'.png', format='png', bbox_inches='tight')
			print(f"Time taken: {time.time() - start_time} seconds", "iteration", i)
		plt.clf()
		plt.close()
		np.savez_compressed('checkpoints/checkpoint.npz', rho=rho, theta=theta, iteration=i, Lx=Lx, Ly=Ly, h=h, Ncopy=Ncopy, Tmax=Tmax, aa=aa, D=D, dtau=dtau, upward=upward, iterations=i, plotStep=plotStep)
def animate_2d_heatmap(rho, theta, filename="2d_evolution.mp4", dnu=None, skip=1, fps=30):
    """
    2D Heatmap Animation for both rho and theta
    rho, theta shape: (Ncopy, Ly, Lx)
    """
    rho = np.asarray(rho).real
    theta = np.asarray(theta).real
    Nt, Ly, Lx = rho.shape
    
    if dnu is None:
        dnu = 1.0 / max(Nt - 1, 1)

    # Downsample frames
    indices = np.arange(0, Nt, skip)
    if len(indices) > 300: # 限制最大幀數以免檔案太大
        indices = np.linspace(0, Nt - 1, 200, dtype=int)
    
    rho_sub = rho[indices]
    theta_sub = theta[indices]
    n_frames = len(indices)

    # Setup Figure (改成 1x2 的子圖，寬度加倍)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
    
    # ---------------------------------------------------------
    # 1. 初始化 Rho 的圖 (左邊)
    # ---------------------------------------------------------
    im1 = ax1.imshow(rho_sub[0], cmap='Reds', vmin=-1.5, vmax=1.5, 
                     origin='lower', interpolation='nearest', animated=True)
    cb1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cb1.set_label(r"Density $\rho$")
    ax1.set_title(r"Density $\rho$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # ---------------------------------------------------------
    # 2. 初始化 Theta 的圖 (右邊)
    # ---------------------------------------------------------
    # 自動抓取 theta 的最大絕對值，讓正負顏色對稱 (白色為 0)
    theta_max = max(np.abs(theta).max(), 0.01) 
    im2 = ax2.imshow(theta_sub[0], cmap='bwr', vmin=-theta_max, vmax=theta_max, 
                     origin='lower', interpolation='nearest', animated=True)
    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cb2.set_label(r"Noise Force $\theta$")
    ax2.set_title(r"Noise Force $\theta$")
    ax2.set_xlabel("x")
    
    # 共同的大標題
    fig.suptitle(f"Time: 0.00 | Path Evolution", fontsize=16)

    def update(frame_idx):
        # 同時更新兩張圖的資料
        im1.set_data(rho_sub[frame_idx])
        im2.set_data(theta_sub[frame_idx])
        
        # 更新大標題的時間
        t = indices[frame_idx] * dnu
        fig.suptitle(f"Time: {t:.2f} | Path Evolution")
        
        return im1, im2

    # 注意：這裡把 blit 改成 False，因為我們要更新 fig.suptitle
    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000.0/fps)
    
    # Save
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "mp4":
        try:
            anim.save(filename, writer="ffmpeg", fps=fps, dpi=100)
        except:
            anim.save(filename.replace(".mp4", ".gif"), writer="pillow", fps=fps)
    else:
        anim.save(filename, writer="pillow", fps=fps)
        
    plt.close(fig)
    print(f"Saved 2D animation: {filename}")


# 傳入原始的 3D rho (Ncopy, Ly, Lx)
animate_2d_heatmap(rho.real, theta.real, filename=f"2d_sim_Lx{int(Lx)}_Ly{int(Ly)}.mp4", dnu=dnu, skip=4, fps=10)












