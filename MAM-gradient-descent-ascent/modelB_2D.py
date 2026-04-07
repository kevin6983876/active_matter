import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import sparse
from scipy import fft as sp_fft
from scipy.linalg import solve_banded
import time
start_time = time.time()
PI = math.pi
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


################################
""" Start MAM """
##############################

Lx = 100
Ly = 100
h  = 0.2
Ncopy = 400
Tmax = 400.

s = np.linspace(0,Tmax,Ncopy)
ds  = s[1]-s[0]
dnu = s[1]-s[0]
aa = 2.   #noise amplitude


upward = True # True if droplet to stripe, False if stripe to droplet
previous_data = True
path_reverse = False
boundary_reverse = True
threshold = 1000

D     = 1
# boundary conditions: droplet to strip
W = 32       # 設定 Stripe 的寬度 (至少 3 格以上以維持物理厚度)
target_area = W * Ly     # 高密度 (solRho[2]) 應該佔據的精確格點總數
print("Width", W*h)
print("target_area", target_area*h**2)

solRho = np.array([-1.0, 0.0, 1.0]) # rho-, rhos, rho+ 

dtau = 0.1 

iterations = 20000
plotStep   = 200

r = dtau/dnu
resume_file = "checkpoints/modelB/checkpoint_local.npz"
relaxed_file = 'checkpoints/modelB/relaxed_2d_6.npz'
print('conditions: Ly,Lx,Ncopy =', Ly,Lx,Ncopy, 'h =', h, 'D =', D, 'dtau =', dtau, 'iterations =', iterations, 'Tmax =', Tmax)
# === IMEX 穩定化（對應 modified 的 k4、gamma）===
gamma = 1.0   # 與 modified 相同，壓制高頻
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
rho_stripe = solRho[0] * np.ones((Ly, Lx), dtype=float)
start_col = (Lx - W) // 2
rho_stripe[:, start_col : start_col + W] = solRho[2]
rho_slab = solRho[0] * np.ones((Ly, Lx), dtype=float)
rho_slab[:, 0:2] = solRho[2]
rho_slab[:, Lx-2:Lx] = solRho[2]
rho_droplet = solRho[0] * np.ones((Ly, Lx), dtype=float)
cy, cx = (Ly - 1) / 2.0, (Lx - 1) / 2.0 # 幾何中心
Y, X = np.meshgrid(np.arange(Ly), np.arange(Lx), indexing='ij')
dist_sq = (Y - cy)**2 + (X - cx)**2
flat_indices = np.argsort(dist_sq.flatten())
rho_droplet_flat = rho_droplet.flatten()
rho_droplet_flat[flat_indices[:target_area]] = solRho[2]
rho_droplet = rho_droplet_flat.reshape((Ly, Lx))

if(upward==True):
	rho1 = rho_droplet
	rho1k = sp_fft.fft2(rho1)
	rho2 = rho_stripe
	rho2k = sp_fft.fft2(rho2)
else:
	rho1 = rho_stripe
	rho1k = sp_fft.fft2(rho1)
	rho2 = rho_droplet
	rho2k = sp_fft.fft2(rho2)

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
	if boundary_reverse == True:
		rho1 = data['rho2']
		rho1k = sp_fft.fft2(rho1)
		rho2 = data['rho1']
		rho2k = sp_fft.fft2(rho2)
	else:
		rho1 = data['rho1']
		rho1k = sp_fft.fft2(rho1)
		rho2 = data['rho2']
		rho2k = sp_fft.fft2(rho2)
else:	
	def relax_model_b_smart(rho_state, tol=1e-7, max_steps=500000, noise_amp=0.01):
		rho_2d = rho_state.copy().real
		target_mass = np.mean(rho_2d)
		implicit_denom = 1.0 + dt_relax * D * k4_2d

		# ====================================================
		# [修改 2] 注入初始微小擾動 (打破完美對稱性)
		# ====================================================
		if noise_amp > 0.0:
			print(f"注入初始雜訊 (振幅: {noise_amp}) 以誘發潛在的不穩定性...")
			# 產生常態分佈的隨機雜訊
			noise = np.random.normal(0, noise_amp, size=rho_2d.shape)
			# 【極度重要】Model B 雜訊必須是「零均值」，否則會破壞系統總質量！
			noise = noise - np.mean(noise) 
			rho_2d += noise

			# 確保加了雜訊後，rho 不會因為極端值超出物理合理範圍太多 (選用保險)
			rho_2d = np.clip(rho_2d, -1.2, 1.2) 
			# 再次強制對齊質量
			rho_2d = rho_2d - np.mean(rho_2d) + target_mass 
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
	target_mass = np.mean(rho2)
	rho1 = rho1 - np.mean(rho1) + target_mass
	# mass_diff = np.sum(rho2) - np.sum(rho1)
	# rho1[0, 0] += mass_diff
	np.savez('checkpoints/modelB/relaxed.npz', rho1=rho1, rho2=rho2)
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
	rho1k = sp_fft.fft2(rho1)
	rho2k = sp_fft.fft2(rho2)
rho1[0, 0] += (np.sum(rho2) - np.sum(rho1))
rho1k = sp_fft.fft2(rho1)
rho2k = sp_fft.fft2(rho2)
print("mass of rho1", np.sum(rho1))
print("mass of rho2", np.sum(rho2))


extract_ratio = 1.0

######  Initial guess

if previous_data == True:
	if os.path.exists('checkpoints/modelB/checkpoint_b_2d_5_6.npz'):
		data = np.load('checkpoints/modelB/checkpoint_b_2d_5_6.npz')
		rho_old = data['rho']
		theta_old = data['theta']
		T_old = data['Tmax']
		Ncopy_old = data['Ncopy']
		Lx_old = data['Lx']
		Ly_old = data['Ly']

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

		# pad_start = 20  # 在 rho1 (Stripe) 停留的步數
		# pad_end   = 20  # 在 rho2 (Droplet) 停留的步數
		# N_active  = Ncopy - pad_start - pad_end  # 中間實際用來演化的步數
		# zoom_factors = (N_active / Ncopy_old, Ly / Ly_old, Lx / Lx_old)
		# rho_active = ndimage.zoom(rho_old.real, zoom_factors, order=1).astype(complex)
		# theta_active = ndimage.zoom(theta_old.real, zoom_factors, order=1).astype(complex)
		zoom_factors = (Ncopy / Ncopy_old, Ly / Ly_old, Lx / Lx_old)
		rho = ndimage.zoom(rho_old.real, zoom_factors, order=1).astype(complex)
		theta = ndimage.zoom(theta_old.real, zoom_factors, order=1).astype(complex)
		if path_reverse == True:
			# reverse rho in time
			rho = rho[::-1,:,:]
			theta = theta[::-1,:,:]
		# if path_reverse == True:
		# 	rho_active = rho_active[::-1,:,:]
		# 	theta_active = theta_active[::-1,:,:]
		# rho = np.zeros((Ncopy, Ly, Lx), dtype=complex)
		# theta = np.zeros((Ncopy, Ly, Lx), dtype=complex)
		# rho[0:pad_start, :, :] = rho1
		# theta[0:pad_start, :, :] = 0.0 + 0j
		# rho[pad_start:Ncopy-pad_end, :, :] = rho_active
		# theta[pad_start:Ncopy-pad_end, :, :] = theta_active
		# rho[Ncopy-pad_end:, :, :] = rho2
		# theta[Ncopy-pad_end:, :, :] = 0.0 + 0j
		# rho[0,:,:]       = rho1
		# rho[Ncopy-1,:,:] = rho2
		# theta[0,:,:]     = 0.0 + 0j
		# theta[Ncopy-1,:,:]= 0.0 + 0j
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
	noise_amp = 0.5  # noise amplitude
	np.random.seed(42 if Lx == 1 else None)

	if upward == True:
		for j in range(1,Ncopy-1):
			tt = float(j)/Ncopy
			linear = rho1*(1-tt) + tt*rho2
			# deterministic initial condition
			# bump = amp * np.square(np.sin(PI * Y / Ly)) * np.power(np.sin(PI * tt), 2)
			# random initial condition
			# bump = noise_amp*np.random.normal(0, 1, size=(Ly,Lx))*np.sin(PI*tt)
			# bump = bump - np.mean(bump)
			rho[j,:,:] = linear + 0j
	else: # downward
		for j in range(1,Ncopy-1):
			tt = float(j)/Ncopy
			linear = rho1*(1-tt) + tt*rho2
			# deterministic initial condition
			# bump = amp * np.square(np.sin(PI * Y / Ly)) * np.power(np.sin(PI * tt), 2)
			# random initial condition
			# bump = noise_amp*np.random.normal(0, 1, size=(Ly,Lx))*np.sin(PI*tt)
			# bump = bump - np.mean(bump)
			rho[j,:,:] = linear + 0j
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
	dtau = data['dtau']
	upward = data['upward']
	end_iterations = data['iterations']
	plotStep = data['plotStep']
	start_iter = end_iterations + 1
else:
	start_iter = 0
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
		H_path = Hamiltonian(h, rho_1d_check, theta_1d_check).real*h**2*aa

		# 2. 計算 H 的標準差 (跳過頭尾兩點，避免邊界差分帶來的微小數值震盪)
		H_std = np.std(H_path[2:-10])

		# # 判定條件：路徑幾乎不動 ( < 1e-5 ) 且 H 非常平坦 ( 標準差 < 1e-3 )
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
		if i > start_iter:
			print('max_diff_rho, H_std', max_diff_rho, H_std)
		# Flatten space to (Ncopy, Ly*Lx) for Lagrangian/Hamiltonian and plots
		rho_1d = rho.reshape(Ncopy, -1)
		theta_1d = theta.reshape(Ncopy, -1)
		plt.gcf()
		fig = plt.figure(figsize=(15,10),layout='constrained')
		# 4 subplots: 1. rho, 2. theta, 3. Lagrangian, 4. Hamiltonian 2 rows 2 columns
		ax0 = fig.add_subplot(233)
		ax1 = fig.add_subplot(231)
		theta_of_ax1 = fig.add_subplot(232)
		ax3 = fig.add_subplot(235)
		theta_of_ax3 = fig.add_subplot(236)
		largest_Lag_ax = fig.add_subplot(234)
		Lag = Lagrangian(h, dnu, rho_1d, theta_1d).real
		actionS = dnu* np.sum(Lag)*h**2*aa
		largest_Lag_time_index = np.argmax(Lag[2:-10]*h**2*aa)+2
		fig.suptitle(r'$N_t=$'+str(int(Ncopy))+r', $L_x\times L_y=$'+str(int(Ly*h))+r'$\times$'+str(int(Lx*h))+r', h='+str(h)+r', $\Delta \tau=$'+str(dtau)+r', $D=$'+str(D) +r', $T_\mathrm{max}=$'+str("%.1f"%Tmax)+"\n"+"$S=$"+str("%.6f"%(actionS))+r', $\tau=$ '+ str(i*dtau), fontsize=20)

		### PLOT rho
		mid_y = Ly // 2
		rho_slice = rho[:, mid_y, :].real
		t_edges = np.linspace(0, Tmax, Ncopy + 1)
		x_edges = np.arange(Lx + 1) - 0.5 # 現在 Y 軸是 X 空間座標
		im = ax1.pcolormesh(x_edges, t_edges, rho_slice, 
                           cmap='coolwarm', 
                           shading='flat',
                           vmin=-1.5, vmax=1.5)
		cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
		cbar.set_label(r'$\rho(x, y_{mid}, t)$', fontsize=20)
		ax1.plot([0,Lx-1], [t_edges[largest_Lag_time_index], t_edges[largest_Lag_time_index]], linewidth=2, color='red')
		ax1.tick_params(axis='both', which='major', labelsize=15)
		ax1.set_xticks([0, Lx-1])
		ax1.set_ylabel('Time', fontsize=20)
		ax1.set_xlabel(f'X Coordinate (at y={mid_y})', fontsize=15)
		Lag = Lagrangian(h, dnu, rho_1d, theta_1d).real
		actionS = dnu* np.sum(Lag)*h**2*aa

		### PLOT theta
		theta_slice = theta[:, mid_y, :].real
		min_theta = theta_slice.min()
		max_theta = theta_slice.max()
		im2 = theta_of_ax1.pcolormesh(x_edges, t_edges , theta_slice, 
							cmap='bwr', 
							shading='flat',
							vmin=-0.05, vmax=0.05)
		cbar2 = fig.colorbar(im2, ax=theta_of_ax1, shrink=0.8)
		cbar2.set_label(r'$\theta(x, y_{mid}, t)$', fontsize=20)
		theta_of_ax1.plot([0,Lx-1], [t_edges[largest_Lag_time_index], t_edges[largest_Lag_time_index]], linewidth=2, color='red')
		theta_of_ax1.tick_params(axis='both', which='major', labelsize=15)
		theta_of_ax1.set_xticks([0, Lx-1])
		theta_of_ax1.set_ylabel('Time', fontsize=20)
		theta_of_ax1.set_xlabel(f'X Coordinate (at y={mid_y})', fontsize=15)
		
		### PLOT Lagrangian and Hamiltonian
		ax0.set_aspect('auto')
		ax0.plot(np.linspace(0,1,Ncopy), Lag*h**2*aa, label=r'$L(\rho,\dot\rho)$', color='black'  ) 
		ax0.plot(np.linspace(0,1,Ncopy), Hamiltonian(h,rho_1d, theta_1d).real*h**2*aa, label=r'$H(\rho,\theta)$', color='brown' , linestyle='-.') 
		ax0.plot([t_edges[largest_Lag_time_index]/Tmax, t_edges[largest_Lag_time_index]/Tmax], [-1, 1], linewidth=2, color='red')
		ax0.legend(loc='best', fontsize=14)
		ax0.tick_params(axis='both', which='major', labelsize=15)
		ax0.set_xlabel(r'$t/T_\mathrm{Max}$', fontsize=20)
		ax0.set_ylabel(r'$L$', fontsize=20)
		ax0.set_xlim(0,1)
		ax0.set_ylim(-0.05*max(Lag[2:-10].max()*h**2*aa, Hamiltonian(h,rho_1d, theta_1d).real[2:-10].max()*h**2*aa), max(Lag[2:-10].max()*h**2*aa, Hamiltonian(h,rho_1d, theta_1d).real[2:-10].max()*h**2*aa)*1.2)
		
		# Plot rho
		mid_x = Lx // 2
		rho_slice = rho[:, :, mid_x].real.T
		t_edges = np.linspace(0, Tmax, Ncopy + 1)
		y_edges = np.arange(Ly + 1) - 0.5 # 現在 Y 軸是 X 空間座標
		im = ax3.pcolormesh(t_edges, y_edges, rho_slice, 
                           cmap='coolwarm', 
                           shading='flat',
                           vmin=-1.5, vmax=1.5)
		cbar = fig.colorbar(im, ax=ax3, shrink=0.8)
		cbar.set_label(r'$\rho(x_{mid}, y, t)$', fontsize=20)
		ax3.plot([t_edges[largest_Lag_time_index], t_edges[largest_Lag_time_index]], [0, Ly-1], linewidth=2, color='red')
		ax3.tick_params(axis='both', which='major', labelsize=15)
		ax3.set_yticks([0, Ly-1])
		ax3.set_xlabel('Time', fontsize=20)
		ax3.set_ylabel(f'Y Coordinate (at x={mid_x})', fontsize=15)
		Lag = Lagrangian(h, dnu, rho_1d, theta_1d).real
		actionS = dnu* np.sum(Lag)*h**2*aa

		### PLOT theta
		theta_slice = theta[:, :, mid_x].real.T
		min_theta = theta_slice.min()
		max_theta = theta_slice.max()
		im2 = theta_of_ax3.pcolormesh(t_edges, y_edges, theta_slice, 
							cmap='bwr', 
							shading='flat',
							vmin=-0.05, vmax=0.05)
		cbar2 = fig.colorbar(im2, ax=theta_of_ax3, shrink=0.8)
		cbar2.set_label(r'$\theta(x_{mid}, y, t)$', fontsize=20)
		theta_of_ax3.plot([t_edges[largest_Lag_time_index], t_edges[largest_Lag_time_index]], [0, Ly-1], linewidth=2, color='red')
		theta_of_ax3.tick_params(axis='both', which='major', labelsize=15)
		theta_of_ax3.set_yticks([0, Ly-1])
		theta_of_ax3.set_xlabel('Time', fontsize=20)
		theta_of_ax3.set_ylabel(f'Y Coordinate (at x={mid_x})', fontsize=15)
		
		# find the time index of the largest Lag
		rho_slice = rho[largest_Lag_time_index, :, :].real
		im = largest_Lag_ax.pcolormesh(x_edges, y_edges, rho_slice, 
							cmap='coolwarm', 
							shading='flat',
							vmin=-1.5, vmax=1.5)
		largest_Lag_ax.plot([0,Lx-1], [Ly//2, Ly//2], linewidth=2, color='red')
		largest_Lag_ax.plot([Lx//2, Lx//2], [0, Ly-1], linewidth=2, color='red')
		cbar = fig.colorbar(im, ax=largest_Lag_ax, shrink=0.8)
		cbar.set_label(r'$\rho(x, y, t)$', fontsize=20)
		largest_Lag_ax.tick_params(axis='both', which='major', labelsize=15)
		largest_Lag_ax.set_xlabel('X Coordinate', fontsize=20)
		largest_Lag_ax.set_ylabel('Y Coordinate', fontsize=20)
		largest_Lag_ax.set_title(r'$\rho$ at largest L', fontsize=20)

		if i > start_iter:
			Lag = Lagrangian(h, dnu, rho_1d, theta_1d).real
			print('Lag max, min:', Lag.max()*h**2*aa, Lag.min()*h**2*aa)
			if Lag.max() * h**2*aa > threshold or Lag.min() * h**2*aa < -threshold:
				print('Lag is not small, save checkpoint')
				np.savez_compressed('checkpoints/modelB/checkpoint_local.npz', rho=rho, theta=theta, iteration=i, Lx=Lx, Ly=Ly, h=h, Ncopy=Ncopy, Tmax=Tmax, aa=aa, D=D, kappa=kappa, dtau=dtau, upward=upward, iterations=i, plotStep=plotStep)
				break
		if(upward==True):
			plt.savefig('upward_Lx'+str(int(Lx))+'_Ly'+str(int(Ly))+'_N'+str(int(Ncopy))+'h'+str(float(h))+'_D'+str(D)+'_dtau'+str("%.1e"%dtau)+'_'+"%09d" % (i,)+'.png', format='png', bbox_inches='tight')
			print(f"Time taken: {time.time() - start_time} seconds", "iteration", i)
		else:
			plt.savefig('downward_Lx'+str(int(Lx))+'_Ly'+str(int(Ly))+'_N'+str(int(Ncopy))+'h'+str(float(h))+'_D'+str(D)+'_dtau'+str("%.1e"%dtau)+'_'+"%09d" % (i,)+'.png', format='png', bbox_inches='tight')
			print(f"Time taken: {time.time() - start_time} seconds", "iteration", i)
		plt.clf()
		plt.close()
		np.savez_compressed('checkpoints/modelB/checkpoint_local.npz', rho=rho, theta=theta, iteration=i, Lx=Lx, Ly=Ly, h=h, Ncopy=Ncopy, Tmax=Tmax, aa=aa, D=D, dtau=dtau, upward=upward, iterations=i, plotStep=plotStep)
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
    im1 = ax1.imshow(rho_sub[0], cmap='bwr', vmin=-1.5, vmax=1.5, 
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
animate_2d_heatmap(rho.real, theta.real, filename=f"2d_sim_Lx{int(Lx)}_Ly{int(Ly)}.mp4", dnu=dnu, skip=2, fps=10)












