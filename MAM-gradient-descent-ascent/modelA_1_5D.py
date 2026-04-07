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

################################
""" Start MAM """
##############################

Lx = 1
Ly = 50
h  = 1
Ncopy = 4000 
Tmax = 80.


s = np.linspace(0,Tmax,Ncopy)
ds  = s[1]-s[0]
dnu = s[1]-s[0]
aa = 2.   #noise amplitude


upward = False # choose if path from -1 to +1 (upward), or the opposite
previous_data = True
reverse = False
threshold = 1000

D     = 5
kappa = 0.26

solRho = np.sort(np.roots([-1, kappa, 1, 0])) # rho-, rhos, rho+ 

dtau = 0.5

iterations = 40000
plotStep   = 200
resume_file = "checkpoints/checkpoint_local.npz"
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
	if os.path.exists('checkpoints/checkpoint24.npz'):
		data = np.load('checkpoints/checkpoint24.npz')
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
for i in range(start_iter, iterations+1):
	U[:] = rho + theta
	V[:] = rho - theta
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
	for col in range(N_space):
		stab_term = gamma * (D * k4_flat[col] + 2.0 * k2_flat[col]) * U_Fourier_flat[:, col]
		RHS_col = RHS_flat[:, col] + dtau * stab_term
		U_new_flat[:, col] = solve_banded((0, 2), A_banded[col], RHS_col)
	U2_Fourier[:] = U_new_flat.reshape(Ncopy, Ly, Lx)

	U[:] = np.fft.ifft2(U2_Fourier, axes=(1, 2)).real
	rho = 0.5 * (U + V)
	theta = 0.5 * (U - V)
	rho[0] = rho1
	rho[Ncopy-1] = rho2
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
		stab_term = gamma * (D * k4_flat[col] + 2.0 * k2_flat[col]) * V_Fourier_flat[:, col]
		RHS_col = RHS_V_flat[:, col] + dtau * stab_term
		V_new_flat[:, col] = solve_banded((2, 0), B_banded[col], RHS_col)
	V2_Fourier[:] = V_new_flat.reshape(Ncopy, Ly, Lx)

	V[:] = np.fft.ifft2(V2_Fourier, axes=(1, 2)).real
	rho = 0.5 * (U + V)
	theta = 0.5 * (U - V)
	rho[0] = rho1
	rho[Ncopy-1] = rho2
	U = rho + theta
	V = rho - theta
	rho_1d = rho.reshape(Ncopy, -1)
	theta_1d = theta.reshape(Ncopy, -1)
	Lag = Lagrangian(h, dnu, rho_1d, theta_1d).real
	actionS = dnu* np.sum(Lag)*h*aa
	if Lag[-1] > threshold or Lag[-1] < -threshold:
		print(f"iter {i} Lagrangian is too large: {Lag[-1]}")
		break
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
		# if max_diff_rho < 5e-5 and H_std < 0.0002:
		# 	print(f"\n✨ 恭喜！系統在第 {i} 步已完美收斂抵達 Instanton 鞍點！")
		# 	print(f"最大數值誤差 dRho: {max_diff_rho:.2e} | 物理守恆 H 標準差: {H_std:.2e}")
		# 	break  # 安全跳出，完美存檔！
			
	# 紀錄當前狀態，留給下一步比對
	old_rho_for_check = rho.copy()
	##################################################
	'''------------ THIRD: PLOT DATA ---------'''
	##################################################	
	    
	if( i%plotStep == 0):
		print('Lag[-1]:', Lag[-1])
		if i > start_iter:
			print('max_diff_rho, H_std:', max_diff_rho, H_std)
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
		cbar.set_label(r'Time', fontsize=20)

		### PLOT rho
		rho_map = rho_1d.real.T  # Shape: (Ly*Lx, Ncopy)
		t_edges = np.linspace(0, Tmax, Ncopy + 1)
		n_space = rho_1d.shape[1]  # Ly*Lx
		x_edges = np.arange(n_space + 1) - 0.5
		im = ax1.pcolormesh(t_edges, x_edges, rho_map, 
                           cmap='coolwarm', 
                           shading='flat',
                           vmin=-1.5, vmax=1.5)
		cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
		cbar.set_label(r'$\rho$', fontsize=20)
		ax1.tick_params(axis='both', which='major', labelsize=15)
		ax1.set_yticks([0, n_space-1])
		ax1.set_xlabel('Time', fontsize=20)
		ax1.set_ylabel('Space index (0 to L-1)', fontsize=20)
		Lag = Lagrangian(h, dnu, rho_1d, theta_1d).real
		actionS = dnu* np.sum(Lag)*h*aa

		### PLOT theta
		theta_map = theta_1d.real.T  # Shape: (Ly*Lx, Ncopy)
		im2 = ax2.pcolormesh(t_edges, x_edges, theta_map, 
							cmap='bwr', 
							shading='flat',
							vmin=-0.5, vmax=0.5)
		cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
		cbar2.set_label(r'$\theta$', fontsize=20)
		ax2.tick_params(axis='both', which='major', labelsize=15)
		ax2.set_yticks([0, n_space-1])
		ax2.set_xlabel('Time', fontsize=20)
		ax2.set_ylabel('Space index (0 to L-1)', fontsize=20)

		### PLOT Lagrangian and Hamiltonian
		ax3.set_aspect('auto')
		ax3.plot(np.linspace(0,1,Ncopy), Lag*h*aa, label=r'$L(\rho,\dot\rho)$', color='black'  ) 
		ax3.plot(np.linspace(0,1,Ncopy), Hamiltonian(h,rho_1d, theta_1d).real*h*aa, label=r'$H(\rho,\theta)$', color='brown' ) 
		plt.legend(loc='best', fontsize=14)
		ax3.tick_params(axis='both', which='major', labelsize=15)
		ax3.set_xlabel(r'$t/T_\mathrm{Max}$', fontsize=20)
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












