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

def apply_grad_y(field):
    """
    ∂_y field (1D along y), via FFT.
    field shape: (N_current, Ly, Lx)
    returns: same shape (real part)
    """
    field_k = sp_fft.fft2(field, axes=(1, 2))
    grad_k = 1j * KY * field_k          # ∂_y ↔ i*k_y
    return sp_fft.ifft2(grad_k, axes=(1, 2)).real

# def Hamiltonian(h, rho, theta):
# 	# rho, theta: (Ncopy, Ly*Lx) 以配合 Lagrangian
# 	rho_3d = rho.reshape(Ncopy, Ly, Lx)
# 	theta_3d = theta.reshape(Ncopy, Ly, Lx)
# 	lap_rho = apply_lap_2d(rho_3d)
# 	lap_theta = apply_lap_2d(theta_3d)
# 	det_part = apply_lap_2d(-D * lap_rho - rho_3d + rho_3d**3)
# 	noise_part = -0.5 * aa * theta_3d * lap_theta
# 	H = np.sum(det_part * theta_3d + noise_part, axis=(1, 2))
# 	return H

# def Lagrangian(h, ds, rho, theta): #return an array of size Ncopy, axis=1 sums the colums
# 	rhoDot      = (np.roll(rho,-1,axis=0) - np.roll(rho,1,axis=0))/(2*ds)
# 	rhoDot[0,:] = (np.roll(rho,-1,axis=0) - rho)[0,:] /(ds)
# 	rhoDot[Ncopy-1,:] = (-np.roll(rho,1,axis=0) + rho)[Ncopy-1,:]/(ds)
# 	Ham = Hamiltonian(h, rho, theta)
# 	L =  np.sum( rhoDot * theta, axis=1 ) - Ham
# 	return L

# def Lagrangian_Theo(h, rho, theta):
# 	# rho, theta: (Ncopy, Ly*Lx)
# 	rho_3d = rho.reshape(Ncopy, Ly, Lx)
# 	theta_3d = theta.reshape(Ncopy, Ly, Lx)
# 	lap_theta = apply_lap_2d(theta_3d)
# 	# L_theo = -0.5 * aa * theta * lap_theta 的空間積分
# 	L_density = -0.5 * aa * theta_3d * lap_theta
# 	return np.sum(L_density, axis=(1, 2))

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


################################
""" Start MAM """
##############################

Lx = 1
Ly = 100
h  = 0.1
Ncopy =400
Tmax = 100.

s = np.linspace(0,Tmax,Ncopy)
ds  = s[1]-s[0]
dnu = s[1]-s[0]
Pe = 6.0   # TODO: pick your Péclet-like parameter (must match KH paper's convention)


upward = True # choose if path from -1 to +1 (upward), or the opposite

rho_0 = 0.55

dtau = 0.1 


# theoreticla spinodal points
high =3/4+1/4*np.sqrt(1-16/Pe**2)
low =3/4-1/4*np.sqrt(1-16/Pe**2)
print('theoretical spinodal points: high =', high, 'low =', low)
r = dtau/dnu
print('conditions: Ly,Lx,Ncopy =', Ly,Lx,Ncopy, 'h =', h, 'Pe =', Pe, 'dtau =', dtau, 'Tmax =', Tmax)
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
	stab = dtau * gamma * (2.0 * k2_flat[col])
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

rho   = np.zeros((Ncopy, Ly, Lx), dtype=complex)
m     = np.zeros((Ncopy, Ly, Lx), dtype=complex)
p_rho = np.zeros((Ncopy, Ly, Lx), dtype=complex)
p_m   = np.zeros((Ncopy, Ly, Lx), dtype=complex)

y_coords = np.arange(Ly)
x_coords = np.arange(Lx)
Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
if(upward==True):
	# ==========================================
	# 使用 tanh 構造完美的物理液滴 (rho1)
	# ==========================================
	# 決定液滴的寬度 (W)。這裡我們可以用槓桿原理算出完美的寬度
	W = Ly/2

	interface_thickness = 50  # 介面的平滑度 (建議 2~3，讓 FFT 算導數時不會爆炸)

	# 利用兩個 tanh 相減，造出一個置中於 Ly/2 的平滑方波
	profile = 0.5*(np.tanh((Y - (Ly/2.0 - W/2.0)) / interface_thickness) - np.tanh((Y - (Ly/2.0 + W/2.0)) / interface_thickness))

	rho1 = profile + 0j
	# 【關鍵防護】強制質量完美守恆，確保總體平均密度絕對是 rho_0
	if rho_0 > 0.5:
		rho1 = (rho1*(1-rho_0)*2+2*rho_0-1)/np.mean(rho1*(1-rho_0)*2+2*rho_0-1)*rho_0
	else:
		rho1 = rho1/np.mean(rho1)*rho_0
	rho1_k = sp_fft.fft2(rho1.real)
	grad_rho1 = sp_fft.ifft2(1j * KY * rho1_k).real
	m1 = np.zeros((Ly,Lx), dtype=complex)
	m1_real = (1.0 / Pe) * grad_rho1 / (1.0 - rho1.real)
	m1 = m1_real + 0j
	
print('rho1 mean',np.mean(rho1),'m1 mean',np.mean(m1))

# ==========================================
# [新增] 預先放鬆 (Pre-relaxation) 區塊
# 目的：讓銳利的初始猜測自然擴散成真實的物理穩態
# ==========================================
print("Relaxing Initial and Final states to exact Model B equilibria...")
dt_relax = 0.001  # 對於你設定的 h=0.3, D=0.002，這個步長非常安全
relax_steps = 20000


def relax_kh_physical_states(rho_state, m_state, tol=1e-7, max_steps=500000):
	rho_2d = rho_state.copy().real
	m_2d   = m_state.copy().real
	target_mass = np.mean(rho_2d)
	print("開始尋找 KH2018 物理穩態 (dot(rho)=dot(m)=0)...")
	for step in range(1, max_steps + 1):
		rho_old = rho_2d.copy()
		m_old   = m_2d.copy()
		# FFT components (reuse for Laplacian)
		rho_k = sp_fft.fft2(rho_2d)
		m_k   = sp_fft.fft2(m_2d)
		# ∂_y^2 f  (1.5D: Lx=1 => k2_2d effectively KY^2)
		rho_dd = sp_fft.ifft2(-k2_2d * rho_k).real
		m_dd   = sp_fft.ifft2(-k2_2d * m_k).real
		# ∂_y [ m(1-rho) ] and ∂_y [ rho(1-rho) ]
		fm = m_2d * (1.0 - rho_2d)
		fr = rho_2d * (1.0 - rho_2d)
		grad_fm = sp_fft.ifft2(1j * KY * sp_fft.fft2(fm)).real
		grad_fr = sp_fft.ifft2(1j * KY * sp_fft.fft2(fr)).real
		# KH2018 at p=0 (真正的物理方程式 RHS):
		# dot_rho = ∂_t ρ = ∇²ρ - Pe*∂_y[m(1-ρ)]
		# dot_m   = ∂_t m = ∇²m - Pe*∂_y[ρ(1-ρ)] - 2m
		dot_rho = rho_dd - Pe * grad_fm
		dot_m   = m_dd   - Pe * grad_fr - 2.0 * m_2d
		# relaxation step: explicit Euler (用加號推進時間)
		rho_2d = rho_2d + dt_relax * dot_rho
		m_2d   = m_2d   + dt_relax * dot_m
		# numeric mass correction
		rho_2d = rho_2d - np.mean(rho_2d) + target_mass
		# ====================================================
		# 每 1000 步檢查一次收斂
		# ====================================================
		if step % 1000 == 0:
			max_resid_rho = np.max(np.abs(dot_rho))
			max_resid_m   = np.max(np.abs(dot_m))
			print(f"   [Step {step:5d}] |dot(rho)|_inf = {max_resid_rho:.2e} |dot(m)|_inf = {max_resid_m:.2e}")
			if max_resid_rho < tol and max_resid_m < tol:
				print(f"✅ 成功！第 {step} 步抵達 KH2018 物理穩態！")
				break
	if step == max_steps:
		print("⚠️ 警告：達到最大步數但尚未完全收斂，請考慮增加 max_steps 或檢查初始物理設定。")
	return rho_2d + 0j, m_2d + 0j
rho1_unrelaxed = rho1.copy()
m1_unrelaxed = m1.copy()
print("--- 放鬆起點 (rho1, m1) ---")
rho1, m1 = relax_kh_physical_states(rho1, m1, max_steps=1000000)

fig_states, axs = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
mid_x = Lx // 2
im_u1 = axs[0, 0].imshow(rho1_unrelaxed.real, cmap='bwr', vmin=0, vmax=1, origin='lower', aspect='auto')
axs[0, 0].set_title("Unrelaxed State (rho1)")
axs[0, 0].set_xlabel("x index")
axs[0, 0].set_ylabel("y index")
# Panel [0, 1]: Unrelaxed m1
im_u2 = axs[0, 1].imshow(m1_unrelaxed.real, cmap='bwr', vmin=m1_unrelaxed.real.min(), vmax=m1_unrelaxed.real.max(), origin='lower', aspect='auto')
axs[0, 1].set_title("Unrelaxed State (m1)")
axs[0, 1].set_xlabel("x index")
axs[0, 1].set_ylabel("y index")
fig_states.colorbar(im_u2, ax=axs[0, 1], shrink=0.8, label=r"$m$")

# Panel [0, 2]: Unrelaxed 1D Cross-section
axs[0, 2].plot(rho1_unrelaxed.real[:, mid_x], 'o-', label='rho1 (Start)', color='blue', markersize=4)
axs[0, 2].plot(rho1.real[:, mid_x], 's-', label='rho1 (End)', color='red', markersize=4)
axs[0, 2].axhline(0, color='black', linestyle='--', alpha=0.5)
axs[0, 2].axhline(1, color='gray', linestyle=':', alpha=0.5)
axs[0, 2].axhline(-1, color='gray', linestyle=':', alpha=0.5)
axs[0, 2].set_title(f"1D Cross-section (at x={mid_x})")
axs[0, 2].set_ylim(0, 1.5)
axs[0, 2].set_xlabel("y index")
axs[0, 2].set_ylabel(r"Density $\rho$")
axs[0, 2].legend()

# ------------------------------------------
# 第二排：已放鬆 (Relaxed)
# ------------------------------------------
# Panel [1, 0]: Relaxed rho1
im_r1 = axs[1, 0].imshow(rho1.real, cmap='bwr', vmin=0, vmax=1, origin='lower', aspect='auto')
axs[1, 0].set_title("Relaxed State (rho1)")
axs[1, 0].set_xlabel("x index")
axs[1, 0].set_ylabel("y index")
fig_states.colorbar(im_r1, ax=axs[1, 0], shrink=0.8, label=r"$\rho$")

# Panel [1, 1]: Relaxed m1
im_r2 = axs[1, 1].imshow(m1.real, cmap='bwr', vmin=m1.real.min(), vmax=m1.real.max(), origin='lower', aspect='auto')
axs[1, 1].set_title("Relaxed State (m1)")
axs[1, 1].set_xlabel("x index")
axs[1, 1].set_ylabel("y index")
fig_states.colorbar(im_r2, ax=axs[1, 1], shrink=0.8, label=r"$m$")

# Panel [1, 2]: Relaxed 1D Cross-section
axs[1, 2].plot(m1_unrelaxed.real[:, mid_x], 'o-', label='m1 (Start)', color='blue', markersize=4)
axs[1, 2].plot(m1.real[:, mid_x], 's-', label='m1 (End)', color='red', markersize=4)
axs[1, 2].set_title(f"1D Cross-section (at x={mid_x})")
axs[1, 2].set_ylim(-1, 1)
axs[1, 2].set_xlabel("y index")
axs[1, 2].set_ylabel(r"Density $m$")
axs[1, 2].legend()
fig_states.suptitle("Pe="+str(Pe)+', rho_0='+str(rho_0), fontsize = 20)
relax_plot_name = f'relaxed_states_Ly{Ly}_Lx{Lx}_Pe{Pe}.png'
plt.savefig(relax_plot_name, dpi=150, facecolor='white')
plt.close(fig_states)
print(f"[*] Saved relaxed states plot to: {relax_plot_name}")
# ==========================================
