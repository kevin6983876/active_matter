# KH2018 ($\rho,m$) Hamiltonian: 1D (along y) MAM/GDA notes

This note explains how to migrate `active_modelB_1_5D.py` (single-field Model B: $\rho,\theta$) to the two-field Kourbane-Houssene-style Hamiltonian, and gives the four functional derivatives in **1D (along y)** for direct coding.

> Convention: you chose 1D, so all $\nabla$ below mean $\partial_y$. Periodic BC is assumed (consistent with FFT), so integration by parts has no boundary term.

---

## 1) Model and variables

- State fields: $\rho(y,s),\; m(y,s)$
- Conjugate fields (momenta): $p_\rho(y,s),\; p_m(y,s)$
- 1D gradient: $\nabla \equiv \partial_y$, shorthand $f'=\partial_y f$, $f''=\partial_y^2 f$

Noise covariance matrix:

$$
C(\rho,m)=
\begin{pmatrix}
\rho(1-\rho) & m(1-\rho)\\
m(1-\rho) & \rho(1-\rho)
\end{pmatrix}
\equiv
\begin{pmatrix}
C_{11} & C_{12}\\
C_{12} & C_{22}
\end{pmatrix}
$$

with

$$
C_{11}=C_{22}=\rho(1-\rho),\qquad C_{12}=m(1-\rho).
$$

---

## 2) Hamiltonian (1D)

Write $H=\int dy\, f$ with

$$
H=\int dy\; f(\rho,m,p_\rho,p_m),
$$

$$
f={}(\nabla p_\rho) \big( \nabla \rho - Pe \cdot m(1-\rho) \big) + (\nabla p_m) \big( \nabla m - Pe \cdot \rho(1-\rho) \big) - 2p_m m + \begin{pmatrix} \nabla p_\rho \\ \nabla p_m \end{pmatrix}^T \underline{C} \begin{pmatrix} \nabla p_\rho \\ \nabla p_m \end{pmatrix} + \rho p_m^2
$$

---

## 3) Spatial operators needed in code

You already have FFT Laplacian (`apply_lap_2d`). For this Hamiltonian in 1D (along y), add:

- $\partial_y f \leftrightarrow i k_y \tilde f$
- $\partial_y^2 f \leftrightarrow -k_y^2 \tilde f$

With `KY` already built (shape `(Ly,Lx)` and `Lx=1`), add `apply_grad_y(field)` for $\partial_y$.

---

## 4) Four functional derivatives (core formulas)

All formulas below assume periodic BC.

### 4.1 $\delta H/\delta p_\rho$

$$
\frac{\delta H}{\delta p_\rho}
=-\partial_y\!\left[
\rho'-Pe\,m(1-\rho)
+2\left(C_{11}p_\rho'+C_{12}p_m'\right)
\right].
$$

### 4.2 $\delta H/\delta p_m$

$$
\frac{\delta H}{\delta p_m}
=-\partial_y\!\left[
m'-Pe\,\rho(1-\rho)
+2\left(C_{12}p_\rho'+C_{22}p_m'\right)
\right]
-2m+2\rho p_m.
$$

### 4.3 $\delta H/\delta \rho$

Useful partials:

$$
\partial_\rho C_{11}=\partial_\rho C_{22}=1-2\rho,\qquad
\partial_\rho C_{12}=-m.
$$

Then

$$\frac{\delta H}{\delta \rho}=-p_{\rho}''+Pe\,m\,p_{\rho}'-Pe(1-2\rho)p_m'+(1-2\rho)((p_{\rho}')^2+(p_m')^2)-2mp_{\rho}'p_m'+p_m^2$$

### 4.4 $\delta H/\delta m$

Useful partials:

$$
\partial_m C_{12}=1-\rho,\qquad
\partial_m C_{11}=\partial_m C_{22}=0.
$$

Then

$$
\frac{\delta H}{\delta m}
=-p_m''
-Pe(1-\rho)\,p_\rho'
-2p_m
+2(1-\rho)\,p_\rho' p_m'.
$$

---

## 5) Canonical equations used by MAM/GDA

Standard form:

$$
\dot\rho=\frac{\delta H}{\delta p_\rho},\qquad
\dot m=\frac{\delta H}{\delta p_m},\qquad
\dot p_\rho=-\frac{\delta H}{\delta \rho},\qquad
\dot p_m=-\frac{\delta H}{\delta m}.
$$

Expanded PDEs:

$$
\dot\rho
=-\partial_y\!\left[
\rho'-Pe\,m(1-\rho)
+2(C_{11}p_\rho'+C_{12}p_m')
\right]
$$

$$
\dot m
=-\partial_y\!\left[
m'-Pe\,\rho(1-\rho)
+2(C_{12}p_\rho'+C_{22}p_m')
\right]
-2m+2\rho p_m
$$

$$
\dot p_\rho
=p_{\rho}''
-Pe\,m\,p_\rho'
+Pe(1-2\rho)\,p_m'
-(1-2\rho)((p_\rho')^2+(p_m')^2)
+2m\,p_\rho' p_m'
-p_m^2
$$

$$
\dot p_m
=p_m''
+Pe(1-\rho)\,p_\rho'
+2p_m
-2(1-\rho)\,p_\rho' p_m'
$$

---

## 6) Mapping to your current `active_modelB_1_5D.py`

Current structure (single-field Model B):

- field: `rho`, conjugate: `theta`
- `Hamiltonian(h, rho, theta)` is Model-B-specific
- loop uses hand-written `dH_drho`, `dH_dtheta`

Target structure (two-field):

- add arrays: `rho, m` and `p_rho, p_m`
- add operators/functions:
  - `apply_grad_y()`
  - `Hamiltonian_KH_1D(h, Pe, rho, m, p_rho, p_m)`
  - `Lagrangian_KH_1D(...) = sum(rhoDot*p_rho + mDot*p_m) - H`

For minimum disruption, keep your banded upwind style and use two pairs:

- $U_\rho=\rho+p_\rho,\;V_\rho=\rho-p_\rho$
- $U_m=m+p_m,\;V_m=m-p_m$

Then rewrite loop reactions using the derivatives in Section 4.

---

## 7) Minimal consistency checks (must do)

1. Deterministic limit: set $p_\rho=p_m=0$ and verify
   - $\dot\rho=\partial_y^2\rho-Pe\,\partial_y[m(1-\rho)]$
   - $\dot m=\partial_y^2 m-Pe\,\partial_y[\rho(1-\rho)]-2m$

2. Cross-noise term in code must be
   - `C11*gpr**2 + 2*C12*gpr*gpm + C22*gpm**2`
   - with `C12 = m*(1-rho)`

3. Nonconserved term check:
   - Hamiltonian contains `rho * p_m**2`
   - $\delta H/\delta p_m$ includes `+2*rho*p_m`
   - $\delta H/\delta rho$ includes `+p_m**2`

---

## 8) Implementation hints (matching your array layout)

- Your fields are `(Ncopy, Ly, Lx)`.
- For Hamiltonian/Lagrangian scalar per path-slice, reshape to `(Ncopy, Ly*Lx)` only at interfaces if needed.
- Keep `apply_grad_y` and `apply_lap_2d` both in FFT form on axes `(1,2)` for consistency.
