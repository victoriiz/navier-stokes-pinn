# NS-PINN: Navier-Stokes Physics-Informed Neural Network

An implementation of a **Physics-Informed Neural Network (PINN)**
that solves the 2D incompressible Navier-Stokes equations for lid-driven cavity flow.

## Project Structure

```
nspinn/
├── README.md
├── requirements.txt
├── src/
│   ├── model.py        # Neural network architecture
│   ├── physics.py      # PDE residuals & boundary conditions
│   ├── trainer.py      # Training loop & loss orchestration
│   └── visualize.py    # Flow field visualization
├── outputs/            # Saved weights & plots (auto-created)
└── main.py             # Entry point — run this
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

After training, a plot of the velocity and pressure fields is saved to `outputs/`.

---

## Governing Equations
The 2D incompressible Navier-Stokes equations in non-dimensional form,
for unknowns u(x,y,t), v(x,y,t), p(x,y,t) over domain [0,1]² × [0,T]:

**Continuity** (mass conservation — fluid doesn't compress or pile up):

```
∂u/∂x + ∂v/∂y = 0
```

**x-momentum** (Newton's 2nd law in the x-direction):

```
∂u/∂t + u·∂u/∂x + v·∂u/∂y = -∂p/∂x + (1/Re)·(∂²u/∂x² + ∂²u/∂y²)
```

**y-momentum** (Newton's 2nd law in the y-direction):

```
∂v/∂t + u·∂v/∂x + v·∂v/∂y = -∂p/∂y + (1/Re)·(∂²v/∂x² + ∂²v/∂y²)
```

Each term has a physical meaning:

| Term | Meaning |
|---|---|
| `∂u/∂t` | Local acceleration |
| `u·∂u/∂x + v·∂u/∂y` | Convective transport (nonlinear — causes turbulence at high Re) |
| `-∂p/∂x` | Pressure gradient force |
| `(1/Re)·(∂²u/∂x² + ∂²u/∂y²)` | Viscous diffusion (smoothing) |

All quantities are non-dimensionalised against lid velocity U=1 and cavity length L=1.
The **Reynolds number** Re = ρUL/μ is the single governing parameter.

---

## Boundary Conditions

The lid-driven cavity on the unit square [0,1]²:

```
 ←——— u=1, v=0 ———→   (moving lid)
┌─────────────────────┐  y = 1
│                     │
│       fluid         │  u=0, v=0 on left (x=0) and right (x=1) walls
│                     │
└─────────────────────┘  y = 0
       u=0, v=0           (stationary bottom wall)
```

Explicitly:

| Wall | Condition | Physical meaning |
|---|---|---|
| Top    (y=1) | u=1, v=0 | Lid moves rightward at unit speed |
| Bottom (y=0) | u=0, v=0 | No-slip stationary wall |
| Left   (x=0) | u=0, v=0 | No-slip stationary wall |
| Right  (x=1) | u=0, v=0 | No-slip stationary wall |

The moving lid drives a recirculating vortex. At Re=100 this is a single
smooth oval vortex. Secondary corner vortices appear as Re increases.

---

## Initial Condition

Fluid at rest everywhere at t=0:

```
u(x, y, 0) = 0    for all (x, y) ∈ [0,1]²
v(x, y, 0) = 0    for all (x, y) ∈ [0,1]²
```

Pressure has no initial condition — in incompressible flow, pressure adjusts
instantaneously to enforce continuity and has no temporal memory.

---

## What it learns

The network takes a point `(x, y, t)` in space-time and outputs `(u, v, p)`:
- `u`, `v` — fluid velocity components
- `p`       — pressure

It is trained with **no simulation data**. Instead, the loss function encodes:
1. The Navier-Stokes momentum equations (PDE residual)
2. The continuity equation (mass conservation)
3. Boundary conditions (no-slip walls + moving lid)
4. Initial condition (fluid at rest)

## Key concepts

| Concept | Where to look |
|---|---|
| Network architecture (stream-function trick) | `src/model.py` |
| Automatic differentiation for PDE terms | `src/physics.py` |
| Composite loss & collocation points | `src/trainer.py` |
| Visualizing the learned flow field | `src/visualize.py` |

## Reynolds number

Configured in `main.py` via `Re = 100`. Lower = smoother flow, faster training.
Increase to 400–1000 to see secondary vortices emerge (needs more epochs).

---

## Reference

Sitzmann et al., *Implicit Neural Representations with Periodic Activation
Functions* (NeurIPS 2020) — the SIREN architecture used here.

Ghia et al. (1982) — benchmark velocity profiles for lid-driven cavity flow,
useful for quantitative validation of the trained solution.
