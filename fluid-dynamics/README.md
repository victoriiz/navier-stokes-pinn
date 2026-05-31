# NS-PINN: Navier-Stokes Physics-Informed Neural Network

A clean, educational implementation of a **Physics-Informed Neural Network (PINN)**
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
