import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


def evaluate_on_grid(
    model,
    device: torch.device,
    N: int = 100,
    t_eval: float = 0.0,
) -> dict:
    """
    Args:
        model  : trained PINN
        device : torch device
        N      : grid resolution
        t_eval : time at which to evaluate (use 0 for steady-state target)

    Returns:
        dict with keys: X, Y, U, V, P, speed
    """
    model.eval()
    with torch.no_grad():
        xs = torch.linspace(0, 1, N)
        ys = torch.linspace(0, 1, N)
        XX, YY = torch.meshgrid(xs, ys, indexing="ij")   # (N, N)

        x_flat = XX.reshape(-1, 1).to(device)
        y_flat = YY.reshape(-1, 1).to(device)
        t_flat = torch.full_like(x_flat, t_eval)

        u, v, p = model(x_flat, y_flat, t_flat)

    U = u.cpu().numpy().reshape(N, N)
    V = v.cpu().numpy().reshape(N, N)
    P = p.cpu().numpy().reshape(N, N)
    X = XX.numpy()
    Y = YY.numpy()

    # normalize
    P -= P.mean()

    return {"X": X, "Y": Y, "U": U, "V": V, "P": P, "speed": np.sqrt(U**2 + V**2)}


def plot_flow_field(
    fields: dict,
    save_path: str = "outputs/flow_field.png",
    title: str = "Lid-Driven Cavity Flow — PINN Solution",
):
    """
    Render the four-panel flow field figure and save to disk.
    """
    X, Y, U, V, P, speed = (
        fields["X"], fields["Y"], fields["U"], fields["V"],
        fields["P"], fields["speed"],
    )

    fig = plt.figure(figsize=(12, 10), facecolor="#0d1117")
    fig.suptitle(title, color="#e6edf3", fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    cmap_vel  = "plasma"
    cmap_comp = "RdBu_r"
    cmap_pres = "viridis"
    text_col  = "#e6edf3"
    label_col = "#8b949e"

    def _ax(pos, title_str):
        ax = fig.add_subplot(pos)
        ax.set_facecolor("#161b22")
        ax.set_title(title_str, color=text_col, fontsize=11, pad=8)
        ax.tick_params(colors=label_col, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.set_xlabel("x", color=label_col, fontsize=9)
        ax.set_ylabel("y", color=label_col, fontsize=9)
        return ax

    # ── Panel 1: Speed + streamlines ─────────────────────────────────
    ax1 = _ax(gs[0, 0], "Velocity Magnitude + Streamlines")
    im1 = ax1.pcolormesh(X, Y, speed, cmap=cmap_vel, shading="gouraud")
    ax1.streamplot(
        X.T, Y.T, U.T, V.T,
        color="white", linewidth=0.6, density=1.4, arrowsize=0.8,
    )
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(colors=label_col, labelsize=7)
    cb1.set_label("|u|", color=label_col, fontsize=8)

    # ── Panel 2: Horizontal velocity u ───────────────────────────────
    ax2 = _ax(gs[0, 1], "Horizontal Velocity  u(x, y)")
    lim2 = max(abs(U.min()), abs(U.max()))
    im2 = ax2.pcolormesh(X, Y, U, cmap=cmap_comp, vmin=-lim2, vmax=lim2, shading="gouraud")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors=label_col, labelsize=7)
    cb2.set_label("u", color=label_col, fontsize=8)

    # ── Panel 3: Vertical velocity v ─────────────────────────────────
    ax3 = _ax(gs[1, 0], "Vertical Velocity  v(x, y)")
    lim3 = max(abs(V.min()), abs(V.max()))
    im3 = ax3.pcolormesh(X, Y, V, cmap=cmap_comp, vmin=-lim3, vmax=lim3, shading="gouraud")
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.ax.tick_params(colors=label_col, labelsize=7)
    cb3.set_label("v", color=label_col, fontsize=8)

    # ── Panel 4: Pressure ─────────────────────────────────────────────
    ax4 = _ax(gs[1, 1], "Pressure  p(x, y)")
    im4 = ax4.pcolormesh(X, Y, P, cmap=cmap_pres, shading="gouraud")
    cb4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cb4.ax.tick_params(colors=label_col, labelsize=7)
    cb4.set_label("p", color=label_col, fontsize=8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Flow field plot saved → {save_path}")


def plot_loss_history(
    history: list[dict],
    save_path: str = "outputs/loss_history.png",
):
    """
    Plot training loss curves (total, PDE, BC, IC) over epochs.
    """
    epochs = [h["epoch"] for h in history]
    total  = [h["total"] for h in history]
    pde    = [h["pde"]   for h in history]
    bc     = [h["bc"]    for h in history]
    ic     = [h["ic"]    for h in history]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")

    colors = {"Total": "#58a6ff", "PDE": "#f78166", "BC": "#3fb950", "IC": "#d2a8ff"}
    for label, data, col in zip(
        ["Total", "PDE", "BC", "IC"],
        [total, pde, bc, ic],
        colors.values(),
    ):
        ax.semilogy(epochs, data, label=label, color=col, linewidth=2)

    ax.set_xlabel("Epoch", color="#8b949e")
    ax.set_ylabel("Loss (log scale)", color="#8b949e")
    ax.set_title("Training Loss History", color="#e6edf3", fontweight="bold")
    ax.tick_params(colors="#8b949e")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, alpha=0.15, color="#8b949e")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Loss history plot saved → {save_path}")