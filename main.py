import torch
from src import PINN, train, evaluate_on_grid, plot_flow_field, plot_loss_history

CONFIG = {
    "Re": 100.0,            # Reynolds number. Try 100 → 400 → 1000

    "hidden_layers": 6,     # Network depth
    "hidden_width":  64,    # Neurons per layer

    "epochs":    5000,      # Total gradient steps
    "lr":        1e-3,      # Initial learning rate (cosine-annealed to 1e-5)
    "N_colloc":  2000,      # Collocation points per step

    "save_dir":  "outputs",
    "log_every": 100,
}

# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Re     : {CONFIG['Re']}")
    print(f"Epochs : {CONFIG['epochs']}")
    print()

    # ── 1. Build model ────────────────────────────────────────────────
    model = PINN(
        hidden_layers=CONFIG["hidden_layers"],
        hidden_width=CONFIG["hidden_width"],
        Re=CONFIG["Re"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── 2. Train ──────────────────────────────────────────────────────
    history = train(
        model=model,
        device=device,
        epochs=CONFIG["epochs"],
        lr=CONFIG["lr"],
        N_colloc=CONFIG["N_colloc"],
        save_dir=CONFIG["save_dir"],
        log_every=CONFIG["log_every"],
    )

    # ── 3. Visualise ──────────────────────────────────────────────────
    fields = evaluate_on_grid(model, device, N=150, t_eval=0.0)

    plot_flow_field(
        fields,
        save_path=f"{CONFIG['save_dir']}/flow_field_Re{int(CONFIG['Re'])}.png",
        title=f"Lid-Driven Cavity Flow — PINN  (Re = {int(CONFIG['Re'])})",
    )

    plot_loss_history(
        history,
        save_path=f"{CONFIG['save_dir']}/loss_history.png",
    )

    print("\nDone. Check the outputs/ directory for plots.")


if __name__ == "__main__":
    main()