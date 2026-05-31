import os
import torch
import time
import torch.optim as optim
from tqdm import tqdm
from model import PINN
from physics import ns_residuals, b_loss, ic_loss

# loss weights
LAMBDA_PDE = 1.0
LAMBDA_BC = 10.0
LAMBDA_IC = 1.0 

def sample_pts(
        N: int,
        device: torch.device,
        t_max: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        N (int): Number of points to sample.
        device (torch.device): The device on which to allocate the tensors.
        t_max (float): Maximum time value.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Sampled points (x, y, t).
    """
    x = torch.rand(N, 1, device=device, requires_grad=True)
    y = torch.rand(N, 1, device=device, requires_grad=True)
    t = torch.rand(N, 1, device=device, requires_grad=True) * t_max
    return x, y, t

def compute_loss(
        model: PINN,
        device: torch.device,
        N: int =2000
) -> tuple[torch.Tensor, dict]:
    """
    Args:
        model (PINN): The PINN model to compute the loss for.
        device (torch.device): The device on which to allocate tensors.
        N (int): Number of points to sample for PDE residuals.

    Returns:
        tuple[torch.Tensor, dict]: Total loss and a dictionary of individual loss components.
    """
    x, y, t = sample_pts(N, device)
    res_u, res_v, res_cont = pde_residual(model, x, y, t)
    
    l_pde = (torch.mean(res_u**2) + torch.mean(res_v**2) + torch.mean(res_cont**2)) * LAMBDA_PDE
    l_bc = b_loss(model, device) * LAMBDA_BC
    l_ic = ic_loss(model, device) * LAMBDA_IC   
    loss = l_pde + l_bc + l_ic
    loss_dict = {
        "total": loss.item(),
        "pde": l_pde.item(),
        "bc": l_bc.item(),
        "ic": l_ic.item()
    }
    return loss, loss_dict

def train(
        model: PINN,
        device: torch.device,
        epochs: int = 5000,
        lr: float = 1e-3,
        N: int = 2000,
        save_dir: str = "outputs",
        log_every: int = 100
) -> list[dict]:
    """
    Args:
        model (PINN): The PINN model to train.
        device (torch.device): The device on which to perform training.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        N (int): Number of points to sample for PDE residuals in each epoch.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # cosine annealing
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    t0 = time.time()
    
    progress = tqdm(range(epochs), desc="Training", unit="epoch")
    for epoch in progress:
        optimizer.zero_grad()
        loss, loss_dict = compute_loss(model, device, N)
        loss.backward()
        optimizer.step()
        sched.step()

        if epoch % log_every == 0 or epoch == 1:
            loss_dict["epoch"] = epoch
            loss_dict["lr"] = sched.get_last_lr()[0]
            history.append(loss_dict)

            progress.set_postfix({
                "loss": f"{loss_dict['total']:.4e}",
                "pde": f"{loss_dict['pde']:.4e}",
                "bc": f"{loss_dict['bc']:.4e}",
            })
        
        t = time.time() - t0
        print(f"\ntraining completed in {t:.2f}s")

        # save final weights
        ckpt_path = os.path.join(save_dir, "pinn_weights.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"model saved to {ckpt_path}")

        return history