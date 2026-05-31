import torch

# compute first and second order gradients for physics loss
def grad(out: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """
    Args:
        out (torch.Tensor): Output tensor from which to compute the gradient.
        inp (torch.Tensor): Input tensor with respect to which the gradient is computed.

    Returns:
        torch.Tensor: Gradient of `out` with respect to `inp`.
    """
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out), create_graph=True, retain_graph=True)[0]

# PDE residuals
def ns_residuals(model,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 t: torch.Tensor) -> tuple:
    """
    Args:
        model (nn.Module): The PINN model that outputs (u, v, p) given (x, y, t).
        x (torch.Tensor): Tensor of shape (N, 1) representing x-coordinates.
        y (torch.Tensor): Tensor of shape (N, 1) representing y-coordinates.
        t (torch.Tensor): Tensor of shape (N, 1) representing time.

    Returns:
        tuple: Residuals for the momentum equations and continuity equation.
    """
    u, v, p = model(x, y, t)
    Re = model.Re
    
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_y = grad(u, y)
    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)

    v_t = grad(v, t)
    v_x = grad(v, x)
    v_y = grad(v, y)
    v_xx = grad(v_x, x)
    v_yy = grad(v_y, y)

    p_x = grad(p, x)
    p_y = grad(p, y)

    # momentum equations residuals
    res_u = u_t + u * u_x + v * u_y + p_x - (1.0 / model.Re) * (u_xx + u_yy)
    res_v = v_t + u * v_x + v * v_y + p_y - (1.0 / model.Re) * (v_xx + v_yy)

    # continuity equation residual
    res_cont= u_x + v_y

    return res_u, res_v, res_cont

# boundary and initial condition losses
def b_loss(model, device: torch.device) -> torch.Tensor:
    """
    Args:
        model (nn.Module): The PINN model that outputs (u, v, p) given (x, y, t).
        device (torch.device): The device on which tensors are allocated.

    Returns:
        torch.Tensor: The computed loss for boundary and initial conditions.
    """
    N_bc = 100
    zeros, ones = torch.zeros(N_bc, 1, device=device), torch.ones(N_bc, 1, device=device)
    lin = torch.linspace(0, 1, N_bc, device=device).unsqueeze(1)
    t_bc =zeros
    loss = torch.tensor(0.0, device=device)

    # top wall: y=1, u=1, v=0
    u, v, _ = model(lin, ones, t_bc)
    loss += torch.mean((u - 1.0) ** 2) + torch.mean(v**2)

    #bottom wall: y=0, u=v=0
    u, v, _ = model(lin, zeros, t_bc)
    loss += torch.mean(u**2) + torch.mean(v**2)

    # left wall: x=0, u=v=0
    u, v, _ = model(zeros, lin, t_bc)
    loss += torch.mean(u**2) + torch.mean(v**2)

    # right wall: x=1, u=v=0
    u, v, _ = model(ones, lin, t_bc)
    loss += torch.mean(u**2) + torch.mean(v**2)

    return loss

def ic_loss(model, device: torch.device) -> torch.Tensor:
    """
    Args:
        model (nn.Module): The PINN model that outputs (u, v, p) given (x, y, t).
        device (torch.device): The device on which tensors are allocated.

    Returns:
        torch.Tensor: The computed loss for initial conditions.
    """
    N_ic = 100
    zeros = torch.zeros(N_ic, 1, device=device)
    lin = torch.linspace(0, 1, N_ic, device=device).unsqueeze(1)
    t_ic = zeros
    u, v, _ = model(lin, lin, t_ic)
    loss = torch.mean(u**2) + torch.mean(v**2)
    return loss