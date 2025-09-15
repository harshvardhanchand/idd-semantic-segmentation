"""Learning rate schedulers for semantic segmentation (step-per-iteration)."""

import torch
from typing import Dict

def create_scheduler(optimizer, config: Dict, total_steps: int):
    """
    Create a step-per-batch scheduler with optional warmup.
    - total_steps: number of optimizer steps in the whole run (len(train_loader) * epochs)
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")

    tr = config["training"]
    name = tr["scheduler"]

   
    warmup_default = int(0.02 * total_steps)  
    warmup_steps = tr.get("warmup_steps", warmup_default)
    warmup_steps = max(0, min(warmup_steps, max(total_steps - 1, 1)))  
    warmup_start = tr.get("warmup_start_factor", 0.01)  

    main_total = max(1, total_steps - warmup_steps) 
    min_lr = tr.get("min_lr", 0.0) 

   
    if name == "poly":
        power = tr.get("poly_power", 0.9)
        main = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=main_total, power=power
        )  

    elif name == "cosine":
       
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_total, eta_min=min_lr
        )

    elif name == "step":
        step_size = tr.get("step_size", max(1, total_steps // 3))
        gamma = tr.get("step_gamma", 0.1)
        main = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    else:
        raise ValueError(f"Unknown scheduler: {name}")

    
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_start, end_factor=1.0, total_iters=warmup_steps
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, main], milestones=[warmup_steps]
        )
    else:
        sched = main

    return sched
