import torch



def interleaving(model , mode:str):
    '''
    mode = 'seg' or 'det
    function for freezing the head that is the opposite of the mode
    eg : mode = 'seg' --> freeze the detection head
    '''
    if mode not in {'seg', 'det'}:
        raise ValueError("mode must be either 'seg' or 'det'")
    
    requires_grad_map = {
        'seg': {'seg_head': True, 'det_head': False},
        'det': {'seg_head': False, 'det_head': True}
    }
    
    for head_name, requires_grad in requires_grad_map[mode].items():
        for p in getattr(model, head_name).parameters():
            p.requires_grad = requires_grad
    model.train()
    return model


def lr_scheduler(optimizer, scheduler: str, **kwargs):
    scheduler = scheduler.lower()
    if scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.1)
        )
    elif scheduler == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get("milestones", [30, 80]),
            gamma=kwargs.get("gamma", 0.1)
        )
    elif scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 50),
            eta_min=kwargs.get("eta_min", 0)
        )
    elif scheduler == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("factor", 0.1),
            patience=kwargs.get("patience", 5),
            verbose=True
        )
    elif scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", 0.01),
            steps_per_epoch=kwargs.get("steps_per_epoch", 100),
            epochs=kwargs.get("epochs", 10),
            pct_start=kwargs.get("pct_start", 0.3)
        )
    elif scheduler == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get("gamma", 0.99)
        )
    else:
        raise ValueError(
            f"Unknown scheduler: {scheduler}. Choose from "
            "[step, multistep, cosine, reduce_on_plateau, onecycle, exponential]"
        )
        