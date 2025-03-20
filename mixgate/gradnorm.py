import torch
from torch import nn
import torch.nn.functional as F


class GradNorm:

    """
    GradNorm implementation designed for maximal compatibility with PyTorch training frameworks.

    API for this GradNorm implementation:
        1. Initialize the GradNorm class with the model, alpha, and number of tasks
        2. Compute your task losses, as you would normally, store in a tensor of shape [T]
        3. Apply gradnorm, passing losses as input; w_i updated automatically
        4. Perform backpropagation to your model as usual
    """
    def __init__(self, layer: nn.Module, alpha: float, number_of_tasks: int, lr: float = None, lr_warmup: float = None, device: str = "cpu"):
        """
        Initialize the GradNorm class.
        
        :param layer: The multitask learning layer shared by all tasks.
        :param alpha: The GradNorm alpha parameter, higher if tasks are more different.
        :param number_of_tasks: Number of tasks in the multitask learning model.
        """
        self.layer = layer
        self.alpha = alpha
        self.T = number_of_tasks
        self.device = torch.device(device)
        self.w_i = torch.nn.Parameter(torch.ones(self.T, device=self.device), requires_grad=True) # Step 1: Initialize task weights
        self.L_i_0 = None  # Placeholder for the initial losses
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.warmup_step = 1

    def gradnorm(self, L_i: torch.Tensor, layer: nn.Module = None) -> torch.Tensor:
        """
        Compute the GradNorm loss.
        
        :param task_losses: A tensor of losses, one for each task.
        :return: The GradNorm loss.
        """

        if layer is None:
            layer = self.layer
        
        assert layer is not None and isinstance(layer, nn.Module), "Must provide a layer to compute the GradNorm loss."
        
        # Step 2: Save the initial losses for each task if not already saved
        if self.L_i_0 is None:
            self.L_i_0 = L_i.detach()

        # Step 3: Compute gradient norms for each task and the average gradient norm
        G_W_i = torch.stack([
            torch.autograd.grad(L_i[i] * self.w_i[i], layer.parameters(), retain_graph=True,
                                create_graph=True)[0].norm()
            for i in range(self.T)])
        G_W_bar = torch.mean(G_W_i)

        # Step 4: Compute relative inverse training rates r_i(t)
        tilde_L_i = L_i / self.L_i_0
        r_i = tilde_L_i / torch.mean(tilde_L_i)

        # Step 5: Calculate the GradNorm loss L_grad
        target_G_W_i = (G_W_bar * torch.pow(r_i, self.alpha)).detach()
        L_grad = F.l1_loss(G_W_i, target_G_W_i)

        return L_grad

    def apply_grads(self, L_grad: torch.Tensor, lr: float = None) -> torch.Tensor:
        """
        Apply the gradients from the GradNorm loss and the total loss.
        
        :param optimizer: The optimizer for the model parameters.
        :param lr: Optional learning rate for updating task weights.
        :return: The updated task weights.
        """

        if lr is None:
            lr = self.lr

            if self.lr_warmup is not None:
                lr = lr * min(1., float(self.warmup_step) / self.lr_warmup)
                self.warmup_step += 1

        assert lr is not None, "Must provide a learning rate to apply_grads."

        # Step 6: Differentiate L_grad with respect to task weights w_i and update
        self.w_i.grad = torch.autograd.grad(L_grad, self.w_i)[0]
        self.w_i.data -= lr * self.w_i.grad

        # # Step 7: Renormalize task weights w_i
        self.w_i.data = self.w_i / torch.sum(self.w_i) * self.T

        if torch.any(self.w_i < 0):
            print("Negative w_i values detected. Consider reducing the gradnorm learning rate.")
            self.w_i.data = torch.clamp(self.w_i.data, min=1e-8)

        return self.w_i
    


from collections import defaultdict
import typing as tp

import torch
from torch import autograd
from torch.nn import Module
import torch.distributed as dist



def averager(beta: float = 1):
    """
    Exponential Moving Average callback.
    Returns a single function that can be called to repeatedly update the EMA
    with a dict of metrics. The callback will return
    the new averaged dict of metrics.

    Note that for `beta=1`, this is just plain averaging.
    """
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: tp.Dict[str, tp.Any], weight: float = 1) -> tp.Dict[str, float]:
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}
    return _update


import torch
import torch.autograd as autograd
from torch.nn import Module

class Balancer(Module):
    """Loss balancer for single-GPU training.

    The loss balancer combines losses together to compute gradients for the backward.
    A call to the balancer will weight the losses according the specified weight coefficients.
    A call to the backward method of the balancer will compute the gradients, combining all the losses and
    potentially rescaling the gradients, which can help stabilize the training and reasonate
    about multiple losses with varying scales.

    Args:
        weights (Dict[str, float]): Weight coefficient for each loss.
        rescale_grads (bool): Whether to rescale gradients or not. If False, this is just
            a regular weighted sum of losses.
        total_norm (float): Reference norm when rescaling gradients, ignored otherwise.
        ema_decay (float): EMA decay for averaging the norms when `rescale_grads` is True.
        per_batch_item (bool): Whether to compute the averaged norm per batch item or not.
        epsilon (float): Epsilon value for numerical stability.
        monitor (bool): Whether to store additional ratio for each loss key in metrics.
    """

    def __init__(self, weights, rescale_grads=True, total_norm=1., ema_decay=0.999, per_batch_item=True, epsilon=1e-12, monitor=False):
        super(Balancer, self).__init__()
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm
        self.averager = averager(ema_decay)
        self.epsilon = epsilon
        self.monitor = monitor
        self.rescale_grads = rescale_grads
        self._metrics = {}

        # Register a buffer for norms
        self.register_buffer('norms_buffer', torch.zeros(len(weights)))

    @property
    def metrics(self):
        return self._metrics

    def compute_scaling_factors(self, norms):
        # avg_norms = average_metrics(self.averager(norms), 1)
        avg_norms = self.averager(norms)

        total = sum(avg_norms.values())
        self._metrics = {}

        if self.monitor:
            for k, v in avg_norms.items():
                self._metrics[f'ratio_{k}'] = v / total

        total_weights = sum([self.weights[k] for k in avg_norms])
        ratios = {k: w / total_weights for k, w in self.weights.items()}

        scaling_factors = {}
        for name, avg_norm in avg_norms.items():
            if self.rescale_grads:
                scale = ratios[name] * self.total_norm / (self.epsilon + avg_norm)
                scaling_factors[name] = scale
            else:
                scaling_factors[name] = self.weights[name]

        return scaling_factors

    def backward(self, losses, layer):
        norms = {}
        for name, loss in losses.items():
            if loss is None:
                continue

            grad = autograd.grad(loss, layer.parameters(), retain_graph=True)[0]

            if self.per_batch_item:
                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims).mean()
            else:
                norm = grad.norm()
            norms[name] = norm

        scaling_factors = self.compute_scaling_factors(norms)

        # scaled_losses = {name: loss * scaling_factors[name] for name, loss in losses.items()}
        scaled_losses = {name: losses[name] * scaling_factors[name] for name in norms.keys()}
        total_loss = sum(scaled_losses.values())
        return total_loss


def averager(beta: float = 1):
    """
    Exponential Moving Average callback.
    Returns a single function that can be called to repeatedly update the EMA
    with a dict of metrics. The callback will return
    the new averaged dict of metrics.

    Note that for `beta=1`, this is just plain averaging.
    """
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: tp.Dict[str, tp.Any], weight: float = 1) -> tp.Dict[str, float]:
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}
    return _update



# Test the implementation
def test():
    from torch.nn import functional as F
    x = torch.zeros(1, requires_grad=True)
    one = torch.ones_like(x)
    loss_1 = F.l1_loss(x, one)
    loss_2 = 100 * F.l1_loss(x, -one)
    losses = {'1': loss_1, '2': loss_2}

    balancer = Balancer(weights={'1': 1, '2': 1}, rescale_grads=False)
    balancer.backward(losses, x)
    assert torch.allclose(x.grad, torch.tensor(99.)), x.grad

    # Test with rescale_grads=True
    loss_1 = F.l1_loss(x, one)
    loss_2 = 100 * F.l1_loss(x, -one)
    losses = {'1': loss_1, '2': loss_2}
    x.grad = None
    balancer = Balancer(weights={'1': 1, '2': 1}, rescale_grads=True)
    balancer.backward({'1': loss_1, '2': loss_2}, x)
    print(f"x.grad: {x.grad}")

