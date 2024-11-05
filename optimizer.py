from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.to(device)
                
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p] # this is the original code
                
                if(len(state) == 0):
                    state = (torch.zeros(p.shape), torch.zeros(p.shape), 0)

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # To complete this implementation:
                # 1. Update the first and second moments of the gradients.
                # 2. Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3. Update parameters (p.data).
                # 4. Apply weight decay after the main gradient-based updates.
                # Refer to the default project handout for more details.

                betas = group['betas']

                m = state[0].to(device)
                v = state[1].to(device)
                t = state[2]
                theta = p.to(device)

                t += 1

                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * (grad * grad)
                

                alpha = alpha * (1 - betas[1]**t)**0.5 / ((1 - betas[0]**t))
                theta = theta - alpha*m/(torch.sqrt(v) + group['eps'])

                theta -= group["lr"] * group['weight_decay'] * theta

                p.data = theta

                self.state[p] = (m, v, t)

                # p.data = theta
                ### TODO
                # raise NotImplementedError

        # print(self.param_groups)
        return loss