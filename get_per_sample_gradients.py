import typing as tp

import torch
from torch import nn, Tensor


class GhostGradModel(nn.Module):
    """reference: https://github.com/Jiachen-T-Wang/GREATS/blob/master/less/train/utils_ghost_dot_prod.py#L27-L146"""
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model
        self.activations: dict[str, Tensor] = {}
        self.gradients: dict[str, Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        # Copilot: better way to save activations
        for name, layer in self.model.named_children():
            if any(p.requires_grad for p in layer.parameters()):
                layer.__name__ = name
                # will be called every time after forward has computed an output.
                layer.register_forward_hook(self._save_activations_hook)
                layer.register_full_backward_hook(self._save_gradients_hook)

    def _save_activations_hook(self, module, input, output):
        self.activations[module.__name__] = input[0].detach()

    def _save_gradients_hook(self, module, grad_input, grad_output):
        self.gradients[module.__name__] = grad_output[0].detach()

    def get_per_sample_grads(self, x:Tensor, y:Tensor, loss_fn):
        batch_size = x.shape[0]
        self.activations.clear()
        self.gradients.clear()

        outputs = self.model(x)
        loss_per_sample = loss_fn(outputs, y)  # (batch_size,)
        loss = loss_per_sample.sum() # (,)

        self.model.zero_grad()
        loss.backward()

        per_sample_grads = []
        for name, layer in self.model.named_children():
            if not any(p.requires_grad for p in layer.parameters()):
                continue

            act = self.activations[name]    # (B, in_dim )
            out_grad = self.gradients[name] # (B, out_dim)

            # broadcast and multiplication
            # (B, in_dim, 1) * (B, 1, out_dim) = (B, in_dim, out_dim)
            weight_grads = act.unsqueeze(2) * out_grad.unsqueeze(1)  # (B, in_dim, out_dim)
            weight_grads = weight_grads.transpose(1, 2)       # (B, out_dim, in_dim)
            weight_grads = weight_grads.reshape(batch_size, -1) # (B, out_dim * in_dim)

            bias_grads = out_grad

            # Combine
            layer_grads = torch.cat([weight_grads, bias_grads], dim=1)
            per_sample_grads.append(layer_grads)

        return torch.cat(per_sample_grads, dim=1)


def manual_per_sample_gradients(model:nn.ModuleDict, x:Tensor, y:Tensor, loss_fn:tp.Callable):
    batch_size = x.shape[0]
    manual_grads = []

    for i in range(batch_size):
        model.zero_grad()
        x_i = x[i].unsqueeze(0)
        y_i = y[i].unsqueeze(0)
        loss:Tensor = loss_fn(model(x_i), y_i)
        loss.backward()

        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        manual_grads.append(torch.cat(grads))

    return torch.stack(manual_grads)


if __name__ == "__main__":
    x = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Linear(10, 2),
    )

    ghost_model = GhostGradModel(model)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # Compute gradients
    ghost_grads = ghost_model.get_per_sample_grads(x, y, loss_fn)
    manual_grads = manual_per_sample_gradients(model, x, y, loss_fn)

    print("Ghost gradients shape:", ghost_grads.shape)
    print("Manual gradients shape:", manual_grads.shape)
    print("Ghost gradients:", ghost_grads, end="\n\n")
    print("Manual gradients:", manual_grads, end="\n\n")

    # Check numerical equivalence
    print("Max difference:", (ghost_grads - manual_grads).abs().max().item())
    assert torch.allclose(ghost_grads, manual_grads, atol=1e-4)
    print("Done")