import torch
from torch import nn
from einops import rearrange

class LinearMoE(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: any = None, dtype: any = None,  num_experts: int = 1000, r=4, top_k: int = 4):
        super(LinearMoE, self).__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.num_experts = num_experts
        self.top_k = top_k
        self.r = r

        self.gate = nn.Linear(in_features, r, bias=False)
        self.experts = nn.Parameter(torch.ones(num_experts), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(num_experts, in_features), requires_grad=True)

        self.reset_parameters()

    def forward(self, x):
        x = x.unsqueeze(-2)

        gate = self.gate(x).softmax(dim=-1).squeeze(-1)
        experts = rearrange(self.experts, 'e -> 1 1 e 1')

        x = x * (1 + (experts * gate).mean(-1, keepdim=True))]
        x = x + self.bias2.unsqueeze(0)

        # Optimize top-k selection and gathering
        _, indices = torch.topk(x, self.top_k, dim=-2, sorted=False)
        x = torch.gather(x, -2, indices).mean(dim=-2)

        x = super().forward(x)

        return x

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
