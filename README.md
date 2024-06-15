# LinearMoE PyTorch implementation
This repo contains my custom implementation of a mixture of experts as an extension of the linear layer.

## Usage
1. Use instead of linear layers during each training phase.
2. Use instead of LoRA when doing further fine-tuning. Froze other layers, unfreeze only gate, experts, and bias2 from the LinearMoE implementation.
