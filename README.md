# LinearMoE PyTorch implementation
This repo contains my custom implementation of a mixture of experts as an extension of the linear layer.

## Usage
1. Use instead of linear layers during each training phase.
2. Use instead of LoRA when doing further fine-tuning. Froze other layers, unfreeze only gate, experts, and bias2 from the LinearMoE implementation.


## Citation
If you use this codebase, or otherwise found my work valuable, please cite:

```
@inproceedings{LinearMoE,
  title={LinearMoE: Parallelized Mixture of Experts on top of the Linear Layer},
  author={Mariusz Kurman @ MedIT Solutions Kurman i Wspolnicy Sp. z o. o.},
  year={2024}
}
```
