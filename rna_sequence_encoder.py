import torch

encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
ignore = ['U', 'N', '\x00', '']


def encode_rna(sequence: str) -> torch.Tensor:
    weird = [x for x in sequence if x not in encoding]
    if weird:
        print(f"Warning: {weird} not in encoding")
    digits = torch.Tensor([encoding[x] for x in sequence if x in encoding])
    one_hot = torch.nn.functional.one_hot(digits.to(torch.int64), num_classes=len(encoding))

    return one_hot.t()
