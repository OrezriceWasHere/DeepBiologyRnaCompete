import torch

encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}


def encode_rna(sequence: str) -> torch.Tensor:
    digits = torch.Tensor([encoding[x] for x in sequence if x in encoding])
    one_hot = torch.nn.functional.one_hot(digits.to(torch.int64), num_classes=len(encoding))

    return one_hot.t()
