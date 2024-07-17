import torch

dna_encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
rna_encoding = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
ignore = ['N']


def encode(sequence: str, mapping: dict) -> torch.Tensor:

    digits = torch.Tensor([mapping[x] for x in sequence if x not in ignore])
    one_hot = torch.nn.functional.one_hot(digits.to(torch.int64), num_classes=len(mapping))

    return one_hot.t()


def encode_dna(sequence: str) -> torch.Tensor:
    return encode(sequence, dna_encoding)


def encode_rna(sequence: str) -> torch.Tensor:
    return encode(sequence, rna_encoding)
