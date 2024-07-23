import torch

rna_encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
dna_encoding = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': 4}


def encode(sequence: str, mapping: dict) -> torch.Tensor:

    digits = torch.tensor([mapping[x] for x in sequence])
    one_hot = torch.nn.functional.one_hot(digits.to(torch.int64), num_classes=len(mapping))

    return one_hot


def encode_dna(sequence: str) -> torch.Tensor:
    """temporal fix to handle changed data source"""
    return encode_rna(sequence)


def encode_rna(sequence: str) -> torch.Tensor:
    return encode(sequence, rna_encoding)
