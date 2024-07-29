import torch
dna_encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'X': 4, 'N': 5}
rna_encoding = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'X': 4, 'N': 5}
ignore = []
max_len = 41


def encode(sequence: str, mapping: dict) -> tuple[torch.Tensor, torch.Tensor]:
    padded_seq = sequence.ljust(max_len, "X")
    length = len(sequence)
    digits = torch.Tensor([mapping[x] for x in padded_seq if x not in ignore])
    one_hot = torch.nn.functional.one_hot(digits.to(torch.int64), num_classes=len(mapping))
    tensor_length = torch.Tensor([length]).long()
    return one_hot, tensor_length


def encode_dna(sequence: str) -> torch.Tensor:
    return encode(sequence, dna_encoding)


def encode_rna(sequence: str) -> torch.Tensor:
    return encode_dna(sequence)


def stack_batch(batch):
    data = [item[0] for item in batch]
    length = [item[1] for item in batch]
    target = [item[2] for item in batch]

    data = torch.stack(data, dim=0)
    length = torch.stack(length, dim=0)
    target = torch.stack(target, dim=0)

    return data, length, target
