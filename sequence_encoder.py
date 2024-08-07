import math
from itertools import permutations, combinations, product

import torch

dna_encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
rna_encoding = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
ignore = ['N']
max_len = 41


def all_possible_encodings(k: int, alphabet: list[str]) -> dict:
    possible_kmers = list(product(alphabet, repeat=k)) + alphabet + ["X"]
    return {"".join(value): index for index, value in enumerate(possible_kmers)}


def encode(sequence: str, mapping: dict) -> tuple[torch.Tensor, torch.Tensor]:
    padded_seq = sequence.ljust(max_len, "X")
    length = len(sequence)
    digits = torch.Tensor([mapping[x] for x in padded_seq if x not in ignore])
    one_hot = torch.nn.functional.one_hot(digits.to(torch.int64), num_classes=len(mapping))
    tensor_length = torch.Tensor([length]).long()
    return one_hot, tensor_length


def encode_embedding(sequence, mapping, k, padded_sequence_max_legnth):
    for delete_char in ignore:
        sequence = sequence.replace(delete_char, "")
    k_length_pieces = math.floor(len(sequence) / k)
    one_length_pieces = len(sequence) % k
    empty_pieces = padded_sequence_max_legnth - k_length_pieces - one_length_pieces
    original_length = k_length_pieces + one_length_pieces
    k_length_encodings = [mapping[sequence[i * k: (i + 1) * k]]
                          for i
                          in range(k_length_pieces)]
    one_length_encodings = [
        mapping[sequence[k * k_length_pieces + i]]
        for i in range(one_length_pieces)
    ]
    empty_encoding = [mapping["X"] for _ in range(empty_pieces)]
    combined_encoding = k_length_encodings + one_length_encodings + empty_encoding

    return torch.tensor(combined_encoding, dtype=torch.long), torch.Tensor([original_length]).long()


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

    pair = (data, length, target)

    return pair
