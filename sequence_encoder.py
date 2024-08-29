import math
from itertools import permutations, combinations, product
import torch

# Define encoding mappings for DNA and RNA sequences.
dna_encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
rna_encoding = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
ignore = ['N']  # Define characters to ignore during encoding.
max_len = 41  # Define the maximum length for sequences, used for padding.




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
    try:
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

    except KeyError as e:
        print(f"Error: {e}")
        raise e


# Function to encode a DNA sequence.
def encode_dna(sequence: str) -> torch.Tensor:
    return encode(sequence, dna_encoding)


# Function to encode an RNA sequence (uses DNA encoding logic).
def encode_rna(sequence: str) -> torch.Tensor:
    return encode_dna(sequence)





# Function to stack a batch of sequences, lengths, and targets into tensors.
def stack_batch(batch):
    # Extract data, lengths, and targets from the batch.
    data = [item[0] for item in batch]
    length = [item[1] for item in batch]
    target = [item[2] for item in batch]

    # Stack the data, lengths, and targets along a new dimension.
    data = torch.stack(data, dim=0)
    length = torch.stack(length, dim=0)
    target = torch.stack(target, dim=0)

    # Return the stacked tensors as a tuple.
    pair = (data, length, target)

    return pair
