import torch

# Dictionaries for encoding DNA and RNA sequences into numerical representations.
dna_encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'X': 4, 'N': 5}  # Encoding for DNA.
rna_encoding = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'X': 4, 'N': 5}  # Encoding for RNA.

# A list to specify any characters to be ignored during encoding.
ignore = []

# Maximum length of the sequence to ensure uniform input size across batches.
max_len = 41


# Function to encode a sequence into a one-hot encoded tensor.
def encode(sequence: str, mapping: dict) -> torch.Tensor:
    # Pad the sequence with "X" characters to make it the same length as max_len.
    padded_seq = sequence.ljust(max_len, "X")
    # Convert the sequence characters into numerical digits based on the mapping dictionary.
    digits = torch.Tensor([mapping[x] for x in padded_seq if x not in ignore])
    # Convert the digit sequence into a one-hot encoded tensor.
    one_hot = torch.nn.functional.one_hot(digits.to(torch.int64), num_classes=len(mapping))

    return one_hot


# Function to encode a DNA sequence using the dna_encoding dictionary.
def encode_dna(sequence: str) -> torch.Tensor:
    return encode(sequence, dna_encoding)


# Function to encode an RNA sequence, which internally uses the DNA encoding function.
def encode_rna(sequence: str) -> torch.Tensor:
    return encode_dna(sequence)  # Note: This uses the DNA encoding function directly, so consider renaming for clarity.


# Utility function to stack a batch of data and targets into a single tensor.
def stack_batch(batch):
    # Extract the input data (sequences) from the batch.
    data = [item[0] for item in batch]
    # Extract the target labels from the batch.
    target = [item[1] for item in batch]

    # Stack the list of tensors (data and target) into a single tensor for batch processing.
    data = torch.stack(data, dim=0)
    target = torch.stack(target, dim=0)

    return data, target
