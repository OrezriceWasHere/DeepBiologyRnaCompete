# sequence generator is is a tool to generate random rbp file.

import random


def generate_random_sequence(length=41):
    """Generate a random sequence of 'A', 'C', 'T', 'G' of given length."""
    return ''.join(random.choice('ACTG') for _ in range(length))


def generate_rbp_list(num_lines=294190) -> list[str]:
    """Generate a text file with the specified number of lines."""
    # rbps = []
    # with open(filename, 'w') as f:
    for _ in range(num_lines):
        random_sequence = generate_random_sequence()
        yield random_sequence
        # rbps.append(random_sequence)
            # line = f"{random_sequence},1\n"
            # f.write(line)

    # return rbps


if __name__ == '__main__':

    # Specify the output file name
    output_filename = 'random_sequences.txt'

    # Generate the text file
    generate_rbp_file(output_filename)

    print(f"File '{output_filename}' has been generated.")
