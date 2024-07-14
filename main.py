import argparse
import torch
from prediction_model import PredictionModel
from hyper_parmas import HyperParams
from clearml_poc import clearml_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(sequence_file, htr_selex_files, rna_compete_intensities):
    params = HyperParams
    clearml_init(params)

    max_len = 41
    input = torch.randint(params.one_hot_size, (params.batch_size, max_len)).to(device)
    input = torch.nn.functional.one_hot(input).float()
    input = input.permute(0, 2, 1)

    model = PredictionModel(params).to(device)
    y = model(input)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('rna_compete_sequences',
                        default="./data/RNAcompete_sequences.txt",
                        nargs='?',
                        type=str,
                        help='sequences file')

    parser.add_argument('htr_selex_files',
                        default=['./data/RBP1_1.txt',
                                 './data/RBP1_2.txt',
                                 './data/RBP1_3.txt',
                                 './data/RBP1_4.txt', ],
                        nargs='*',
                        help='htr selex files')
    default_rna_compete_intensities = './data/RBP1.txt'

    args = parser.parse_args()

    main(args.rna_compete_sequences,
         args.htr_selex_files,
         default_rna_compete_intensities
         )
