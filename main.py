import argparse

def main(sequence_file, htr_selex_files, rna_compete_intensities):
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rna_compete_sequences",
                        type="str",
                        help="path to RNA compete sequences",
                        default="./data/RNAcompete_sequences.txt")

    parser.add_argument('htr_selex',
                        metavar='N',
                        type=str,
                        nargs='+',
                        default=['./data/RBP1_1.txt',
                                 './data/RBP1_2.txt',
                                 './data/RBP1_3.txt',
                                 './data/RBP1_4.txt', ],
                        help='a list of strings')

    parser.add_argument("-rna",
                        "--rna_intensities",
                        action='store_true',
                        dest='rna_compete_intensities',
                        default="./data/RBP1.txt",
                        help='Enabling debugging.')

    parser.parse_args()
    main(parser.rna_compete_sequences, parser.htr_selex, parser.rna_compete_intensities)
