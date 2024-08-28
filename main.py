import argparse
from pathlib import Path
from tqdm import tqdm, trange
import torch
import trainer
from hyper_parmas import HyperParams
from prediction_model import PredictionModel
from rbpselexdataset import RbpSelexDataset
from rbpcompetesequencedataset import RbpCompeteSequenceNoIntensityDataset

def load_data(params:HyperParams, rna_compete_file, slx_files):
    print("loading train files")
    train_dataset = RbpSelexDataset(
        rbps_files=slx_files,
        embedding_size=params.embedding_char_length,
        padded_sequence_max_legnth=params.padding_max_size
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    print("done loading train files")

    test_dataset = RbpCompeteSequenceNoIntensityDataset(
        sequence_file=rna_compete_file,
        k=params.embedding_char_length,
        padded_sequence_max_legnth=params.padding_max_size
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True)
    print("done loading test files")

    return train_loader, test_loader

def main(output_file, rna_compete_file, slx_files):
    params = HyperParams()
    train_loader, predict_loader = load_data(params, rna_compete_file, slx_files)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = PredictionModel(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    path = "model.pt"
    if Path(path).exists() and len(slx_files) < 4:
        model.load_state_dict(torch.load(path, weights_only=True))

    print("Start training")
    for epoch in trange(params.epochs):
        trainer.train(model, optimizer, train_loader, device, epoch, params)

    print("Start predicting")
    with open(output_file, 'w') as f:
        predict_generator = tqdm(trainer.predict(model, predict_loader, device), total=len(predict_loader))
        for sequences, intensities in predict_generator:
            for intensity in intensities:
                f.write(f"{intensity}\n")

    print(f"results were written to {output_file}. done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process RNA compete and SLX files.')

    parser.add_argument('output_file',
                        type=str,
                        help='Output file ')

    parser.add_argument('rna_compete_file',
                        type=str,
                        help='RNA compete file')

    parser.add_argument('slx_files',
                        type=str,
                        nargs='+',
                        help='SLX files')

    args = parser.parse_args()

    # Main logic to handle the parsed arguments
    print(f'Output file: {args.output_file}')
    print(f'RNA compete file: {args.rna_compete_file}')
    print(f'SLX files: {args.slx_files}')

    main(
        output_file=args.output_file,
        rna_compete_file=args.rna_compete_file,
        slx_files=args.slx_files
   )