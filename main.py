import argparse
from pathlib import Path
from tqdm import tqdm, trange
import torch
import trainer  # Import the trainer module for training and prediction logic.
from hyper_parmas import HyperParams
from prediction_model import PredictionModel  # Import the PredictionModel class for model architecture.
from rbpselexdataset import RbpSelexDataset
from rbpcompetesequencedataset import RbpCompeteSequenceNoIntensityDataset


# Function to load training and testing data
def load_data(params: HyperParams, rna_compete_file, slx_files):
    print("loading train files")

    # Create the training dataset and DataLoader
    train_dataset = RbpSelexDataset(
        rbps_files=slx_files,  # SLX files for training data
        embedding_size=params.embedding_char_length,  # Embedding size from hyperparameters
        padded_sequence_max_legnth=params.padding_max_size  # Maximum sequence length for padding
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    print("done loading train files")

    # Create the testing dataset and DataLoader
    test_dataset = RbpCompeteSequenceNoIntensityDataset(
        sequence_file=rna_compete_file,  # RNA compete file for testing data
        k=params.embedding_char_length,  # Embedding size from hyperparameters
        padded_sequence_max_legnth=params.padding_max_size  # Maximum sequence length for padding
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True)
    print("done loading test files")

    # Return the DataLoaders for training and testing
    return train_loader, test_loader


# Main function to execute training and prediction
def main(output_file, rna_compete_file, slx_files):
    params = HyperParams()  # Initialize hyperparameters using HyperParams class
    train_loader, predict_loader = load_data(params, rna_compete_file, slx_files)  # Load data
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
        'cpu')  # Set device to GPU if available, otherwise CPU
    model = PredictionModel(params).to(device)  # Initialize the model and move it to the specified device
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=params.lr)  # Set up the optimizer (Adam) with learning rate from hyperparameters

    # Check if the model checkpoint exists and load it if possible
    path = "model.pt"
    if Path(path).exists() and len(slx_files) < 4:
        model.load_state_dict(torch.load(path, weights_only=True))

    # Training loop
    print("Start training")
    for epoch in trange(params.epochs):  # Loop over the number of epochs
        trainer.train(model, optimizer, train_loader, device, epoch, params)  # Train the model for one epoch

    # Prediction loop
    print("Start predicting")
    with open(output_file, 'w') as f:  # Open the output file for writing predictions
        predict_generator = tqdm(trainer.predict(model, predict_loader, device), total=len(predict_loader))
        for sequences, intensities in predict_generator:  # Loop over the predictions
            for intensity in intensities:  # Write each predicted intensity to the output file
                f.write(f"{intensity}\n")

    print(f"results were written to {output_file}. done")  # Notify that results have been written


# Entry point of the program
if __name__ == "__main__":
    # Set up argument parsing for command-line interface
    parser = argparse.ArgumentParser(description='Process RNA compete and SLX files.')

    parser.add_argument('output_file',
                        type=str,
                        help='Output file ')  # Argument for output file

    parser.add_argument('rna_compete_file',
                        type=str,
                        help='RNA compete file')  # Argument for RNA compete file

    parser.add_argument('slx_files',
                        type=str,
                        nargs='+',
                        help='SLX files')  # Argument for SLX files (can take multiple files)

    args = parser.parse_args()  # Parse the arguments from the command line

    # Print out the provided arguments for verification
    print(f'Output file: {args.output_file}')
    print(f'RNA compete file: {args.rna_compete_file}')
    print(f'SLX files: {args.slx_files}')

    # Call the main function with the parsed arguments
    main(
        output_file=args.output_file,
        rna_compete_file=args.rna_compete_file,
        slx_files=args.slx_files
    )
