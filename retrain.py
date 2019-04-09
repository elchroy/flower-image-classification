# AUTHOR : ELISHA-WIGWE CHIJIOKE O.
# NAME: retrain.py:
# This file is used to retrain the model
# It first loads the saved checkpoint, trains the model, and updates/resaves the checkpoint
import argparse
from workspace_utils import active_session
from train import train
from my_utils import save_checkpoint, get_device, get_criterion, get_optimizer, build_classifier, connect_classifier_to_model, load_checkpoint, get_dirs, get_data_transforms, get_image_datasets, get_data_loaders, get_model, freeze_parameters

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", default=True, type=bool, help="If available, Use the GPU device to predict the probabilities")
    parser.add_argument("--learning_rate", default=0.0009, type=float, help="The learning rate for the model training")
    parser.add_argument("--save_dir", default="checkpoints_dir", type=str, help="The directory to save checkpoints") # optional - default to checkpoints_dir
    parser.add_argument("--epochs", default=20, type=int, help="The number of epochs desired for training")
    parser.add_argument("--check_every", default=30, type=int, help="The epochs for each check with validation data")
    parser.add_argument("data_dir", type=str, help="The directory containing the dataset") # required
    parser.add_argument("checkpoint", type=str, help="The path to checkpoint saved from training") # required

    return parser.parse_args()

def main():
    # Handle case where requried arguments are missing
    input_args = get_input_args()

    train_dir, valid_dir, test_dir = get_dirs(input_args.data_dir)

    data_transforms = get_data_transforms()

    image_datasets = get_image_datasets(data_transforms, train_dir, valid_dir, test_dir)

    data_loaders = get_data_loaders(image_datasets)

    # load checkpoint
    checkpoint = load_checkpoint(input_args.checkpoint)

    # Load model and freeze model parameters
    model = get_model(checkpoint['architecture'])
    freeze_parameters(model)
    
    n_in_features = checkpoint['n_in_features']
    n_out_features = checkpoint['n_out_features']
    hidden_units = checkpoint['hidden_units']
    architecture = checkpoint['architecture']
    
    # Build classifier from checkpoint values
    classifier = build_classifier(
        architecture=architecture,
        n_in=n_in_features,
        hidden_units=hidden_units,
        n_out=n_out_features,
        dropout=0.5
    )

    # Connect this classifier to the model.
    connect_classifier_to_model(
        _model=model,
        _classifier=classifier,
        _model_state_dict=checkpoint['model_state_dict']
    )
    
    # Load optimizer and criterion
    criterion = get_criterion()
    optimizer = get_optimizer(_model=model, _lr=input_args.learning_rate)
    device = get_device(input_args.gpu)

    # Continue training the network
    with active_session():
	    train(
	        model=model,
	        train_loader=data_loaders['train_loader'],
	        valid_loader=data_loaders['valid_loader'],
	        epochs=input_args.epochs,
	        check_at=input_args.check_every,
	        criterion=criterion,
	        optimizer=optimizer,
	        device=device
	    )

    # Save the checkpoint 
    save_checkpoint(
        architecture=checkpoint['architecture'],
        n_in_features=n_in_features,
        n_out_features=n_out_features,
        hidden_units=hidden_units,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epochs=input_args.epochs,
        class_to_idx=checkpoint['class_to_idx'],
        checkpoint_path=input_args.save_dir,
    )





if __name__ == "__main__":
    main()



