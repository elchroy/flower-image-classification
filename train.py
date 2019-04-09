# AUTHOR : ELISHA-WIGWE CHIJIOKE O.
# NAME: train.py
# This file trains a builds and trains a model, before saving the parameters to a checkpoint

import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F

from workspace_utils import active_session
from my_utils import validation, get_criterion, process_training_args, check_data_validity, get_optimizer, build_classifier, save_checkpoint, get_device, connect_classifier_to_model, get_num_input_features, freeze_parameters, get_model, get_data_transforms, get_image_datasets, get_data_loaders, get_dirs

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", nargs='?', default="checkpoints_dir", type=str, help="The directory to save checkpoints") # optional - default to checkpoints_dir
    parser.add_argument("--learning_rate", nargs='?', default=0.001, type=float, help="The learning rate for the model training")
    parser.add_argument("--hidden_units", nargs='+', type=int, default=512, help="The number of units in each hidden layer of the model network")
    parser.add_argument("--epochs", nargs='?', default=20, type=int, help="The number of epochs desired for training")
    parser.add_argument("--check_every", nargs='?', default=30, type=int, help="The epochs for each check with validation data")
    parser.add_argument("--gpu", default=True, type=bool, help="Train the model on a GPU device if available")
    parser.add_argument("--architecture", nargs='?', default="vgg13", type=str, help="The architecture of the model") # optional - default to vgg13
    parser.add_argument("data_dir", nargs='?', default=None, type=str, help="The directory containing the dataset") # required

    return parser.parse_args()

def train (model, train_loader, valid_loader, epochs, check_at, criterion, optimizer, device):
    print("Training started; training for {} epochs...".format(epochs))
    steps = 0
    
    # move the operation/executive to the available device/machine
    model.to(device)
    print("Running on {}".format(device))

    for e in range(epochs):
        running_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            steps += 1
            
            images, labels = images.to(device), labels.to(device)
            images.resize_(images.size()[0], 3, 224, 224)
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if steps % check_at == 0:
                model.eval()
                print("\nEpoch: {}/{}...  ".format(e+1, epochs),
                    "Training Loss: {:.4f}".format(running_loss/check_at))
                
                with torch.no_grad():
                    valid_loss, valid_accuracy = validation(
                        _model=model,
                        _loader=valid_loader,
                        _criterion=criterion,
                        _device=device
                    )

                print("Validation Loss: {:.4f}...   ".format(valid_loss),
                      "Validation Accuracy: {:.4f}".format(valid_accuracy))

                running_loss = 0

                model.train()
    
    print("Training finished; trained for {} epochs.".format(epochs))


def main():
    input_args = get_input_args()
    
    valid_data_dir, msg = check_data_validity(input_args)
    if not valid_data_dir: print(msg); return

    train_dir, valid_dir, test_dir = get_dirs(input_args.data_dir)

    process_training_args(input_args)

    data_transforms = get_data_transforms()

    image_datasets = get_image_datasets(data_transforms, train_dir, valid_dir, test_dir)

    data_loaders = get_data_loaders(image_datasets)

    model = get_model(input_args.architecture)
    freeze_parameters(model)

    n_in_features = get_num_input_features(model.classifier)
    n_out_features = len(image_datasets['test_data'].class_to_idx)
    hidden_units = input_args.hidden_units
    architecture = input_args.architecture
    
    # Build classifier from checkpoint values
    new_classifier = build_classifier(
        architecture=architecture,
        n_in=n_in_features,
        hidden_units=hidden_units,
        n_out=n_out_features,
        dropout=0.5
    )

    # Connect this classifier to the model.
    connect_classifier_to_model(
        _model=model,
        _classifier=new_classifier
    )

    criterion = get_criterion()
    optimizer = get_optimizer(_model=model, _lr=input_args.learning_rate)
    device = get_device(input_args.gpu)

    epochs = input_args.epochs

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
        architecture=input_args.architecture,
        n_in_features=n_in_features,
        n_out_features=n_out_features,
        hidden_units=hidden_units,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epochs=input_args.epochs,
        class_to_idx=image_datasets['test_data'].class_to_idx,
        checkpoint_path=input_args.save_dir,
    )




if __name__ == "__main__":
    main()



