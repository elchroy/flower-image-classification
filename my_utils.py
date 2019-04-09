# AUTHOR : ELISHA-WIGWE CHIJIOKE O.
# NAME: my_utils.py
# This file contains necessary methods that are used by other files

import torch
from torch import nn, optim
from os import path, makedirs
from collections import OrderedDict
from torchvision import datasets, transforms, models

def get_model(architecture = 'vgg13'):
    print('Fetching pretrained model ({})...'.format(architecture))
    model = getattr(models, architecture)(pretrained=True)
    print('Model fetched: ({}).'.format(architecture))
    return model

def freeze_parameters(model):
	for parameter in model.parameters():
		parameter.requires_grad = False
	print('Model parameters frozen...')


def get_num_input_features(model_classifier):
    """
    Get the number of input_features from the classifier of the ImageNet pretrained network
    """
    if isinstance(model_classifier, nn.Linear):
        return model_classifier.in_features
    return [ c for c in model_classifier if isinstance(c, nn.Linear) ][0].in_features

def get_criterion():
	return nn.NLLLoss()

def get_optimizer(_model, _lr=0.001):
	return optim.Adam(_model.classifier.parameters(), lr=_lr)

def check_data_validity(_training_args):
	if _training_args.data_dir == None:
		return False, "Missing --data_dir: Dataset directory missing"
	if not path.exists(_training_args.data_dir):
		return False, "Unavailable --data_dir: Directory ({}) not found".format(_training_args.data_dir)
	return True, ""

def process_training_args(_input_args):
	print("Hidden Units: {}".format(" ".join(map(str, _input_args.hidden_units))))



def get_data_transforms():
    normalize_tranform = transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
    rand_flip_v_transform = transforms.RandomVerticalFlip()
    rand_flip_h_transform = transforms.RandomHorizontalFlip()
    center_crop_transform = transforms.CenterCrop(224)
    resize_transform = transforms.Resize(225)
    rotation_transform = transforms.RandomRotation(180)
    tensor_transform = transforms.ToTensor()

    return {
        'train_transforms': transforms.Compose([
            resize_transform,
            center_crop_transform,
            rand_flip_h_transform,
            rand_flip_v_transform,
            rotation_transform,
            tensor_transform,
            normalize_tranform
        ]),
        'test_val_transforms': transforms.Compose([
            resize_transform,
            center_crop_transform,
            rotation_transform,
            tensor_transform,
            normalize_tranform
        ])
    }

def for_alexnet(n_in, n_hid, dropout):
    return [ nn.Dropout(dropout), nn.Linear(n_in, n_hid), nn.ReLU() ]

def for_vgg_densenet(n_in, n_hid, dropout):
    return [ nn.Linear(n_in, n_hid), nn.ReLU(), nn.Dropout(dropout) ]

def build_classifier(architecture, n_in, hidden_units, n_out, dropout=0.5):
    # layer_builder for the chosen architecture
    build_layer = for_alexnet if architecture[:7] == 'alexnet' else for_vgg_densenet
	
    # 1st hidden layer 
    hidden_layers = build_layer(n_in, hidden_units[0], dropout)

    for hl1, hl2 in zip(hidden_units[:-1], hidden_units[1:]):
        hidden_layers += build_layer(hl1, hl2, dropout)

    hidden_layers += [ nn.Linear(hidden_units[-1], n_out), nn.LogSoftmax(dim=1) ]
    hidden_layers = [ ("fc{}".format(i), layer) for i, layer in enumerate(hidden_layers) ] # convert each to a tuple

    _classifier = nn.Sequential(OrderedDict(hidden_layers))
    print('Classifier built...')
    return _classifier

def connect_classifier_to_model(_model, _classifier, _model_state_dict=None):
	_model.classifier = _classifier
	print('Classifier added to model...')

	if _model_state_dict != None:
		_model.load_state_dict(_model_state_dict)
		print('Loaded model\'s state dict...')

def validation (_model, _loader, _criterion, _device):
    loss = 0
    accuracy = 0
    total = 0
    correct = 0
    
    for images, labels in _loader:
        images, labels = images.to(_device), labels.to(_device)
        output = _model.forward(images)
        loss += _criterion(output, labels).item()

        _, pred = torch.max(output.data, 1)
        total += labels.size()[0]
        correct += (pred == labels).sum().item()
    accuracy = (100*correct)/total
    return loss, accuracy

def evaluate_test_data(_model, _test_loader):    
    correct = 0
    total = 0
    device = get_device()
    _model = _model.to(device)
    with torch.no_grad():
        for images, labels in _test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = _model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy: {}%".format(100*correct/total))


def check_for_file(filepath):
    return path.isfile(filepath)

def save_checkpoint(architecture, n_in_features, n_out_features, hidden_units, model_state_dict, optimizer_state_dict, epochs, class_to_idx, checkpoint_path='checkpoints_dir'):
	_checkpoint = {
		'architecture': architecture,
		'n_in_features': n_in_features,
		'n_out_features': n_out_features,
		'hidden_units': hidden_units,
		'model_state_dict': model_state_dict,
		'optimizer_state_dict': optimizer_state_dict,
		'epochs': epochs,
		'class_to_idx': class_to_idx
	}

	if not path.exists(checkpoint_path):
		makedirs('./' + checkpoint_path)
	save_to = path.join(checkpoint_path, '__checkpoint_{}.pth'.format(architecture))
	torch.save(_checkpoint, save_to)
	print('Checkpoint saved...')

def get_device(_gpu=False):
    print("Fetching device...")
    device = torch.device("cuda:0" if _gpu and torch.cuda.is_available() else 'cpu')
    print("Device {} fetched!".format(device))
    return device

def load_checkpoint(filepath):
    _checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    print("Checkpoint loaded...")
    return _checkpoint


def get_image_datasets(data_transforms, train_dir, valid_dir, test_dir):
    return {
        'train_data': datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
        'valid_data': datasets.ImageFolder(valid_dir, transform=data_transforms['test_val_transforms']),
        'test_data': datasets.ImageFolder(test_dir, transform=data_transforms['test_val_transforms']),
    }

def get_data_loaders(image_datasets):
    return {
        'train_loader': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
        'test_loader': torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=64, shuffle=True),
        'valid_loader': torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64, shuffle=True)
    }

def get_dirs(data_dir):
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'
	return train_dir, valid_dir, test_dir