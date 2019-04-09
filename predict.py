# AUTHOR : ELISHA-WIGWE CHIJIOKE O.
# NAME: predict.py
# This file is used to predict the class/probability of an image (flower)

import json
import argparse
import torch
from os import path, listdir
from PIL import Image as IM
from torchvision import transforms
from my_utils import build_classifier, get_device, connect_classifier_to_model, get_model, freeze_parameters, load_checkpoint

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", default=5, type=int, help="The highest probabilities to return, default is 3") # optional - default to checkpoints_dir
    parser.add_argument("--category_names", type=str, help="The file.json containing category to name mapping")
    parser.add_argument("--gpu", default=False, type=bool, help="If available, Use the GPU device to predict the probabilities")
    parser.add_argument("--path_to_dir", type=str, help="The path to directory of images for batch-predicting.") # required
    # parser.add_argument("path_to_image", type=str, help="The path to image for which the category is being predicted") # required
    parser.add_argument("checkpoint", type=str, help="The path to checkpoint saved from training") # required

    return parser.parse_args()


def main():
    # Handle case where requried arguments are missing
    input_args = get_input_args()

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

    if input_args.category_names:
        with open(input_args.category_names, 'r') as f:
            categories = json.load(f)
    else:
        categories = None

    if input_args.path_to_dir:
        imgs = listdir(input_args.path_to_dir)
        for img in imgs:
            # print("\nPredicting image ({}) ...".format(img))
            path_to_image = "{0}/{1}".format(input_args.path_to_dir, img)
            # print(path_to_image, 'imgs')
            c, p = batch_predict(
                image_path=path_to_image,
                model=model,
                categories=categories,
                topk=1,
                class_to_idx=checkpoint['class_to_idx']
            )
            print("{} - {}".format(path_to_image, c[0]))


    # print("In descending order of probabilities, here are the top {} predictions:".format(input_args.top_k))
    # for c, p in zip(classes, probs):
    #     print("\nFLOWER NAME/INDEX: {0}\nPROBABILITY: {1}".format(c, p))


def batch_predict(image_path, model, categories=None, topk=5, class_to_idx=None, gpu=False):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # device = get_device(gpu)
    # model = model.to(device)

    image = IM.open(image_path)
    
    # make a tensor transformer and a resize transformer
    tensor_transform = transforms.ToTensor()
    resize_transform = transforms.Resize([224, 224])
    
    transformed_image = tensor_transform(resize_transform(image)) # and apply them to the PIL image
    
    with torch.no_grad():
        transformed_image = transformed_image.unsqueeze(0)
        # transformed_image = transformed_image.to(device)
        logits = model.forward(transformed_image)
    
    ps = torch.exp(logits) # because of the NLLLoss()
    ps_topk = ps.topk(topk, sorted=True) # this has the highest prob at index 0, hence the [0], and the index of each probability at index 1
    ps_topk_probs = ps_topk[0] # this is the highest prob at index 0
    ps_topk_index = ps_topk[1] # this is the index of each prob at index 1
    class_to_idx = class_to_idx # this should have been saved as part of the checkpoint.
    idx_to_class = { i: c for c, i in class_to_idx.items() if i in ps_topk_index.cpu().numpy() } # Only take the indices that are part of the topk probabilities.

    classes = list(idx_to_class.values())
    probs = ps_topk_probs.cpu().numpy().tolist()[0]

    if not categories == None:
        names = [ categories[k] for k in categories if k in classes ]
        return names, probs
    return classes, probs


if __name__ == "__main__":
    main()


def predict(image_path, model, categories=None, topk=5, class_to_idx=None, gpu=False):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''

    device = get_device(gpu)
    model = model.to(device)

    image = IM.open(image_path)
    
    # make a tensor transformer and a resize transformer
    tensor_transform = transforms.ToTensor()
    resize_transform = transforms.Resize([224, 224])
    
    transformed_image = tensor_transform(resize_transform(image)) # and apply them to the PIL image
    
    with torch.no_grad():
        transformed_image = transformed_image.unsqueeze(0)
        transformed_image = transformed_image.to(device)
        logits = model.forward(transformed_image)
    
    ps = torch.exp(logits) # because of the NLLLoss()
    ps_topk = ps.topk(topk, sorted=True) # this has the highest prob at index 0, hence the [0], and the index of each probability at index 1
    ps_topk_probs = ps_topk[0] # this is the highest prob at index 0
    ps_topk_index = ps_topk[1] # this is the index of each prob at index 1
    class_to_idx = class_to_idx # this should have been saved as part of the checkpoint.
    idx_to_class = { i: c for c, i in class_to_idx.items() if i in ps_topk_index.cpu().numpy() } # Only take the indices that are part of the topk probabilities.

    classes = list(idx_to_class.values())
    probs = ps_topk_probs.cpu().numpy().tolist()[0]

    if not categories == None:
        names = [ categories[k] for k in categories if k in classes ]
        return names, probs
    return classes, probs


if __name__ == "__main__":
    main()



