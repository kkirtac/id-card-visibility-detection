"""Prediction script. Runs id card visibility prediciton with given resnet18 model."""

import sys
sys.path.append('.')
from argparse import ArgumentParser
import pytorch_lightning as pl
import cv2
from albumentations.pytorch.transforms import ToTensorV2
from modules.model import ResnetLightning
from modules.transforms import TransformationsBase

INT_LABEL_MAPPING = {
    0: 'FULL_VISIBILITY',
    1: 'PARTIAL_VISIBILITY',
    2: 'NO_VISIBILITY',
}

# we are using the pixel mean std of ImageNet training set
# because our trained model used these values for image normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main(model_path: str, image_path: str):
    """Run prediction of the model with a single image.
    Args:
        model_path (str): path to the model weights artifact.
        image_path (str): path to the query image.
    """
    # load model
    model = ResnetLightning.load_from_checkpoint(model_path).eval()

    # repeat the blue channel on green and red channels to receive a 3-channel image
    # because our CNN model requires 3-channel input
    image = cv2.imread(image_path)
    image[:, :, 1] = image[:, :, 0]
    image[:, :, 2] = image[:, :, 0]

    # we convert from BGR to RGB because our model is trained so.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = TransformationsBase(scale_size=224, norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD)

    # scaling and normalization transforms
    image = transform.eval_transform(image=image)["image"]
    image = transform.norm_transform(image=image)["image"]
    image = ToTensorV2()(image=image)["image"]

    # unsqueeze the 3-D tensor to obtain 4-D because pytorch model requires 4-D (batch dimension)
    outputs = model(image.unsqueeze(0))

    # get the index of the top-1 class
    _, pred = outputs.topk(1)

    print(f"prediction: {INT_LABEL_MAPPING[pred.item()]}")


if __name__ == '__main__':
    parser = ArgumentParser(description="This script runs prediction with the Resnet18 model.")
    parser.add_argument('--model_path', type=str, help="The absolute path to the Resnet model checkpoint file.")
    parser.add_argument('--image_path', type=str, help="Absolute path to the query image.")

    args = parser.parse_args()

    main(**vars(args))
