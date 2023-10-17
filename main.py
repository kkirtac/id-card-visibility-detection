"""Prediction script. Runs id card visibility prediciton with given resnet18 model."""
import sys
sys.path.append('.')
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from fastapi import FastAPI, File, UploadFile
from modules.model import ResnetLightning
from modules.transforms import TransformationsBase


INT_LABEL_MAPPING = {
    0: 'FULL_VISIBILITY',
    1: 'PARTIAL_VISIBILITY',
    2: 'NO_VISIBILITY',
}

def read_from_file(file_object):
    arr = np.fromstring(file_object.read(), np.uint8)
    img_np = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    return img_np

# we are using the pixel mean std of ImageNet training set
# because our trained model used these values for image normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# load model
model = ResnetLightning.load_from_checkpoint("/app/artifacts/model_ckpt.ckpt").eval()

app = FastAPI()

@app.post("/predict_visibility")
def predict_from_file(file: UploadFile = File(...)):
    """Run prediction of the model with a single image."""

    # repeat the blue channel on green and red channels to receive a 3-channel image
    # because our CNN model requires 3-channel input
    image = read_from_file(file.file)
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
    scores = torch.nn.functional.softmax(model(image.unsqueeze(0)), dim=1).flatten()
    
    return {
        'FULL_VISIBILITY': scores[0].item(),
        'PARTIAL_VISIBILITY': scores[1].item(),
        'NO_VISIBILITY': scores[2].item(),
    }
