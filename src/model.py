from loader import load_data
from fastai.vision.all import vision_learner, resnet18, error_rate


def model(path):
    dls = load_data(path)
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)
    return learn
