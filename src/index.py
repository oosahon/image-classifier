from model import model
from loader import load_data
from fastai.vision.all import PILImage


def image_classifier(image):
    dls = load_data("data/trainers")
    learner = model(dls)
    prediction, _, probs = learner.predict(PILImage.create(image))
    print(f"This is a: {prediction}.")
    print(f"Confidence level: {probs[0]:.4f}")


image_classifier("./data/validators/croc-1.jpg")
