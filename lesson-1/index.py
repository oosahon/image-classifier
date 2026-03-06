from model import model
from fastai.vision.all import PILImage


def bird_predictor(image):
    learner = model("downloads")
    is_bird, _, probs = learner.predict(image)
    print(f"This is a: {is_bird}.")
    print(f"Probability it's a bird: {probs[0]:.4f}")


bird_predictor("test-images/12.jpg")
# bird_predictor("downloads/forests/0e1134f5-5e9a-489a-a8cd-067da4ab907b.jpg")
