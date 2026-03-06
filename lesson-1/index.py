from model import model


def bird_predictor(image):
    learner = model("downloads")
    is_bird, _, probs = learner.predict(image)
    print(f"This is a: {is_bird}.")
    print(f"Probability it's a bird: {probs[0]:.4f}")


bird_predictor("test-images/12.jpg")
