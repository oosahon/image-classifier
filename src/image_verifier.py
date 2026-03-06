from fastai.vision.all import verify_images, get_image_files, get_files
from pathlib import Path


def verify_image(dir):
    Path(dir).mkdir(exist_ok=True)
    files = get_files(dir)
    images = get_image_files(dir)

    for file in files:
        if file not in images:
            print(f"Cleaning up: {file}")
            Path.unlink(file)
