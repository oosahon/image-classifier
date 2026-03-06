from asyncio import sleep
from ddgs import DDGS
from fastcore.all import *
import time
from fastai.vision.all import (
    download_images as fast_download_images,
    resize_images as fast_resize_images,
)
from image_verifier import verify_image


def search_images(keywords, max_images=200):
    res = DDGS().images(keywords, max_results=max_images)
    return L(res).itemgot("image")


def download_images(keyword, max_images=200):
    dir = f"data/trainers/{keyword}"
    Path(dir).mkdir(exist_ok=True)
    urls = search_images(keyword, max_images)

    fast_download_images(urls=urls, dest=dir)
    fast_resize_images(dir, max_size=400, dest=dir)
    time.sleep(5)
    verify_image(dir)


# download_images("forest", max_images=200)

# download_images("bird", max_images=200)
download_images("crocodile", max_images=200)
