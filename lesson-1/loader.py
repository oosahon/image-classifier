from fastai.vision.all import (
    DataBlock,
    CategoryBlock,
    ImageBlock,
    RandomSplitter,
    get_image_files,
    Resize,
)


def load_data(path):
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2),
        get_y=lambda x: x.parent.name,
        item_tfms=Resize(224),
    ).dataloaders(path, bs=32)

    return dls
