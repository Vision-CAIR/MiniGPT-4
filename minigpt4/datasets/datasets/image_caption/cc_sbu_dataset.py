import os
from PIL import Image
import webdataset as wds
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.image_caption.image_caption_datasets import ImageCaptionDataset


class CCSBUDataset(BaseDataset):
    def __init__(self, vision_processor, text_processor, location):
        super().__init__(x_processor=vision_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.x_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "vision": sample[0],
            "text_input": self.text_processor(sample[1]["caption"]),
        }


class CCSBUAlignDatasetImageImageCaptionDataset(ImageCaptionDataset):

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.x_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.x_processor(image)
        caption = ann["caption"]

        return {
            "vision": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }


class CCDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(x_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "txt", handler=wds.warn_and_continue),
            wds.map_tuple(self.x_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "vision": sample[0],
            "text_input": sample[1],
        }
