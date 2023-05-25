import PIL
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict
from segment_anything import build_sam, SamPredictor
from segment_anything.utils.amg import remove_small_regions
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import numpy as np


def load_groundingdino_model(model_config_path, model_checkpoint_path):
    args = SLConfig.fromfile(model_config_path)
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print('loading GroundingDINO:', load_res)
    _ = model.eval()
    return model


class GroundingModule(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
        groundingdino_checkpoint = "./checkpoints/groundingdino_swint_ogc.pth"
        groundingdino_config_file = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"

        self.grounding_model = load_groundingdino_model(groundingdino_config_file, groundingdino_checkpoint).to(device)
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    def prompt2mask(self, original_image, prompt, box_threshold=0.25, text_threshold=0.25, num_boxes=10):
        def image_transform_grounding(init_image):
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image, _ = transform(init_image, None)  # 3, h, w
            return init_image, image

        image_np = np.array(original_image, dtype=np.uint8)
        prompt = prompt.lower()
        prompt = prompt.strip()
        if not prompt.endswith("."):
            prompt = prompt + "."
        _, image_tensor = image_transform_grounding(original_image)
        boxes, logits, phrases = predict(self.grounding_model,
                                         image_tensor, prompt, box_threshold, text_threshold, device='cpu')
        print(phrases)
        # from PIL import Image, ImageDraw, ImageFont
        H, W = original_image.size[1], original_image.size[0]

        draw_img = original_image.copy()
        draw = ImageDraw.Draw(draw_img)
        for box in boxes:
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            # draw
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            draw.rectangle([x0, y0, x1, y1], outline=color, width=10)

        if boxes.size(0) > 0:
            boxes = boxes * torch.Tensor([W, H, W, H])
            boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
            boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]

            self.sam_predictor.set_image(image_np)

            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes, image_np.shape[:2])
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.device),
                multimask_output=False,
            )

            # remove small disconnected regions and holes
            fine_masks = []
            for mask in masks.to('cpu').numpy():  # masks: [num_masks, 1, h, w]
                fine_masks.append(remove_small_regions(mask[0], 400, mode="holes")[0])
            masks = np.stack(fine_masks, axis=0)[:, np.newaxis]
            masks = torch.from_numpy(masks)

            num_obj = min(len(logits), num_boxes)
            mask_map = None

            full_img = None
            for obj_ind in range(num_obj):
                # box = boxes[obj_ind]

                m = masks[obj_ind][0]

                if full_img is None:
                    full_img = np.zeros((m.shape[0], m.shape[1], 3))
                    mask_map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)

                mask_map[m != 0] = obj_ind + 1
                color_mask = np.random.random((1, 3)).tolist()[0]
                full_img[m != 0] = color_mask
            full_img = (full_img * 255).astype(np.uint8)
            full_img = PIL.Image.fromarray(full_img)
            draw_img = PIL.Image.blend(draw_img, full_img, 0.5)
        return draw_img
