from typing import Optional
import os

from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


# A local RMBG-2.0 or BiRefNet directory can replace the Hub model.
RMBG2_MODEL_PATH = os.environ.get("PIXAL3D_RMBG_MODEL_PATH", "briaai/RMBG-2.0")


class RMBG2:
    def __init__(
        self,
        model_name: str = RMBG2_MODEL_PATH,
        alpha_threshold: Optional[float] = None,
    ):
        if model_name == "briaai/RMBG-2.0":
            model_name = RMBG2_MODEL_PATH
        if alpha_threshold is not None and not 0.0 <= alpha_threshold <= 1.0:
            raise ValueError(f"alpha_threshold must be in [0, 1], got {alpha_threshold}")
        self.model_name = model_name
        self.alpha_threshold = alpha_threshold
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        if self.alpha_threshold is not None:
            cutoff = int(round(self.alpha_threshold * 255))
            mask = mask.point([0 if value < cutoff else 255 for value in range(256)])
        image.putalpha(mask)
        return image
