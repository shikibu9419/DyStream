"""
Image processing utilities aligned with app.py:
face detection, cropping, masking, and motion latent extraction.
"""

import os
import importlib.util as _ilu
import logging
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms

logger = logging.getLogger(__name__)

_face_detector = None


def load_face_detector(tools_path: str):
    """Load MediaPipe face detector used in app.py."""
    global _face_detector

    if _face_detector is not None:
        return

    face_detector_path = os.path.join(tools_path, "utils", "face_detector.py")
    if not os.path.exists(face_detector_path):
        raise FileNotFoundError(f"Face detector not found: {face_detector_path}")

    _fd_spec = _ilu.spec_from_file_location("vis_face_detector", face_detector_path)
    _fd_mod = _ilu.module_from_spec(_fd_spec)
    _fd_spec.loader.exec_module(_fd_mod)
    FaceDetector = _fd_mod.FaceDetector

    model_path = os.path.join(tools_path, "utils", "face_landmarker.task")
    if not os.path.exists(model_path):
        # Match app.py behavior: download if missing
        import urllib.request
        logger.info("Downloading face landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)

    _face_detector = FaceDetector(
        mediapipe_model_asset_path=model_path,
        face_detection_confidence=0.5,
        num_faces=1,
    )

    logger.info("Face detector loaded for streaming image processing")


def scale_bbox(bbox, h, w, scale=1.8):
    sw = (bbox[2] - bbox[0]) / 2
    sh = (bbox[3] - bbox[1]) / 2
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    sw *= scale
    sh *= scale
    scaled = [cx - sw, cy - sh, cx + sw, cy + sh]
    scaled[0] = np.clip(scaled[0], 0, w)
    scaled[2] = np.clip(scaled[2], 0, w)
    scaled[1] = np.clip(scaled[1], 0, h)
    scaled[3] = np.clip(scaled[3], 0, h)
    return scaled


def get_mask(bbox, hd, wd, scale=1.0, return_pil=True):
    if min(bbox) < 0:
        raise ValueError("Invalid mask")
    bbox = scale_bbox(bbox, hd, wd, scale=scale)
    x0, y0, x1, y1 = [int(v) for v in bbox]
    mask = np.zeros((hd, wd, 3), dtype=np.uint8)
    mask[y0:y1, x0:x1, :] = 255
    if return_pil:
        return Image.fromarray(mask)
    return mask


def generate_crop_bounding_box(h, w, center, size=512):
    half_size = size // 2
    y1 = max(center[0] - half_size, 0)
    x1 = max(center[1] - half_size, 0)
    y2 = min(center[0] + half_size, h)
    x2 = min(center[1] + half_size, w)
    return [x1, y1, x2, y2]


def crop_from_bbox(image, center, bbox, size=512):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    half_size = size // 2
    cropped = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    cropped[(y1 - (center[0] - half_size)):(y2 - (center[0] - half_size)),
            (x1 - (center[1] - half_size)):(x2 - (center[1] - half_size))] = image[y1:y2, x1:x2]
    return cropped


def process_image(
    image_pil: Image.Image,
    device: torch.device,
    vis_models: dict,
    tools_path: str,
    crop: bool = True,
    union_bbox_scale: float = 1.6
) -> Tuple[Image.Image, Image.Image, torch.Tensor]:
    """
    app.py-equivalent image processing pipeline:
    face detection, crop, resize, mask, and motion latent extraction.
    """
    load_face_detector(tools_path)

    cfg_path = os.path.join(tools_path, "configs", "audio_head_animator.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    pixel_transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize([0.5], [0.5]),
    ])
    resize_transform = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)

    img = image_pil.convert("RGB")
    img_np = np.array(img)
    state = torch.get_rng_state()

    det_res = _face_detector.get_face_xy_rotation_and_keypoints(
        img_np, cfg.data.mouth_bbox_scale, cfg.data.eye_bbox_scale
    )

    if not det_res or len(det_res[6]) == 0:
        raise RuntimeError("No face detected. Please upload an image with a clear face.")

    person_id = 0
    mouth_bbox = np.array(det_res[6][person_id])
    eye_bbox = det_res[7][person_id]
    face_contour = np.array(det_res[8][person_id])
    left_eye_bbox = eye_bbox["left_eye"]
    right_eye_bbox = eye_bbox["right_eye"]

    if crop:
        face_bbox = det_res[5][person_id]
        x1, y1 = face_bbox[0]
        x2, y2 = face_bbox[1]
        center = [(y1 + y2) // 2, (x1 + x2) // 2]
        width = x2 - x1
        height = y2 - y1
        max_size = int(max(width, height) * union_bbox_scale)
        hd, wd = img.size[1], img.size[0]
        crop_bbox = generate_crop_bounding_box(hd, wd, center, max_size)
        img_array = np.array(img)
        cropped_img = crop_from_bbox(img_array, center, crop_bbox, size=max_size)
        img = Image.fromarray(cropped_img)

        det_res = _face_detector.get_face_xy_rotation_and_keypoints(
            cropped_img, cfg.data.mouth_bbox_scale, cfg.data.eye_bbox_scale
        )
        if not det_res or len(det_res[6]) == 0:
            raise RuntimeError("No face detected after cropping. Please try a different image.")
        mouth_bbox = np.array(det_res[6][person_id])
        eye_bbox = det_res[7][person_id]
        face_contour = np.array(det_res[8][person_id])
        left_eye_bbox = eye_bbox["left_eye"]
        right_eye_bbox = eye_bbox["right_eye"]

    def augmentation(images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, list):
            transformed = [transforms.functional.to_tensor(img_item) for img_item in images]
            return transform(torch.stack(transformed, dim=0))
        return transform(transforms.functional.to_tensor(images))

    pixel_values_ref = augmentation([img], pixel_transform, state)
    pixel_values_ref = (pixel_values_ref + 1) / 2
    new_hd, new_wd = img.size[1], img.size[0]

    mouth_mask = resize_transform(get_mask(mouth_bbox, new_hd, new_wd, scale=1.0))
    left_eye_mask = resize_transform(get_mask(left_eye_bbox, new_hd, new_wd, scale=1.0))
    right_eye_mask = resize_transform(get_mask(right_eye_bbox, new_hd, new_wd, scale=1.0))
    face_contour_resized = resize_transform(Image.fromarray(face_contour))

    eye_mask = np.bitwise_or(np.array(left_eye_mask), np.array(right_eye_mask))
    combined_mask = np.bitwise_or(eye_mask, np.array(mouth_mask))

    combined_mask_tensor = torch.from_numpy(combined_mask / 255.0).permute(2, 0, 1).unsqueeze(0)
    face_contour_tensor = torch.from_numpy(np.array(face_contour_resized) / 255.0).permute(2, 0, 1).unsqueeze(0)

    masked_ref = pixel_values_ref * combined_mask_tensor + face_contour_tensor * (1 - combined_mask_tensor)
    masked_ref = masked_ref.clamp(0, 1)

    # Convert to PIL
    resized_np = (pixel_values_ref.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    masked_np = (masked_ref.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    resized_pil = Image.fromarray(resized_np)
    masked_pil = Image.fromarray(masked_np)

    # Extract motion latent using motion encoder
    motion_encoder = vis_models.get("motion_encoder")
    if motion_encoder is None:
        raise RuntimeError("Motion encoder not found in visualization models")

    vis_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    resized_img_tensor = vis_transform(resized_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        motion_latent = motion_encoder(resized_img_tensor)[0]  # [1, 512]

    return resized_pil, masked_pil, motion_latent.cpu()
