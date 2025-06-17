import cv2
import torch
import numpy as np
from skimage.morphology import skeletonize
from typing import Tuple, List, Dict, Optional
from line_fitting import LaneLineFitter

# 상태 저장용 전역 변수
last_valid_center_x: Optional[int] = None
fitter = LaneLineFitter(angle_threshold_deg=5)  # 안정화용 객체

def preprocess_image(frame: np.ndarray, input_size: Tuple[int, int] = (512, 256)) -> torch.Tensor:
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size)
    img_tensor = torch.from_numpy(img_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0)
    return img_tensor

def infer_mask(model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred_mask

def compute_skeleton(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask > 0).astype(np.uint8)

def extract_skeleton_points(skeleton: np.ndarray) -> np.ndarray:
    points = np.column_stack(np.where(skeleton > 0))
    return points

def remap_skeleton_coords(points: np.ndarray, orig_shape: Tuple[int, int, int], mask_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    if points.size == 0:
        return []
    scale_x = orig_shape[1] / mask_shape[1]
    scale_y = orig_shape[0] / mask_shape[0]
    remapped = [(int(x * scale_x), int(y * scale_y)) for y, x in points]
    return remapped

def generate_straight_line(mask_shape: Tuple[int, int], x_pos: int) -> np.ndarray:
    h, w = mask_shape
    line = np.zeros(mask_shape, dtype=np.uint8)
    line[:, max(0, min(x_pos, w - 1))] = 1
    return line

def process_frame(frame: np.ndarray, model: torch.nn.Module, device: torch.device, uuid: int,
                  input_size: Tuple[int, int] = (512, 256), mode: str = "center",
                  frame_count: int = 0) -> Dict:
    global last_valid_center_x

    skeleton_points_mask_res = np.array([])
    img_tensor = preprocess_image(frame, input_size)
    pred_mask = infer_mask(model, img_tensor, device)

    h, w = pred_mask.shape
    # center_x = w // 2
    center_x = int(w * 0.485)
    merged_mask = np.zeros_like(pred_mask, dtype=bool)
    use_fallback_line = False
    fitted_line = None

    if mode == "center":
        ratio = 0.15
        sx = int(w * (0.5 - ratio / 2))
        ex = int(w * (0.5 + ratio / 2))

        temp = np.zeros_like(pred_mask)
        temp[:, sx:ex] = pred_mask[:, sx:ex]
        center_mask = np.isin(temp, [1, 2, 3])
        skeleton = compute_skeleton(center_mask.astype(np.uint8))
        skeleton_points_mask_res = extract_skeleton_points(skeleton)

        if skeleton_points_mask_res.shape[0] >= 10:
            fitted_line = fitter.update(skeleton_points_mask_res)
            merged_mask = center_mask
        else:
            if last_valid_center_x is not None:
                print("[CENTER] 중심선 부족 → 이전 선으로 보정")
                fallback_line = generate_straight_line((h, w), last_valid_center_x)
                skeleton = fallback_line
                skeleton_points_mask_res = extract_skeleton_points(skeleton)
                fitted_line = None  # fallback이므로 fit 생략
                use_fallback_line = True
            else:
                print("[CENTER] 중심선 없음 → 추론 불가")
                skeleton_points_mask_res = np.array([])

    elif mode in ["left", "right"]:
        ratio = 0.4
        if mode == "left":
            sx, ex = 0, int(w * ratio)
            center_x = int(w * 0.3)
        else:
            sx, ex = int(w * (1 - ratio)), w
            center_x = int(w * 0.7)

        temp = np.zeros_like(pred_mask)
        temp[:, sx:ex] = pred_mask[:, sx:ex]
        dashed_only = np.logical_and(temp == 2, temp > 0)

        if np.count_nonzero(dashed_only) > 50:
            merged_mask = dashed_only
            skeleton = compute_skeleton(merged_mask.astype(np.uint8))
            skeleton_points_mask_res = extract_skeleton_points(skeleton)
            fitted_line = fitter.update(skeleton_points_mask_res)
        else:
            print(f"[{mode.upper()}] 점선 없음 → 보정선 사용")
            fallback_line = generate_straight_line((h, w), center_x)
            skeleton = fallback_line
            skeleton_points_mask_res = extract_skeleton_points(skeleton)
            fitted_line = None
            use_fallback_line = True

    # === 조향각 계산 ===
    offset, angle, avg_center_x = None, None, None
    if fitted_line is not None:
        x_at_bottom = fitter.get_x_at_bottom(fitted_line, h)
        if x_at_bottom is not None:
            offset = x_at_bottom - center_x
            dy = h // 2
            angle_rad = np.arctan2(offset, dy)
            angle = np.degrees(angle_rad)
            avg_center_x = x_at_bottom
            last_valid_center_x = x_at_bottom
    else:
        if last_valid_center_x is not None:
            avg_center_x = last_valid_center_x
            offset = last_valid_center_x - center_x
            angle = np.degrees(np.arctan2(offset, h // 2))

    skeleton_xy_orig_res = remap_skeleton_coords(skeleton_points_mask_res, frame.shape, (h, w))

    return {
        "uuid": uuid,
        "offset": float(offset) if offset is not None else None,
        "steering_angle": float(angle) if angle is not None else None,
        "skeleton_points": [list(p) for p in skeleton_xy_orig_res],
        "pred_mask": pred_mask,
        "avg_center_x_mask_res": avg_center_x,
        "pred_mask_shape": pred_mask.shape,
        "mode": mode,
        "fallback": use_fallback_line,
        "frame_count": frame_count,
        "steering_center_x": center_x
    }
