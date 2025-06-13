import cv2
import torch
import numpy as np
from skimage.morphology import skeletonize
from typing import Tuple, List, Dict, Optional
import scipy.ndimage


def preprocess_image(frame: np.ndarray, input_size: Tuple[int, int] = (512, 256)) -> torch.Tensor:
    """
    BGR -> RGB, resize, normalize, tensor 변환
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size)
    img_tensor = torch.from_numpy(img_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def infer_mask(model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    모델 추론 후 class mask 반환 (H, W) uint8
    """
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred_mask


def extract_center_lane_mask(pred_mask: np.ndarray, center_ratio: float = 0.5) -> np.ndarray:
    """
    중앙 차선 마스크 추출 (1 or 2 클래스만 포함)
    """
    h, w = pred_mask.shape
    start_x = int(w * (0.5 - center_ratio / 2))
    end_x = int(w * (0.5 + center_ratio / 2))
    center_zone = np.zeros_like(pred_mask, dtype=bool)
    center_zone[:, start_x:end_x] = True
    center_mask = np.logical_and(np.isin(pred_mask, [1, 2]), center_zone)
    return center_mask.astype(np.uint8)


def extract_left_lane_mask(pred_mask: np.ndarray, lane_width_ratio: float = 0.5) -> np.ndarray:
    """
    좌측 두 개 차선을 기반으로 마스크 생성 (class 1 + class 2)
    """
    h, w = pred_mask.shape
    end_x = int(w * lane_width_ratio)
    left_zone = np.zeros_like(pred_mask, dtype=bool)
    left_zone[:, :end_x] = True

    # class 1 (좌측 차선) 또는 class 2 (우측 차선) 중 좌측에 위치한 영역 추출
    left_mask = np.logical_and(np.logical_or(pred_mask == 1, pred_mask == 2), left_zone)

    return left_mask.astype(np.uint8)



def extract_right_lane_mask(pred_mask: np.ndarray, lane_width_ratio: float = 0.25) -> np.ndarray:
    """
    우측 차선 마스크 추출 (class 2)
    """
    h, w = pred_mask.shape
    start_x = int(w * (1 - lane_width_ratio))
    right_zone = np.zeros_like(pred_mask, dtype=bool)
    right_zone[:, start_x:] = True
    right_mask = np.logical_and(np.logical_or(pred_mask == 1, pred_mask == 2), right_zone)

    return right_mask.astype(np.uint8)


def compute_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    마스크에 대해 skeletonize 수행
    """
    return skeletonize(mask > 0).astype(np.uint8)


def extract_skeleton_points(skeleton: np.ndarray) -> np.ndarray:
    """
    skeleton (H, W)에서 (y, x) 좌표 배열 추출
    """
    points = np.column_stack(np.where(skeleton > 0))
    return points


def remap_skeleton_coords(points: np.ndarray, orig_shape: Tuple[int, int, int], mask_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    skeleton (y, x) 좌표를 원본 이미지 크기로 리맵 (x, y) 튜플 리스트 반환
    """
    if points.size == 0:
        return []
    scale_x = orig_shape[1] / mask_shape[1]
    scale_y = orig_shape[0] / mask_shape[0]
    remapped = [(int(x * scale_x), int(y * scale_y)) for y, x in points]
    return remapped


def compute_steering(skeleton_points_mask_res: np.ndarray, mask_shape: Tuple[int, int],
                    lower_third_ratio: float = 0.66) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    Calculates raw steering offset and angle based on skeleton points in the lower third of the mask.
    Args:
        skeleton_points_mask_res: (N, 2) ndarray, (y, x) coordinates in mask resolution.
        mask_shape: (H, W) tuple of the mask resolution.
        lower_third_ratio: Ratio of the image height to consider for steering calculation.
    Returns:
        Tuple of (offset_x, angle_deg, avg_center_x) or (None, None, None) if no suitable points are found.
    """
    h, w = mask_shape
    frame_center_x = w // 2
    avg_center_x = None
    offset_x = None # Initialize offset_x
    angle_deg = None

    if skeleton_points_mask_res.size == 0:
        # print("No skeleton points provided for steering calculation.")
        return None, None, None

    # 하단 1/3만 고려 (하단에서 주행 판단이 유효)
    lower_third_points = skeleton_points_mask_res[skeleton_points_mask_res[:, 0] > h * lower_third_ratio]

    if lower_third_points.size == 0:
        # print("No skeleton points found in the lower third.")
        return None, None, None

    # Heuristic: require a minimum number of points in the original skeleton
    # before considering points in the lower third for steering.
    if len(skeleton_points_mask_res) > 10: # Check original points count
        if len(lower_third_points) > 0:
            avg_center_x = int(np.mean(lower_third_points[:, 1]))

            offset_x = avg_center_x - frame_center_x
            # Use h // 2 as the "lookahead distance" for angle calculation, similar to predict_video.py
            angle_rad = np.arctan2(offset_x, h // 2)
            angle_deg = np.degrees(angle_rad)
            # print(f"[Raw Steering] Offset: {offset_x}, Angle: {angle_deg:.2f} deg")
        else:
            # print("No skeleton points in lower third, though >10 points overall.")
            pass # offset_x and angle_deg remain None
    else:
        # print("Too few skeleton points overall (<10) to calculate steering.")
        pass # offset_x and angle_deg remain None

    return float(offset_x) if offset_x is not None else None, \
           float(angle_deg) if angle_deg is not None else None, \
           avg_center_x


def process_frame(frame: np.ndarray, model: torch.nn.Module, device: torch.device, uuid: int, input_size: Tuple[int, int] = (512, 256), mode: str = "center") -> Dict:
    """
    전체 프레임 처리 흐름
    mode: "center", "left", "right"
    """
    mode_funcs = {
        "left": extract_left_lane_mask,
        "right": extract_right_lane_mask,
        # "center": extract_center_lane_mask  # fallback only
    }

    skeleton_points_mask_res = np.array([]) # Initialize (y,x) points in mask resolution

    img_tensor = preprocess_image(frame, input_size)
    pred_mask = infer_mask(model, img_tensor, device)

    if mode == "center":
        # Primary method: Focus on class 1 or 2 lanes within a central horizontal region.
        # This helps to ignore extraneous lane detections far from the vehicle's path.
        center_lane_mask = extract_center_lane_mask(pred_mask)
        skeleton = compute_skeleton(center_lane_mask)
        skeleton_points_mask_res = extract_skeleton_points(skeleton)

        # Fallback: If no lanes are found in the central zone, try the broader approach.
        # This might be useful if the vehicle is significantly off-center but lanes are still detected.
        if skeleton_points_mask_res.size == 0:
            # print("[Process Frame - Center Mode] No skeleton from central zone, trying broader mask.")
            broader_center_mask = np.logical_or(pred_mask == 1, pred_mask == 2)
            skeleton = compute_skeleton(broader_center_mask)
            skeleton_points_mask_res = extract_skeleton_points(skeleton)

    elif mode in mode_funcs: # "left" or "right"
        mask = mode_funcs[mode](pred_mask)
        skeleton = compute_skeleton(mask)
        skeleton_points_mask_res = extract_skeleton_points(skeleton)
    else:
        print(f"Warning: Unknown mode '{mode}'. Defaulting to center mode logic.")
        # Default to center mode logic if mode is invalid
        center_lane_mask = np.logical_or(pred_mask == 1, pred_mask == 2)
        skeleton = compute_skeleton(center_lane_mask)
        skeleton_points_mask_res = extract_skeleton_points(skeleton)

    # Calculate raw offset and steering angle
    offset, steering_angle, avg_center_x_mask_res = compute_steering(skeleton_points_mask_res, pred_mask.shape)

    # Remap all drivable skeleton points for visualization
    skeleton_xy_orig_res = remap_skeleton_coords(skeleton_points_mask_res, frame.shape, pred_mask.shape)

    return {
        "uuid": uuid,
        "offset": offset,
        "steering_angle": steering_angle, # This is the RAW angle
        "skeleton_points": [list(p) for p in skeleton_xy_orig_res], # Drivable path skeleton (model-based)
        "pred_mask": pred_mask, # Model's raw prediction mask (resized resolution)
        "avg_center_x_mask_res": avg_center_x_mask_res, # Average x of centerline in mask coords
        "pred_mask_shape": pred_mask.shape # (H, W) of the prediction mask
    }
