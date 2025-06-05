import cv2
import torch
import numpy as np
from skimage.morphology import skeletonize


def process_frame(frame, model, device, input_size=(512, 256)):
    original_frame = frame.copy()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size)
    img_tensor = torch.from_numpy(img_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    h, w = pred_mask.shape
    frame_center_x = w // 2

    center_mask = np.logical_or(pred_mask == 1, pred_mask == 2)
    skeleton = skeletonize(center_mask)
    center_points = np.column_stack(np.where(skeleton))  # [y, x]

    avg_center_x = None
    if len(center_points) > 10:
        lower_third = center_points[center_points[:, 0] > h * 0.66]
        if len(lower_third) > 0:
            avg_center_x = int(np.mean(lower_third[:, 1]))

    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[pred_mask == 1] = [0, 255, 0]
    color_mask[pred_mask == 2] = [0, 0, 255]
    color_mask[skeleton] = [255, 255, 0]

    color_mask_resized = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]))
    overlay = cv2.addWeighted(original_frame, 0.7, color_mask_resized, 0.3, 0)

    if avg_center_x is not None:
        scale = frame.shape[1] / w
        cx = int(avg_center_x * scale)
        cy_bottom = frame.shape[0]
        cy_top = int(frame.shape[0] * 0.5)
        cv2.line(overlay, (cx, cy_bottom), (cx, cy_top), (255, 255, 0), 2)
    cv2.line(overlay, (frame.shape[1]//2, frame.shape[0]), (frame.shape[1]//2, int(frame.shape[0]*0.5)), (0, 255, 255), 1)

    return overlay
