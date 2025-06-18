import numpy as np
def estimate_stopline_distance(mask: np.ndarray, scale_factor: float) -> float | None:
    """
    정지선 클래스가 있는 mask에서 거리 추정 (픽셀 → 실제 거리(m))
    Args:
        mask (np.ndarray): segmentation 결과 마스크 (H, W), 각 픽셀은 class ID
        scale_factor (float): 픽셀 → 미터 변환 계수 (ex: 0.05 m/pixel)
    Returns:
        float | None: 정지선까지의 거리 (m), 없으면 None 반환
    """
    STOP_LINE_CLASS_ID = 4  # 예: 정지선 class가 2로 라벨링 되어 있을 경우
    # 정지선에 해당하는 모든 픽셀 좌표 (y, x)
    stopline_ys = np.where(mask == STOP_LINE_CLASS_ID)[0]
    if stopline_ys.size == 0:
        return None  # 정지선 없음
    # 화면 하단에서 가장 가까운 정지선 y좌표
    y_max = np.max(stopline_ys)
    frame_height = mask.shape[0]
    pixel_distance = frame_height - y_max
    real_distance = pixel_distance * scale_factor
    return real_distance