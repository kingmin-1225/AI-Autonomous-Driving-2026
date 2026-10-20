"""수집·추론 공통 제어: 명령은 유지하고 실제 조작만 완만하게 만든다."""
import math
from terminal_text import tr
import numpy as np

DEFAULT_DRIVING = dict(max_speed=22.5, throttle=0.375, steering_limit=0.45,
                       steering_rate=1.5)
SPEED_MARGIN = 3.0  # 제한 속도에 도달하기 전 가속을 줄이는 구간 (환경 속도 단위)
SPEED_BRAKE_GAIN = 0.3
COAST_BRAKE = 0.05  # 가속 명령이 없을 때 서서히 멈추는 약한 자동 제동


def driving_settings(settings=None):
    result = dict(DEFAULT_DRIVING)
    if settings is not None:
        if not isinstance(settings, dict) or settings.keys() - result.keys():
            raise ValueError(tr('driving 설정 이름을 확인하세요.'))
        result.update(settings)
    for key, value in result.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(tr('driving.{key}는 유한한 양수여야 합니다.', key=key))
    if result['throttle'] > 1 or result['steering_limit'] > 1:
        raise ValueError(tr('throttle과 steering_limit은 1 이하여야 합니다.'))
    return result


class DrivingController:
    def __init__(self, settings=None, fps=50):
        self.settings = driving_settings(settings)
        self.step_size = self.settings['steering_rate'] / fps
        self.reset()

    def reset(self):
        self.steering = 0.0

    def apply(self, command, speed):
        steer, gas, brake = np.asarray(command, dtype=float)
        target = np.clip(steer, -1, 1) * self.settings['steering_limit']
        self.steering += np.clip(target - self.steering, -self.step_size, self.step_size)
        # 속도 제한으로 가속이 차단된 경우와 운전자가 가속을 놓은 경우를 구분한다.
        if gas <= 0:
            brake = max(brake, COAST_BRAKE)
        # 가속 크기와 제한 속도를 분리한다. 제한 속도 초과 시 자동 감속한다.
        gas = min(max(gas, 0), self.settings['throttle'])
        gas *= np.clip((self.settings['max_speed'] - speed) / SPEED_MARGIN, 0, 1)
        brake = np.clip(max(brake, (speed - self.settings['max_speed']) * SPEED_BRAKE_GAIN), 0, 1)
        if brake > 0:
            gas = 0.0
        return np.array([self.steering, gas, brake], dtype=np.float32)
