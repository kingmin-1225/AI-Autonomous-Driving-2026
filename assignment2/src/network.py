# [수정 가능] ① 모델 구조 ② 보상함수 ③ 수집 행동 선택
# 클래스·함수의 이름과 입출력 규약은 유지하세요. 고정 보조 코드는 파일 하단에 있습니다.
# 모델 입력 (B,96,96,3), 출력 유한한 행동 점수 (B,9). 행동 9개의 순서·의미는 고정입니다.
# 추론·서버 채점은 점수가 가장 높은 행동을 선택합니다.
# 학습은 제공된 Double DQN 코드로 고정합니다.
# 학습 후 저장되는 .network.py와 .pth 두 파일을 제출합니다.
import torch

# ① 모델: 영상·센서 입력으로 행동 9개의 Q값을 계산합니다.
# [고정 규약] ClassificationNetwork(device=None), forward(observation)의 입출력.
class ClassificationNetwork(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()

        device = _setup_model(self, device)

        # 수정: 영상 특징 추출.
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=5, stride=4),
            torch.nn.ReLU(),
            MPSCompatibleAdaptiveAvgPool2d((4, 4)),
        )

        # 수정: 영상 특징 128개 + 센서 7개 → 행동별 점수.
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(8 * 4 * 4 + 7, self.n_classes),
        )

        # 베이스라인의 가속 점수 계산 (학습 가능).
        self.throttle_head = torch.nn.Linear(1, 2)
        with torch.no_grad():
            self.throttle_head.weight.copy_(torch.tensor([[0.], [-20.]]))
            self.throttle_head.bias.copy_(torch.tensor([0., 4.7]))
        self.to(device)

    def forward(self, observation):
        # 전처리 → 특징 추출 → 조향·가속 점수 결합.

        batch_size = observation.shape[0]

        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(
            observation.detach(), batch_size)

        x = observation / 255.0

        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.conv(x)
        x = x.reshape(batch_size, -1)

        x = torch.cat([x, speed, abs_sensors, steering, gyroscope], dim=1)
        raw = self.fc(x)
        steering_values = torch.stack([raw[:, [0,1,2]].max(1).values,
                                       raw[:, [3,5,7]].max(1).values,
                                       raw[:, [4,6,8]].max(1).values], dim=1)
        throttle_values = self.throttle_head(speed)

        steer_index = [0,0,0,1,2,1,2,1,2]
        throttle_index = [0,1,0,0,0,1,1,0,0]
        values = steering_values[:, steer_index] + throttle_values[:, throttle_index]

        brake_penalty = raw.new_tensor([0,0,5,0,0,0,0,5,5])
        return values + .01 * raw - brake_penalty

    def actions_to_classes(self, actions):
        return _fixed_actions_to_classes(self, actions)

    def scores_to_action(self, scores):
        return _fixed_scores_to_action(self, scores)

    def extract_sensor_values(self, observation, batch_size):
        return _fixed_extract_sensor_values(self, observation, batch_size)

# ② 보상: 아래 계수 또는 calculate_reward()를 수정한 뒤 데이터를 다시 수집합니다.
REWARD_SETTINGS = dict(new_road=1.0, center=0.5, grass=0.5, time=0.0, finish=0.0)

# [고정 규약] calculate_reward(context)는 유한한 숫자 하나를 반환합니다.
def calculate_reward(context):
    # 새 구간 수, 중앙 근접도(0~1), 도로 여부, 완주 여부로 스텝 보상을 계산합니다.
    # context에는 speed, dt, elapsed, aborted도 제공됩니다.

    reward = context['new_road_segments'] * (
        REWARD_SETTINGS['new_road'] + REWARD_SETTINGS['center'] * context['center_alignment'])
    reward -= REWARD_SETTINGS['time']
    if not context['on_road']:
        reward -= REWARD_SETTINGS['grass']
    if context['finished']:
        reward += REWARD_SETTINGS['finish']
    return reward


# ③ 수집: 아래 ε-greedy 구현은 자유롭게 변경할 수 있습니다.
def collection_action(model, observation, device, epsilon, rng):
    # [고정 규약] 위 함수 이름·인자 유지, 반환값은 행동 번호 0~8의 정수.
    # [수정 가능] 함수 본문 전체. ε-greedy 대신 정책 확률 샘플링 등 사용 가능.
    if epsilon > 0 and rng.random() < epsilon:
        return int(rng.integers(9))
    with torch.inference_mode():
        scores = model(torch.as_tensor(observation[None].copy(), device=device, dtype=torch.float32))
        if scores.shape != (1, 9) or not torch.isfinite(scores).all():
            raise ValueError('모델 출력은 유한한 (1, 9) 행동 점수여야 합니다.')
        return int(scores.argmax(1).item())





# ===========================================
# ===== 고정 보조 코드: 아래는 수정하지 않습니다 =====
# ===========================================

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def _setup_model(self, device):
    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")
    self.device = device

    self.n_classes = 9
    # [고정] 행동 순서·제어값: 중립, 가속, 제동, 좌, 우, 좌+가속, 우+가속, 좌+제동, 우+제동.
    self.action_classes = [
        (0.0,  0.0, 0.0),
        (0.0,  0.5, 0.0),
        (0.0,  0.0, 0.8),
        (-1.0, 0.0, 0.0),
        (1.0,  0.0, 0.0),
        (-1.0, 0.5, 0.0),
        (1.0,  0.5, 0.0),
        (-1.0, 0.0, 0.8),
        (1.0,  0.0, 0.8),
    ]

    return device

def _fixed_actions_to_classes(self, actions):

    class_indices = []
    for action in actions:
        steer = action[0].item()
        gas   = action[1].item()
        brake = action[2].item()

        is_left  = steer < -0.1
        is_right = steer > 0.1
        is_gas   = gas   > 0.1
        is_brake = brake > 0.1

        if is_left and is_gas:
            cls = 5
        elif is_right and is_gas:
            cls = 6
        elif is_left and is_brake:
            cls = 7
        elif is_right and is_brake:
            cls = 8
        elif is_left:
            cls = 3
        elif is_right:
            cls = 4
        elif is_gas:
            cls = 1
        elif is_brake:
            cls = 2
        else:
            cls = 0

        class_indices.append(torch.tensor([cls], dtype=torch.long))
    return class_indices


def _fixed_scores_to_action(self, scores):

    cls = torch.argmax(scores, dim=1).item()
    return self.action_classes[cls]


def _fixed_extract_sensor_values(self, observation, batch_size):

    speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
    speed = speed_crop.sum(dim=1, keepdim=True) / 255 / 5

    abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
    abs_sensors = abs_crop.sum(dim=1) / 255 / 5

    steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1) / 255 / 10
    steer_crop[:, :10] *= -1
    steering = steer_crop.sum(dim=1, keepdim=True)

    gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1) / 255 / 5
    gyro_crop[:, :14] *= -1
    gyroscope = gyro_crop.sum(dim=1, keepdim=True)

    return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope


# MPS 호환 보조 계층: 풀링 계산은 유지합니다.
class MPSCompatibleAdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool2d):
    def forward(self, x):
        if x.device.type != 'mps':
            return super().forward(x)
        height, width = x.shape[-2:]
        size = self.output_size
        out_h, out_w = (size, size) if isinstance(size, int) else size
        out_h = height if out_h is None else out_h
        out_w = width if out_w is None else out_w
        if out_h <= 0 or out_w <= 0 or height == 0 or width == 0:
            return super().forward(x)
        if height % out_h == 0 and width % out_w == 0:
            return super().forward(x)

        rows = []
        for i in range(out_h):
            start_h = i * height // out_h
            end_h = ((i + 1) * height + out_h - 1) // out_h
            cells = []
            for j in range(out_w):
                start_w = j * width // out_w
                end_w = ((j + 1) * width + out_w - 1) // out_w
                cells.append(x[..., start_h:end_h, start_w:end_w].mean(dim=(-2, -1)))
            rows.append(torch.stack(cells, dim=-1))
        return torch.stack(rows, dim=-2)

