import torch
import numpy as np

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False


class MPSCompatibleAdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool2d):
    """MPS에서도 원래 adaptive pooling의 영역과 평균을 유지한다."""

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

        # 시작=floor(i*N/M), 끝=ceil((i+1)*N/M). 경계의 중첩도 원본과 같다.
        # CPU 이동이나 detach 없이 평균을 계산하므로 MPS 역전파가 유지된다.
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


class ClassificationNetwork(torch.nn.Module):
    def __init__(self, device=None):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.

        공개된 작은 starter 구조를 개선하는 과제입니다.
        모델은 입력 (B, 96, 96, 3)에 대해 출력 (B, 9)를 반환해야 합니다.
        """
        super().__init__()

        # Setting device on GPU if available, else CPU.
        device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device

        # --- Action class definitions ---
        # 9 mutually exclusive classes that cover all meaningful combinations
        # of the three keyboard controls [steer, gas, brake].
        self.n_classes = 9
        self.action_classes = [
            (0.0,  0.0, 0.0),   # 0: do nothing
            (0.0,  0.5, 0.0),   # 1: accelerate
            (0.0,  0.0, 0.8),   # 2: brake
            (-1.0, 0.0, 0.0),   # 3: steer left
            (1.0,  0.0, 0.0),   # 4: steer right
            (-1.0, 0.5, 0.0),   # 5: steer left + accelerate
            (1.0,  0.5, 0.0),   # 6: steer right + accelerate
            (-1.0, 0.0, 0.8),   # 7: steer left + brake
            (1.0,  0.0, 0.8),   # 8: steer right + brake
        ]

        # ===== TODO: (구현 영역) 아래 conv / fc를 개선하세요 =====
        # 이 starter는 비교 기준으로 제공하는 작은 CNN입니다.
        # 아래 두 블록을 개선하여 영상에서 주행에 필요한 특징을 학습해 보세요.
        # 단순히 층을 늘린다고 성능이 좋아지는 것은 아닙니다. 주행 결과로 비교하세요.
        #
        # [전체 데이터 흐름]
        # B는 한 번에 처리하는 영상 수(batch size)이며, 추론할 때는 보통 1입니다.
        # 원본 입력 (B, 96, 96, 3)은 forward()에서 픽셀을 0~1로 정규화하고
        # 채널 순서를 바꾸어 (B, 3, 96, 96)으로 만든 뒤 self.conv에 전달합니다.
        # conv 출력 → 펼치기(flatten) → 센서 7개 결합 → self.fc → 행동 9개 점수
        # 펼치기와 센서 결합은 아래 forward()에서 이미 처리합니다.
        #
        # [1. 영상 특징 추출: self.conv]
        # Conv2d(입력 채널, 출력 채널, kernel_size, stride)
        # - 입력 채널 3: RGB 색상 채널 수입니다.
        # - 출력 채널 8: 서로 다른 특징을 추출하는 필터 8개를 학습합니다.
        # - kernel_size=5: 5×5 영역을 보고 특징을 계산합니다.
        # - stride=4: 필터를 4픽셀 간격으로 이동하므로 공간 해상도가 줄어듭니다.
        # 현재 padding=0, dilation=1이므로 출력 한 변은
        # floor((96 - 5) / 4) + 1 = 23입니다. 즉 (B, 8, 23, 23)이 됩니다.
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(16), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            MPSCompatibleAdaptiveAvgPool2d((4, 4)),
        )
        #
        # [2. 행동 분류: self.fc]
        # forward()가 conv 출력을 펼치면 영상당 8×4×4 = 128개 특징이 됩니다.
        # 여기에 영상 하단 계기판에서 추출한 센서 특징 7개를 붙입니다.
        # 센서 7개 = 속도 1개 + 바퀴별 ABS 4개 + 조향 1개 + 자이로 1개
        # 따라서 첫 Linear의 입력 크기는 128 + 7 = 135입니다.
        # 출력 (B, 9)의 각 값은 위 action_classes 순서에 해당하는 행동 점수(logit)입니다.
        # 학습의 CrossEntropyLoss가 logits를 받으므로 마지막에 Softmax를 넣지 마세요.
        # 추론에서는 가장 큰 점수의 행동을 선택합니다.
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4 + 7, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, self.n_classes),
        )
        #
        # [수정할 때 확인할 것]
        # - Conv 층을 추가하면 앞 층의 출력 채널 = 다음 층의 입력 채널이어야 합니다.
        # - 최종 conv 출력이 (B, C, H, W)라면 첫 Linear 입력은 C*H*W + 7입니다.
        #   예: 최종 채널을 16으로 바꾸고 4×4 풀링을 유지하면 16*4*4 + 7입니다.
        # - FC 중간층을 추가하면 앞 Linear의 출력 크기와 다음 입력 크기를 맞추세요.
        # - 마지막 출력은 self.n_classes(9)를 유지하고 행동 클래스 순서는 바꾸지 마세요.
        # - 층/채널이 많아지면 연산량과 메모리 사용도 늘어납니다.
        # - train.start_model: auto를 유지하면 구조/연산 변경 시 처음부터 학습합니다.
        # ===== (구현 영역) 끝 =====

        self.to(device)

    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, C)

        All 3 color channels are kept: the green grass vs. grey road boundary
        is a strong cue for staying on track that would be lost in grayscale.
        """
        batch_size = observation.shape[0]

        # Extract scalar sensor readings from the HUD status bar.
        # The detach() prevents autograd issues caused by the in-place sign-flip
        # inside extract_sensor_values, since we do not need gradients here.
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(
            observation.detach(), batch_size)

        # Normalise image pixels to [0, 1] and convert channels-last to
        # channels-first (B, H, W, C) -> (B, C, H, W) as required by Conv2d.
        x = observation / 255.0
        # MPS (PyTorch 2.5.1) convolution backward requires contiguous NCHW
        # storage; permute alone leaves a channels-last memory layout.
        x = x.permute(0, 3, 1, 2).contiguous()

        # CNN 출력을 펼친 뒤 센서 특징 7개를 연결한다.
        x = self.conv(x)
        x = x.reshape(batch_size, -1)

        # Append sensor features and run through FC classifier.
        x = torch.cat([x, speed, abs_sensors, steering, gyroscope], dim=1)
        return self.fc(x)

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Every action is represented by a 1-dim vector
        with the entry corresponding to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size 1

        Thresholds (|steer| > 0.1, gas > 0.1, brake > 0.1) separate the
        discrete keyboard inputs captured in the expert demonstrations.
        """
        class_indices = []
        for action in actions:
            steer = action[0].item()
            gas   = action[1].item()
            brake = action[2].item()

            is_left  = steer < -0.1
            is_right = steer > 0.1
            is_gas   = gas   > 0.1
            is_brake = brake > 0.1

            # Combined actions take priority over individual ones.
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

    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
                        C = number of classes
        scores:         torch.Tensor of size (batch_size, C)
        return          (float, float, float)
        """
        cls = torch.argmax(scores, dim=1).item()
        return self.action_classes[cls]

    def extract_sensor_values(self, observation, batch_size):
        # just approximately normalized, usually this suffices.
        # can be changed by you
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
