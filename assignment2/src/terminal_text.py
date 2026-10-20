"""스크립트 터미널 메시지의 실행 단위 언어 설정 (표준 라이브러리만 사용)."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import re

_language = ContextVar('terminal_language', default='kor')


def selected_language(path):
    """설치 전에도 읽을 수 있는 최상위 language 설정. YAML 검증은 별도로 한다."""
    try:
        source = Path(path).read_text(encoding='utf-8-sig')
    except OSError:
        return 'kor'
    match = re.search(r'^language:\s*[\'\"]?(kor|eng|chn)[\'\"]?\s*(?:#.*)?$', source, re.M)
    return match.group(1) if match else 'kor'


@contextmanager
def use_language(language):
    token = _language.set(language)
    try:
        yield
    finally:
        _language.reset(token)


def current_language():
    return _language.get()


def tr(message, **values):
    language = current_language()
    template = message if language == 'kor' else TEXT[message][0 if language == 'eng' else 1]
    return template.format(**values)


TEXT = {
    '제출할 구조 파일: {path}': ('Model source to submit: {path}', '待提交的模型结构文件：{path}'),
    '지원하지 않는 체크포인트입니다. 현재 network.py로 scratch 학습하세요.': ('Unsupported checkpoint. Train from scratch with the current network.py.', '不支持的检查点。请使用当前 network.py 从头训练。'),
    'network.py와 함께 저장한 format_version=1 체크포인트가 필요합니다.': ('A format_version=1 checkpoint saved with network.py is required.', '需要与 network.py 一起保存的 format_version=1 检查点。'),
    'network.py가 저장 당시와 다릅니다. 학습은 train.start_model: auto를 사용하거나 추론할 모델의 .network.py를 복원하세요.': ('network.py differs from the saved source. Use train.start_model: auto for training, or restore the model’s .network.py for inference.', 'network.py 与保存时不同。训练请使用 train.start_model: auto，推理请恢复模型对应的 .network.py。'),
    '모델 구조와 가중치가 일치하지 않습니다.': ('Model architecture and weights do not match.', '模型结构与权重不匹配。'),

    '자동 선택: 최신 사용자 모델의 가중치/연산 호환성을 확인하지 못해 처음부터 학습합니다.': ('Auto: weight/computation compatibility could not be verified; training from scratch.', '自动选择：无法确认最新用户模型的权重或运算兼容性，将从头训练。'),
    '참고: 선택하지 않은 GPU 방식 — {reason}': ('Note: GPU backends not selected — {reason}', '说明：未选择的 GPU 后端 — {reason}'),
    'NVIDIA GPU 가속 사용 (CUDA)': ('Using NVIDIA GPU acceleration (CUDA)', '使用 NVIDIA GPU 加速 (CUDA)'),
    'Apple GPU 가속 사용 (MPS / Metal)': ('Using Apple GPU acceleration (MPS / Metal)', '使用 Apple GPU 加速 (MPS / Metal)'),
    'CPU 사용 (GPU 가속 없이 실행)': ('Using CPU (without GPU acceleration)', '使用 CPU（无 GPU 加速）'),
    '동작 확인: GPU 기본 연산 및 역전파 성공': ('Verification: GPU computation and backpropagation succeeded', '验证成功：GPU 基本运算和反向传播正常'),
    'train.start_model은 latest, scratch 또는 모델 경로여야 합니다.': ('train.start_model must be latest, scratch, or a model path.', 'train.start_model 必须为 latest、scratch 或模型路径。'),
    '추가 학습에 사용할 수 없는 모델입니다: {path}': ('Model is not compatible with continued training: {path}', '此模型不支持继续训练：{path}'),
    '최신 학습 모델이 없어 새 모델로 학습을 시작합니다.': ('No trained model found; starting a new model from scratch.', '没有已训练模型，将从头训练新模型。'),
    '새 모델로 처음부터 학습합니다.': ('Training a new model from scratch.', '从头训练新模型。'),
    '기존 모델에서 추가 학습합니다: {path}': ('Continuing training from model: {path}', '从现有模型继续训练：{path}'),
    'GUI 글꼴을 찾을 수 없습니다. Noto Sans CJK를 설치하거나 language: eng로 설정하세요.': ('GUI font missing. Install Noto Sans CJK or set language: eng.', '找不到 GUI 字体。请安装 Noto Sans CJK 或设置 language: eng。'),
    'language는 kor, eng, chn 중 하나여야 합니다.': ('language must be kor, eng, or chn.', 'language 必须为 kor、eng 或 chn。'),
    '추론': ('Inference', '推理'),
    '학습': ('Training', '训练'),
    '사용법: python scripts/{script} configs/config.yaml': ('Usage: python scripts/{script} configs/config.yaml', '用法：python scripts/{script} configs/config.yaml'),
    '실행 오류: {error}': ('Execution error: {error}', '运行错误：{error}'),
    '설치 오류: {error}': ('Installation error: {error}', '安装错误：{error}'),
    'YAML 문법 오류: {path}': ('YAML syntax error: {path}', 'YAML 语法错误：{path}'),
    '설정은 YAML 객체여야 합니다.': ('Configuration must be a YAML mapping.', '配置必须是 YAML 映射。'),
    '{section}에 필요한 설정: {required}': ('Required settings for {section}: {required}', '{section} 所需配置：{required}'),
    '{key}는 비어 있지 않은 경로여야 합니다.': ('{key} must be a nonempty path.', '{key} 必须是非空路径。'),
    '{key}는 양의 정수여야 합니다.': ('{key} must be a positive integer.', '{key} 必须是正整数。'),
    'learning_rate는 양수여야 합니다.': ('learning_rate must be a positive number.', 'learning_rate 必须是正数。'),
    '베이스라인 모델은 덮어쓸 수 없습니다.': ('The baseline model cannot be overwritten.', '不能覆盖基线模型。'),
    '{key}는 true 또는 false여야 합니다.': ('{key} must be true or false.', '{key} 必须为 true 或 false。'),
    'driving 설정 이름을 확인하세요.': ('Check the driving setting names.', '请检查 driving 配置项名称。'),
    'driving.{key}는 유한한 양수여야 합니다.': ('driving.{key} must be finite and positive.', 'driving.{key} 必须是有限正数。'),
    'throttle과 steering_limit은 1 이하여야 합니다.': ('throttle and steering_limit must not exceed 1.', 'throttle 和 steering_limit 不能大于 1。'),
    '학습된 모델이 없습니다: {directory}. 먼저 train.py로 학습하거나 evaluate.model_path를 지정하세요.': ('No trained model found in {directory}. Run train.py first or set evaluate.model_path.', '{directory} 中没有已训练模型。请先运行 train.py 或设置 evaluate.model_path。'),
    '모델이 이미 있습니다. YAML의 model_path를 바꾸세요: {target}': ('Model already exists. Change model_path in YAML: {target}', '模型已存在，请修改 YAML 中的 model_path：{target}'),
    '{name} 사용 불가': ('{name} unavailable', '{name} 不可用'),
    '{name} 연산 확인 실패: {error}': ('{name} computation check failed: {error}', '{name} 运算检查失败：{error}'),
    '[{task}] 장치: {device} ({name}) | PyTorch {version}': ('[{task}] Device: {device} ({name}) | PyTorch {version}', '[{task}] 设备：{device} ({name}) | PyTorch {version}'),
    '장치 선택: {reason}': ('Device selection: {reason}', '设备选择：{reason}'),
    '설정: {settings}': ('Settings: {settings}', '配置：{settings}'),
    '잘못된 배열 크기': ('Invalid array shape', '数组形状无效'),
    'NaN 또는 무한대': ('NaN or infinity detected', '检测到 NaN 或无穷值'),
    '픽셀 범위 오류': ('Pixel values are out of range', '像素值超出范围'),
    '행동 범위 오류': ('Action values are out of range', '动作值超出范围'),
    '학습 데이터 기준 폴더: {root}': ('Training data root: {root}', '训练数据根目录：{root}'),
    '전체 학습 대상: 관측 파일 {count}개 (각 파일과 같은 번호의 행동 파일 사용)': ('Training total: {count} observation files (paired with action files of the same number)', '训练总计：{count} 个观测文件（使用编号相同的动作文件）'),
    '하위 폴더를 포함하여 다음 폴더의 데이터를 사용합니다:': ('Using data from the following folders, including subfolders:', '使用以下目录（含子目录）中的数据：'),
    '  - {folder} (관측 파일 {count}개)': ('  - {folder} ({count} observation files)', '  - {folder}（{count} 个观测文件）'),
    '학습 데이터가 없습니다: {folder}. 먼저 collect로 수집하세요.': ('No training data found in {folder}. Run collect first.', '{folder} 中没有训练数据。请先运行 collect 采集数据。'),
    '데이터 {count}쌍 로드: {folder}': ('Loaded {count} data pairs: {folder}', '已加载 {count} 对数据：{folder}'),
    '관측과 행동의 개수가 다릅니다.': ('Observation and action counts do not match.', '观测和动作的数量不一致。'),
    '{count}쌍 저장: {root}': ('Saved {count} data pairs: {root}', '已保存 {count} 对数据：{root}'),
    'GUI 화면이 없습니다. 데스크톱 터미널에서 실행하거나 SSH X11 전달을 사용하세요.': ('No GUI display available. Run from a desktop terminal or use SSH X11 forwarding.', '没有可用的图形显示。请从桌面终端运行或使用 SSH X11 转发。'),
    '창 클릭 → Enter 시작 | 방향키 운전 | 복귀 후 Tab 저장 | Space 폐기/다시 시작 | Esc 종료': ('Click the window → Enter: Start | Arrow keys: Drive | Tab: Save after lap | Space: Discard/restart | Esc: Exit', '点击窗口 → Enter 开始 | 方向键驾驶 | 完成一圈后 Tab 保存 | Space 丢弃/重新开始 | Esc 退出'),
    '종료: 저장하지 않은 {count}프레임은 폐기합니다.': ('Exiting: discarding {count} unsaved frames.', '退出：丢弃 {count} 帧未保存数据。'),
    '한 바퀴 완료 후 저장할 수 있습니다.': ('Complete a lap before saving.', '完成一圈后才能保存。'),
    '{count}프레임 폐기': ('Discarded {count} frames', '已丢弃 {count} 帧数据'),
    '주행 중단: 영역을 벗어났습니다. Space 폐기 / Esc 종료': ('Driving stopped: outside the driving area. Space: Discard / Esc: Exit', '驾驶已停止：驶出驾驶区域。Space 丢弃 / Esc 退出'),
    '한 바퀴 완료: 주행 점수 {score:.2f} / 1000. Tab 저장 / Space 폐기': ('Lap complete: driving score {score:.2f} / 1000. Tab: Save / Space: Discard', '已完成一圈：驾驶得分 {score:.2f} / 1000。Tab 保存 / Space 丢弃'),
    '주행 중단': ('Driving stopped', '驾驶已停止'),
    '주행 종료: 출발점 복귀 완료': ('Driving complete: crossed the start/finish line', '驾驶完成：已通过起终点线'),
    '주행 점수: {score:.2f} / 1000 | 진행률: {progress:.1%} | 시간: {seconds:.2f}초': ('Driving score: {score:.2f} / 1000 | Progress: {progress:.1%} | Time: {seconds:.2f}s', '驾驶得分：{score:.2f} / 1000 | 进度：{progress:.1%} | 时间：{seconds:.2f}秒'),
    '기존 누적 보상: {reward:.2f}': ('Original cumulative reward: {reward:.2f}', '原始累计奖励：{reward:.2f}'),
    '주행 창을 닫아 평가를 종료합니다.': ('Closing the driving window; ending evaluation.', '关闭驾驶窗口，结束评估。'),
    '주행 {index}: 주행 점수 {score:.2f} / 1000': ('Run {index}: driving score {score:.2f} / 1000', '第 {index} 次驾驶：驾驶得分 {score:.2f} / 1000'),
    '평균 주행 점수: {score:.2f} / 1000': ('Mean driving score: {score:.2f} / 1000', '平均驾驶得分：{score:.2f} / 1000'),
    '기존 평균 점수: {score:.2f}': ('Original mean score: {score:.2f}', '原始平均得分：{score:.2f}'),
    'Epoch {epoch:5d}\t[Train]\tloss: {loss:.6f} \tETA: +{seconds:f}s': ('Epoch {epoch:5d}\t[Train]\tloss: {loss:.6f} \tETA: +{seconds:f}s', '轮次 {epoch:5d}\t[训练]\t损失：{loss:.6f} \t预计剩余：{seconds:f}秒'),
    '모델 저장 완료: {target}': ('Model saved: {target}', '模型已保存：{target}'),
    'Python 3.12로 실행하세요.': ('Run with Python 3.12.', '请使用 Python 3.12 运行。'),
    '사용 불가': ('unavailable', '不可用'),
    '환경 감지: {system} {machine} | NVIDIA 드라이버 CUDA: {cuda}': ('Detected environment: {system} {machine} | NVIDIA driver CUDA: {cuda}', '检测到环境：{system} {machine} | NVIDIA 驱动 CUDA：{cuda}'),
    'PyTorch 설치 시도: {backend}': ('Installing PyTorch: {backend}', '正在尝试安装 PyTorch：{backend}'),
    '환경 설치 완료': ('Environment setup complete', '环境安装完成'),
    '{backend} 설치 실패 (종료 코드 {code})': ('{backend} installation failed (exit code {code})', '{backend} 安装失败（退出码 {code}）'),
    '다음 호환 설치 후보를 시도합니다.': ('Trying the next compatible installation option.', '正在尝试下一个兼容的安装选项。'),
    '호환 PyTorch를 설치하지 못했습니다.': ('Could not install a compatible PyTorch version.', '未能安装兼容的 PyTorch 版本。'),
    '설치 확인:': ('Installation check:', '安装检查：'),
    'CUDA 연산을 사용할 수 없습니다': ('CUDA computation is unavailable', 'CUDA 运算不可用'),
    '파일을 찾을 수 없습니다: {path}': ('File not found: {path}', '找不到文件：{path}'),
}

# Three independent driving metrics replace the legacy combined score.
TEXT.update({
    '주행 결과': ('Driving results', '驾驶结果'),
    '주행 {index}': ('Run {index}', '第 {index} 次驾驶'),
    '미인정 (미완주)': ('Invalid (unfinished)', '无效（未完成）'),
    '인정 기록 없음': ('No valid lap', '无有效记录'),
    '{label}: 진행률 {progress:.2f}/100 | 중앙선 유지 {center:.2f}/100 | 랩타임 {record}':
        ('{label}: Progress: {progress:.2f}/100 | Lane centering: {center:.2f}/100 | Lap time: {record}',
         '{label}：进度 {progress:.2f}/100 | 中心线保持 {center:.2f}/100 | 单圈时间 {record}'),
    '평균 진행률: {score:.2f} / 100': ('Mean progress: {score:.2f} / 100', '平均进度：{score:.2f} / 100'),
    '평균 중앙선 유지: {score:.2f} / 100': ('Mean lane centering: {score:.2f} / 100', '平均中心线保持：{score:.2f} / 100'),
    '완주 랩타임: {records}': ('Completed lap times: {records}', '完成单圈时间：{records}'),
})

TEXT.update({
    '주행 종료: 차선에 5초 동안 복귀하지 못했습니다.': ('Drive ended: failed to return to the lane within 5 seconds.', '驾驶结束：5秒内未能返回车道。'),
    '주행 종료: 차선에 5초 동안 복귀하지 못했습니다. Space 폐기 / Esc 종료': ('Drive ended: no lane recovery within 5 seconds. Space: discard / Esc: exit', '驾驶结束：5秒内未返回车道。Space 丢弃 / Esc 退出'),
    '데이터 수집을 시작합니다.': ('Starting data collection.', '开始采集数据。'),
    '수집한 데이터로 모델을 학습합니다.': ('Training the model on collected data.', '使用采集的数据训练模型。'),
    '학습된 모델을 테스트합니다.': ('Testing the trained model.', '测试训练后的模型。'),
    '학습된 모델 파일 경로': ('Trained model file path', '训练模型文件路径'),
    '학습 모델 저장 경로': ('Model output path', '模型保存路径'),
    '학습 데이터 경로': ('Training data path', '训练数据路径'),
    '학습 횟수': ('Training epochs', '训练轮数'),
    '학습률': ('Learning rate', '学习率'),
    '출력 언어 (kor / eng / chn)': ('Output language (kor / eng / chn)', '输出语言 (kor / eng / chn)'),
})

TEXT.update({
    '사용 가능한 모델이 없습니다: {directory}. baseline.pth를 넣거나 모델 경로를 지정하세요.':
        ('No available model in {directory}. Add baseline.pth or specify a model path.',
         '{directory} 中没有可用模型。请放入 baseline.pth 或指定模型路径。'),
    '학습 모델과 baseline.pth가 없어 새 모델로 학습을 시작합니다.':
        ('No trained model or baseline.pth found; starting training from scratch.',
         '没有训练模型或 baseline.pth，将从头开始训练。'),
})


TEXT.update({
    '비교할 저장 소스가 없습니다: {path}':
        ('Saved source for comparison is missing: {path}', '缺少用于比较的已保存源代码：{path}'),
    '저장 소스와 체크포인트의 해시가 다릅니다: {path}':
        ('Saved source hash does not match the checkpoint: {path}', '已保存源代码的哈希与检查点不匹配：{path}'),
    '연산 비교 실패 (종료 코드 {code}): {detail}':
        ('Computation comparison failed (exit code {code}): {detail}', '运算比较失败（退出码 {code}）：{detail}'),
    '연산 비교 제한 시간(60초)을 초과했습니다.':
        ('Computation comparison exceeded the 60-second timeout.', '运算比较超过了60秒时限。'),
    '호환성 검사 실패 원인: {reason}':
        ('Compatibility check failure: {reason}', '兼容性检查失败原因：{reason}'),
})
