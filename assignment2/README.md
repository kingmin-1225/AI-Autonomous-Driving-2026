# 과제 2: Reinforcement Learning

데이터 수집/추론과 서버 채점은 모두 **동일한 고정 트랙 1개**(시드 `22597174`)를 사용합니다.

**해야 할 일은 네 가지입니다.**

1. 아래 명령어로 **환경을 설치**합니다.
2. **[src/network.py](src/network.py)의 모델 구조, 보상함수, 탐색 전략을 개선**합니다.
3. **데이터 수집 → 모델 학습 → 모델 추론** 명령어를 실행합니다.
4. **제출 사이트에 `network.py`와 `.pth`를 제출**합니다.

제출 장소: https://drive.hyuaisl.com/?assignment=rl

## 1. 환경 설치 — 최초 한 번

과제 1에서 설치한 가상환경(`assignment1/.venv/`)을 그대로 사용합니다.

모든 명령은 **`assignment2/` 폴더에서** 실행하세요.

**Ubuntu / macOS**

```bash
source ../assignment1/.venv/bin/activate
```

**Windows / PowerShell**

```powershell
$env:Path = "$PWD\..\assignment1\.venv\Scripts;$env:Path"
```

과제 1의 가상환경이 없다면 Python 3.12, 64비트 환경에서 아래 명령어로 새로 설치하세요.

<details>
<summary>Windows / PowerShell</summary>

Python 3.12가 없다면 설치 후 PowerShell을 다시 여세요.

```powershell
winget install --exact --id Python.Python.3.12 --source winget
```

```powershell
py -3.12 -m venv ..\assignment1\.venv
..\assignment1\.venv\Scripts\python.exe -m pip install --upgrade pip
..\assignment1\.venv\Scripts\python.exe scripts/setup.py configs/config.yaml
$env:Path = "$PWD\..\assignment1\.venv\Scripts;$env:Path"
```

Windows ARM64는 Visual Studio 2022 Build Tools의 C++ 데스크톱 개발, MSVC ARM64/ARM64EC 빌드 도구와 Windows 11 SDK가 추가로 필요합니다.

</details>

<details>
<summary>macOS / 터미널 (Homebrew 설치 필요)</summary>

```bash
brew install python@3.12
python3.12 -m venv ../assignment1/.venv
../assignment1/.venv/bin/python -m pip install --upgrade pip
../assignment1/.venv/bin/python scripts/setup.py configs/config.yaml
source ../assignment1/.venv/bin/activate
```

</details>

<details>
<summary>Ubuntu 24.04 / 터미널</summary>

```bash
sudo apt update
sudo apt install -y python3.12-venv python3.12-dev build-essential libgl1 libsdl2-2.0-0
python3.12 -m venv ../assignment1/.venv
../assignment1/.venv/bin/python -m pip install --upgrade pip
../assignment1/.venv/bin/python scripts/setup.py configs/config.yaml
source ../assignment1/.venv/bin/activate
```

</details>

## 2. 모델 구조, 보상함수 및 탐색 전략 개선

[src/network.py](src/network.py)의 **구현 내용**을 수정하세요.

- **모델 구조:** `ClassificationNetwork` — 상태를 입력받아 각 행동의 Q값을 계산하는 신경망
- **보상함수:** `REWARD_SETTINGS`, `calculate_reward()` — 주행 결과에 따라 보상을 계산하는 방법
- **탐색 전략:** `collection_action()` — 데이터 수집 중 행동을 선택하는 방법 (기본 ε-greedy)

**학습 알고리즘은 Double DQN으로 고정**되어 있으며, 제공된 학습 코드를 그대로 사용합니다.

입력 `(B, 96, 96, 3)`, 출력 `(B, 9)`와 제공된 행동 순서, 함수 이름 및 인자는 유지하세요. `network.py` 하단의 고정 보조 코드는 수정하지 않습니다.

보상함수를 변경한 경우에는 **별도 데이터 폴더에 새로 수집**해야 합니다. 서버 채점 기준은 변경되지 않습니다.

## 3. 데이터 수집 → 모델 학습 → 모델 추론

가상환경을 활성화한 뒤 아래 명령어를 순서대로 실행하세요. Windows / macOS / Ubuntu 공통입니다.

```bash
# 1) 데이터 수집
python scripts/collect.py configs/config.yaml

# 2) 모델 학습
python scripts/train.py configs/config.yaml

# 3) 모델 추론 — 주행 확인
python scripts/infer.py configs/config.yaml
```

**참고사항**

- **데이터 수집:** 기본적으로 ε-greedy 전략을 사용하며, 90%는 현재 모델의 최선 행동, 10%는 무작위 행동을 선택합니다. 저장된 모델이 없으면 baseline을 사용합니다. 탐색 비율은 `collect.epsilon`에서 조절할 수 있습니다.
- **데이터 수집 조작:** Space는 저장 후 다음 에피소드 진행, Esc 또는 창 닫기는 저장 후 종료합니다.
- **모델 학습:** `data/`의 완료된 에피소드를 사용합니다. 학습 장치는 CUDA → MPS → CPU 순으로 자동 선택됩니다.
- **모델 추론:** 학습된 Q값이 가장 높은 행동을 선택하며, 무작위 탐색은 사용하지 않습니다.
- **성능 비교:** 동일한 모델 구조에서는 새 모델의 성능이 기존보다 나쁘면 기존 가중치를 유지합니다. 학습 후보는 `models/candidates/`에 저장됩니다.

**주의사항:** 모델 구조를 변경하면 재학습이 필요하며, 보상함수나 탐색 전략을 변경하면 데이터를 새로 수집해야 합니다. 데이터 수집 시에는 기존에 저장된 모델이 사용될 수 있습니다.

## 4. 과제 제출

**제출 사이트**에 실명으로 가입/로그인합니다. 실명과 비밀번호로 바로 가입할 수 있습니다.

- **사용자 코드:** 학습할 때 자동 저장된 `models/날짜.network.py`
- **모델 가중치:** 같은 이름의 `models/날짜.pth`

코드 파일은 `network.py`의 저장 당시 사본입니다. **같은 학습에서 생성된 두 파일을 함께 제출**하세요.

학습 후 수정한 현재 `src/network.py`를 이전 `.pth`와 섞어 제출하면 평가에 실패합니다.

제출 후 사이트에서 대기 상태, 주행 영상, 평가 지표와 순위를 확인할 수 있습니다. 채점 장치는 제출 화면에서 선택하며, 기본값은 자동(GPU 우선)입니다.
