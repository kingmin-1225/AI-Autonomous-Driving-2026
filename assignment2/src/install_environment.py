"""표준 라이브러리만으로 실행하는 플랫폼별 PyTorch 설치기."""
from pathlib import Path
import os
import platform
import re
import subprocess
import sys
import sysconfig
from terminal_text import tr, current_language

ROOT = Path(__file__).resolve().parents[1]
TORCH_VERSION = '2.5.1'
CPU_INDEX = 'https://download.pytorch.org/whl/cpu'
WIN_ARM_WHEEL = 'https://download.pytorch.org/whl/cpu/torch-2.10.0%2Bcpu-cp312-cp312-win_arm64.whl'


def cuda_version():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', result.stdout)
    return tuple(map(int, match.groups())) if result.returncode == 0 and match else None


def candidates(system, machine, cuda):
    machine = machine.lower()
    x64 = machine in ('amd64', 'x86_64')
    if cuda and cuda >= (12, 8) and (x64 or (system == 'Linux' and machine in ('arm64', 'aarch64'))):
        yield 'cuda', ['torch==2.10.0+cu128', '--index-url', 'https://download.pytorch.org/whl/cu128']
    if system in ('Linux', 'Windows') and x64 and cuda:
        for minimum, tag in [((12, 4), 'cu124'), ((12, 1), 'cu121'), ((11, 8), 'cu118')]:
            if cuda >= minimum:
                yield 'cuda', [f'torch=={TORCH_VERSION}+{tag}', '--index-url', f'https://download.pytorch.org/whl/{tag}']
    if system == 'Darwin':
        # macOS 공식 wheel 하나가 CPU와 MPS를 함께 지원한다.
        yield 'mps' if not x64 else 'cpu', ['torch==2.2.2' if x64 else f'torch=={TORCH_VERSION}']
    elif system == 'Windows' and machine == 'arm64':
        yield 'cpu', [WIN_ARM_WHEEL]
    elif x64:
        yield 'cpu', [f'torch=={TORCH_VERSION}+cpu', '--index-url', CPU_INDEX]
    else:
        yield 'cpu', [f'torch=={TORCH_VERSION}']


def pip_install(arguments):
    subprocess.run([sys.executable, '-m', 'pip', 'install', *arguments], check=True)


def verify_backend(backend):
    code = ("from terminal_text import use_language, tr; "
            f"context=use_language({current_language()!r}); context.__enter__(); "
            "from install_environment import select_device; d,r=select_device(); print(tr('설치 확인:'), d, r); ")
    if backend == 'cuda':
        code += "assert d.type == 'cuda', tr('CUDA 연산을 사용할 수 없습니다')"
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT / 'src')
    return result.returncode == 0


def install(config_path):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(tr('Python 3.12로 실행하세요.'))
    if not config_path.is_file():
        raise FileNotFoundError(tr('파일을 찾을 수 없습니다: {path}', path=config_path))
    os.environ['PATH'] = sysconfig.get_path('scripts') + os.pathsep + os.environ.get('PATH', '')
    system, machine = platform.system(), platform.machine()
    cuda = cuda_version()
    print(tr('환경 감지: {system} {machine} | NVIDIA 드라이버 CUDA: {cuda}', system=system, machine=machine, cuda=cuda or tr('사용 불가')), flush=True)
    pip_install(['setuptools==78.1.0', 'wheel==0.45.1', 'swig==4.5.0'])
    pip_install(['--no-build-isolation', '-r', str(ROOT / 'requirements.txt')])
    for backend, arguments in candidates(system, machine, cuda):
        print(tr('PyTorch 설치 시도: {backend}', backend=backend.upper()), flush=True)
        try:
            pip_install(arguments)
            if verify_backend(backend):
                print(tr('환경 설치 완료'), flush=True)
                return
        except subprocess.CalledProcessError as error:
            print(tr('{backend} 설치 실패 (종료 코드 {code})', backend=backend.upper(), code=error.returncode), flush=True)
        print(tr('다음 호환 설치 후보를 시도합니다.'), flush=True)
    raise RuntimeError(tr('호환 PyTorch를 설치하지 못했습니다.'))


def probe_device(name):
    import torch
    # 드라이버/아키텍처 호환성을 실제 연산으로 확인한다.
    x = torch.ones((8, 8), device=name, requires_grad=True)
    (x @ x).sum().backward()
    if name == 'cuda':
        torch.cuda.synchronize()
    elif name == 'mps':
        torch.mps.synchronize()


def select_device():
    import torch
    reasons = []
    for name, available in [('cuda', torch.cuda.is_available()),
                            ('mps', torch.backends.mps.is_available())]:
        if not available:
            reasons.append(tr('{name} 사용 불가', name=name.upper()))
            continue
        try:
            probe_device(name)
            return torch.device(name), '; '.join(reasons)
        except (RuntimeError, AssertionError, NotImplementedError) as error:
            reasons.append(tr('{name} 연산 확인 실패: {error}', name=name.upper(), error=error))
    return torch.device('cpu'), '; '.join(reasons)


