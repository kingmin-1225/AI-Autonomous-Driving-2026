"""환경 설치: python scripts/setup.py configs/config.yaml"""
from pathlib import Path
import sys
import subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from install_environment import install
from terminal_text import tr, use_language, selected_language

if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).resolve().parents[1] / 'configs/config.yaml'
    with use_language(selected_language(path)):
        if len(sys.argv) != 2:
            raise SystemExit(tr('사용법: python scripts/{script} configs/config.yaml', script='setup.py'))
        try:
            install(path)
        except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
            raise SystemExit(tr('설치 오류: {error}', error=error)) from error
