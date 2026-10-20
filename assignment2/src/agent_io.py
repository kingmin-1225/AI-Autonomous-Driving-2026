"""최신 agent는 저장 당시 구조와 함께 로드한다. 추론 장치는 사용자가 선택할 수 있다."""
import hashlib
import importlib.util
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from datetime import datetime
from terminal_text import tr

PROTOCOL = 'driving-50hz-rl-v2'


def inference_runtime(seed=22597174, device="auto", *, warn_only=False):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device not in ('auto','cpu','cuda','mps'):raise ValueError('Invalid device')
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    return torch.device(device)


def load_agent(path, device='cpu'):
    path = Path(path)
    source = path.with_suffix('.network.py')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format_version') != 1 or payload.get('protocol') != PROTOCOL:
        raise ValueError(f'현재 RL 프로토콜의 agent가 아닙니다: {path}')
    if hashlib.sha256(source.read_bytes()).hexdigest() != payload['network_sha256']:
        raise ValueError(f'가중치와 구조 파일이 다릅니다: {source}')
    spec = importlib.util.spec_from_file_location('saved_agent', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ClassificationNetwork(device=device).to(device).eval()
    model.load_state_dict(payload['state_dict'], strict=True)
    with torch.inference_mode():
        sample=torch.zeros((1,96,96,3),device=device,dtype=torch.float32)
        for _ in range(3):probe=model(sample)
        if tuple(probe.shape)!=(1,9) or not torch.isfinite(probe).all():
            raise ValueError('모델 출력은 유한한 (1, 9) 행동 가치여야 합니다.')
    return model, payload


def save_agent(model, directory, source, metadata=None, target=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = Path(target) if target else timestamp_path(directory)
    content = Path(source).read_bytes()
    snapshot = path.with_suffix('.network.py')
    payload = dict(format_version=1, protocol=PROTOCOL,
                   network_sha256=hashlib.sha256(content).hexdigest(),
                   state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                   metadata=metadata or {})
    with snapshot.open('xb') as f:
        f.write(content)
    with tempfile.NamedTemporaryFile(dir=directory, suffix='.tmp', delete=False) as f:
        temporary = Path(f.name)
    try:
        torch.save(payload, temporary)
        # Publish only a complete pair; no overwrites, even on duplicate names.
        os.link(temporary, path)
    except BaseException:
        snapshot.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return path


def select_action(model, observation, device='cpu', details=None):
    with torch.inference_mode():
        scores = model(torch.as_tensor(np.array(observation[None], copy=True), dtype=torch.float32, device=device))
    if tuple(scores.shape) != (1, 9) or not torch.isfinite(scores).all():
        raise ValueError('모델 출력은 유한한 (1, 9) 행동 가치여야 합니다.')
    action = int(scores.argmax(1).item())
    if details is not None:
        details.update(source="model", action=action)
    return action


def collect_action(model, observation, device, epsilon, rng, details=None):
    """Uniform exploration over the same nine actions; evaluation stays greedy."""
    if not 0. <= epsilon <= 1.:
        raise ValueError('collect.epsilon must be between 0 and 1')
    if epsilon > 0. and rng.random() < epsilon:
        action = int(rng.integers(9))
        if details is not None:
            details.update(source="random", action=action)
        return action
    if details is not None:
        return select_action(model, observation, device, details=details)
    return select_action(model, observation, device)


class ActionCounts:
    """Display completed one-second blocks of 50 simulation decisions."""
    def __init__(self):
        self.pending = [0] * 9
        self.display = [0] * 9
        self.steps = 0

    def add(self, action):
        self.pending[action] += 1
        self.steps += 1
        if self.steps == 50:
            self.display = self.pending.copy()
            self.pending = [0] * 9
            self.steps = 0
        return self.display


def model_definition(source):
    """Compare model code while allowing independent learning/reward edits."""
    import ast
    tree=ast.parse(source)
    # Expand the fixed helpers so a layout-only refactor still resumes old weights.
    import copy
    helpers = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in {'_setup_model', '_fixed_actions_to_classes',
                                 '_fixed_scores_to_action', '_fixed_extract_sensor_values'}}
    model = next((node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == 'ClassificationNetwork'), None)
    expanded = set()
    if model:
        for method in model.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            helper = helpers.get('_fixed_' + method.name)
            if helper and len(method.body) == 1:
                expected = ast.parse(f'def {method.name}(' + ', '.join(a.arg for a in helper.args.args)
                                     + '):\n    return ' + helper.name + '(' +
                                     ', '.join(a.arg for a in helper.args.args) + ')').body[0]
                if ast.dump(method) == ast.dump(expected):
                    method.body = copy.deepcopy(helper.body)
                    expanded.add(helper.name)
            if method.name == '__init__' and '_setup_model' in helpers:
                helper = helpers['_setup_model']
                expected = ast.parse('device = _setup_model(self, device)').body[0]
                if (ast.dump(helper.args) == ast.dump(ast.parse('def f(self, device): pass').body[0].args)
                        and ast.dump(helper.body[-1]) == ast.dump(ast.parse('return device').body[0])):
                    body = []
                    for node in method.body:
                        if ast.dump(node) == ast.dump(expected):
                            body.extend(copy.deepcopy(helper.body[:-1]))
                            expanded.add(helper.name)
                        else:
                            body.append(node)
                    method.body = body
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.FunctionDef) and node.name in expanded)]
    # These standalone setup statements/classes may appear above or below the model.
    fixed_assignments = {ast.dump(ast.parse(line).body[0]) for line in (
        'torch.backends.cuda.matmul.allow_tf32 = False',
        'torch.backends.cudnn.allow_tf32 = False')}
    setup = [node for node in tree.body if ast.dump(node) in fixed_assignments]
    pools = [node for node in tree.body if isinstance(node, ast.ClassDef)
             and node.name == 'MPSCompatibleAdaptiveAvgPool2d']
    rest = [node for node in tree.body if node not in setup + pools]
    imports = [node for node in rest if isinstance(node, (ast.Import, ast.ImportFrom))]
    tree.body = imports + setup + pools + [node for node in rest if node not in imports]
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                node.body.pop(0)
    if not any(isinstance(node,ast.Name) and node.id=='np' for node in ast.walk(tree)):
        tree.body=[node for node in tree.body if not (isinstance(node,ast.Import) and len(node.names)==1 and node.names[0].name=='numpy' and node.names[0].asname=='np')]
    editable={'LearningAlgorithm','calculate_reward','REWARD_SETTINGS','collection_action'}
    def training_only(node):
        return (isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in editable or
                isinstance(node,ast.Assign) and all(isinstance(t,ast.Name) and t.id in editable for t in node.targets))
    retained=[node for node in tree.body if not training_only(node)]
    # If the model references these symbols, they are part of its definition too.
    if not any(isinstance(node,ast.Name) and node.id in editable for item in retained for node in ast.walk(item)):
        tree.body=retained
    return ast.dump(tree)


def timestamp_path(directory):
    directory = Path(directory)
    name = datetime.now().strftime('%Y%m%d_%H%M%S')
    target = directory / f'{name}.pth'
    index = 1
    while target.exists():
        target = directory / f'{name}_{index}.pth'
        index += 1
    return target


def latest_model(directory, allow_baseline=True, prefer_trained=True):
    candidates = [p for p in Path(directory).glob('*.pth')
                  if p.is_file() and (p.name != 'baseline.pth' or (allow_baseline and not prefer_trained))]
    if not candidates:
        baseline = Path(directory) / 'baseline.pth'
        if allow_baseline and baseline.is_file():
            return baseline
        raise FileNotFoundError(tr('사용 가능한 모델이 없습니다: {directory}. baseline.pth를 넣거나 모델 경로를 지정하세요.', directory=directory))
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))
