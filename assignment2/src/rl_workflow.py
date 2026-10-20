"""수집 → 고정 Double DQN 학습 → 모델 검증·저장·추론."""
from pathlib import Path
import hashlib
import secrets
import json
import time
import numpy as np
import torch
from agent_io import inference_runtime, load_agent, save_agent, select_action, collect_action, ActionCounts, model_definition
from agent_io import latest_model
from replay import EpisodeWriter, dataset_files, completed_episode_count
from rl_env import RLDrivingEnv
from track import TRACK_SEED


def prepare_gui():
    """Prepare SDL for a native desktop window."""
    import os
    import sys
    from terminal_text import tr
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    if os.environ.get('SDL_VIDEODRIVER') == 'dummy':
        os.environ.pop('SDL_VIDEODRIVER')
    if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError(tr('GUI 화면이 없습니다. 데스크톱 터미널에서 실행하거나 SSH X11 전달을 사용하세요.'))


def run_driving(root, config, collect=False, headless=False, episodes=None, max_steps=30000, delay=0.):
    device=inference_runtime(device=config.get('inference_device','auto'))
    section=config['collect' if collect else 'evaluate']
    epsilon=float(section.get('epsilon', .1)) if collect else 0.
    if not 0. <= epsilon <= 1.:raise ValueError('collect.epsilon must be between 0 and 1')
    requested=section.get('model_path','latest')
    path=latest_model(root/'models') if requested=='latest' else root/requested
    model,payload=load_agent(path,device)
    print(f'Agent: {path}\nSource: {path.with_suffix(".network.py")}\nInference: {device} / 50 Hz simulation',flush=True)
    if headless:
        import os
        os.environ.setdefault('SDL_VIDEODRIVER','dummy')
        os.environ.setdefault('SDL_AUDIODRIVER','dummy')
    else:
        prepare_gui()
    results=[]
    saved_episodes=completed_episode_count(root/section['data_dir']) if collect else 0
    from network import calculate_reward, collection_action
    with RLDrivingEnv(display=not headless,max_steps=max_steps,
                      reward_function=calculate_reward if collect else None) as env:
        raw=env.unwrapped
        raw.gui_language=config.get('language','kor')
        raw.gui_rl_collect=collect
        raw.gui_collect=collect
        raw.gui_model=path.name
        raw.gui_epsilon=epsilon
        raw.gui_reward_notes = {
            'kor': ('Reward Function: src/network.py · calculate_reward()', '보상 기준은 위 함수에서 설정합니다. 변경 후 데이터를 다시 수집하세요.'),
            'eng': ('Reward Function: src/network.py · calculate_reward()', 'Edit reward rules there; collect new data after changes.'),
            'chn': ('Reward Function: src/network.py · calculate_reward()', '在该函数中设置奖励规则；修改后请重新采集数据。'),
        }[raw.gui_language]
        for episode in range(1,(episodes or 1000000)+1):
            raw.gui_episode=saved_episodes+1 if collect else episode
            raw.gui_saved_episodes=saved_episodes
            raw.gui_action = None
            action_counts = ActionCounts()
            raw.gui_action_counts = action_counts.display
            obs,info=env.reset()
            raw.gui_status='recording' if collect else 'driving'
            exploration_seed=secrets.randbits(64) if collect else None
            exploration_rng=np.random.default_rng(exploration_seed) if collect else None
            writer=EpisodeWriter(root/section['data_dir'],dict(reward_source_sha256=hashlib.sha256(Path(__import__('network').__file__).read_bytes()).hexdigest(),agent=path.name,agent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),network_sha256=payload['network_sha256'],seed=TRACK_SEED,collection_policy='network.collection_action',epsilon=epsilon,exploration_seed=exploration_seed),async_write=True) if collect else None
            reason=None
            closing=False
            try:
                while reason is None:
                    if not headless:
                        import pygame
                        for event in pygame.event.get():
                            if (event.type==pygame.QUIT or (event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE) or
                                (collect and event.type==pygame.MOUSEBUTTONDOWN and event.button==1 and
                                 getattr(raw,'rl_exit_rect',pygame.Rect(0,0,0,0)).collidepoint(event.pos))):
                                reason='window_closed';closing=True
                                break
                            elif collect and ((event.type==pygame.KEYDOWN and event.key==pygame.K_SPACE) or
                                (event.type==pygame.MOUSEBUTTONDOWN and event.button==1 and getattr(raw,'rl_stop_rect',pygame.Rect(0,0,0,0)).collidepoint(event.pos))):
                                reason='manual_stop'
                            elif (not collect and event.type==pygame.KEYDOWN and
                                  event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and
                                  not getattr(event, 'repeat', False)):
                                reason='manual_restart'
                        if reason:break
                    decision = {} if collect and not headless else None
                    action=collection_action(model,obs,device,epsilon,exploration_rng) if collect else select_action(model,obs,device)
                    if not isinstance(action, (int, np.integer)) or not 0 <= action < 9:
                        raise ValueError('행동은 0~8 정수여야 합니다.')
                    raw.gui_action = decision
                    if decision is not None:
                        raw.gui_action_counts = action_counts.add(action)
                    if delay:time.sleep(delay)
                    next_obs,reward,done,truncated,info=env.step(action)
                    if writer:writer.append(obs,action,reward,next_obs,done,truncated)
                    obs=next_obs
                    if done or truncated:
                        reason=info.get('termination_reason') or 'time_limit'
                folder=writer.finish(reason,info) if writer else None
            finally:
                if writer:writer.close()
            if folder:saved_episodes+=1
            result=dict(info,reason=reason,agent=str(path),data=str(folder) if folder else None, runtime=dict(device=device.type,torch=torch.__version__,seed=TRACK_SEED))
            results.append(result)
            print(json.dumps(result,ensure_ascii=False),flush=True)
            if closing:break
            if not collect and not headless and reason != 'manual_restart':
                raw.render()
                # Keep final metrics visible until Enter or closing the window.
                waiting=True
                while waiting:
                    for event in pygame.event.get():
                        if event.type==pygame.QUIT or (event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE):return results
                        if (event.type==pygame.KEYDOWN and
                            event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and
                            not getattr(event, 'repeat', False)):
                            waiting=False
                    raw.render()
            if headless and episodes is None:break
    return results


def validate_policy(model, device, max_steps=5000):
    """배포 전 간단 주행 검사. 서버 공식 채점 기록을 만들지는 않는다."""
    import os
    os.environ.setdefault('SDL_VIDEODRIVER','dummy')
    os.environ.setdefault('SDL_AUDIODRIVER','dummy')
    inference_runtime(device=str(device))
    model.eval()
    with RLDrivingEnv(max_steps=max_steps) as env:
        observation,info=env.reset()
        for _ in range(max_steps):
            action=select_action(model,observation,device)
            observation,_,done,truncated,info=env.step(action)
            if done or truncated:break
    return info


def validation_rank(info):
    return (0,info['lap_time']) if info.get('lap_finished') else (1,-info['progress'])


# 고정 학습 알고리즘: Double DQN + 기존 보수적 손실 항.
class LearningAlgorithm:
    name = 'offline Double DQN + conservative penalty'

    def __init__(self, model, settings):
        import copy
        self.model = model
        self.target = copy.deepcopy(model).eval()
        self.settings = settings
        self.updates = 0

        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(settings['learning_rate']))

    def compute_loss(self, batch):
        # Double DQN 목표와 손실 계산. terminated는 이후 보상을 사용하지 않습니다.

        model = self.model
        values = model(batch['observations'].float())
        chosen = values.gather(1, batch['actions'].long()[:, None]).squeeze(1)
        with torch.no_grad():
            model.eval()
            try:

                next_observations = batch['next_observations'].float()
                next_actions = model(next_observations).argmax(1, keepdim=True)
                next_values = self.target(next_observations).gather(1, next_actions).squeeze(1)

            finally:
                model.train()
            gamma = float(self.settings.get('gamma', .99))
            expected = batch['rewards'] + gamma * (~batch['terminated'].bool()).float() * next_values

        td = torch.nn.functional.smooth_l1_loss(chosen, expected)
        conservative = (torch.logsumexp(values, dim=1) - chosen).mean()
        return td + float(self.settings.get('conservative_weight', .1)) * conservative

    def update(self, batch):
        self.model.train()
        loss = self.compute_loss(batch)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError('학습 loss는 유한한 scalar tensor여야 합니다.')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        self.optimizer.step()
        self.updates += 1
        self.update_target()
        return float(loss.detach().cpu())

    def update_target(self):

        if self.updates % 100 == 0:
            self.target.load_state_dict(self.model.state_dict())


def train(root, config, epochs=None):
    import math
    import inspect
    from network import ClassificationNetwork
    from replay import epoch_batches
    settings=config['train']
    # CUDA adaptive pooling backward has no deterministic implementation.
    # Prefer deterministic kernels, but allow unsupported training operations.
    # Evaluation below restores strict determinism via inference_runtime().
    device=inference_runtime(device=settings.get('device','auto'), warn_only=True)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is not available')
    if device.type=='mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is not available')
    epoch_count=epochs if epochs is not None else settings.get('epochs',5)
    batch_size=settings.get('batch_size',64)
    workers=settings.get('num_workers',2)
    if type(epoch_count) is not int or epoch_count < 1:
        raise ValueError('epochs must be a positive integer')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if type(workers) is not int or not 0 <= workers <= 16:
        raise ValueError('num_workers must be an integer from 0 to 16')
    source=root/'src/network.py'
    model=ClassificationNetwork(device=device)
    start=latest_model(root/'models')
    previous,payload=load_agent(start,device)
    same=model_definition(source.read_bytes())==model_definition(start.with_suffix('.network.py').read_bytes())
    if same:model.load_state_dict(previous.state_dict())
    initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()} if same else None
    print(f'Training: {device}; {"continue " + start.name if same else "new architecture, scratch"}',flush=True)
    del previous
    algorithm=LearningAlgorithm(model,settings)
    files=dataset_files(root/settings['data_dir'])
    losses=[]
    epoch_samples=[]
    updates=0
    model.train()
    print(f'Data: {len(files)} shards; {epoch_count} epochs; batch={batch_size}; readers={workers}',flush=True)
    for epoch in range(epoch_count):
        samples=0
        for arrays in epoch_batches(files, batch_size, workers, seed=epoch):
            batch = {}
            for key, value in arrays.items():
                tensor = torch.from_numpy(value)
                if device.type == 'cuda':
                    tensor = tensor.pin_memory()
                batch[key] = tensor.to(device, non_blocking=device.type == 'cuda')
            loss = float(algorithm.update(batch))
            if not math.isfinite(loss):raise ValueError('Non-finite training loss')
            updates+=1
            samples+=len(arrays['actions'])
            losses.append(loss)
            if updates==1 or updates%50==0:
                print(f'Epoch {epoch+1}/{epoch_count}, update {updates}, samples {samples}: loss={losses[-1]:.5f}',flush=True)
        if samples == 0:
            raise ValueError('학습 epoch에서 처리한 전이가 없습니다.')
        epoch_samples.append(samples)
        print(f'Epoch {epoch+1}/{epoch_count}: all {samples} transitions processed',flush=True)
    metadata=dict(algorithm=algorithm.name,updates=updates,
                  algorithm_sha256=hashlib.sha256(Path(inspect.getfile(LearningAlgorithm)).read_bytes()).hexdigest(),
                  epochs=epoch_count,epoch_samples=epoch_samples,batch_size=batch_size,
                  training_device=str(device),num_workers=workers,
                  initial_agent=str(start) if same else 'scratch',data_shards=len(files))
    candidate=save_agent(model,root/'models/candidates',source,metadata)
    evaluation_device=inference_runtime(device=config.get('inference_device','auto'))
    model.to(evaluation_device);model.device=evaluation_device
    print('Validating the trained candidate on the fixed track...',flush=True)
    candidate_info=validate_policy(model,evaluation_device)
    metadata['candidate_validation']=candidate_info
    metadata['candidate_file']=str(candidate.relative_to(root))
    metadata['validation_device']=str(evaluation_device)
    selected='trained_candidate'
    if same:
        trained_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(initial_state)
        print('Comparing with the previous agent...',flush=True)
        previous_info=validate_policy(model,evaluation_device)
        metadata['previous_validation']=previous_info
        if validation_rank(candidate_info)<=validation_rank(previous_info):
            model.load_state_dict(trained_state)
        else:
            selected='previous_agent'
            print('Candidate regressed. Keeping the previous agent; candidate remains in models/candidates/.',flush=True)
    metadata['selected']=selected
    path=save_agent(model,root/'models',source,metadata)
    path.with_suffix('.training.json').write_text(json.dumps(dict(losses=losses,selection=metadata,data_files=[str(p.relative_to(root)) for p in files]),indent=2))
    print(f'Saved: {path}\nSubmit with: {path.with_suffix(".network.py")}',flush=True)
    return path
