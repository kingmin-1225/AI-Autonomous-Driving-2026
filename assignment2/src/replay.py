"""프레임별 전이를 작은 파일로 저장한다. 마지막 전이는 종료 사유를 보존한다."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from agent_io import PROTOCOL


class EpisodeWriter:
    def __init__(self, directory, metadata, chunk_size=256, async_write=False):
        stamp = datetime.now()
        while True:
            self.folder = Path(directory) / stamp.strftime('%Y%m%d_%H%M%S_%f')
            try:
                self.folder.mkdir(parents=True)
                break
            except FileExistsError:
                stamp += timedelta(microseconds=1)
        self.metadata = dict(metadata, protocol=PROTOCOL, steps=0, reward=0., complete=False)
        self.chunk_size = chunk_size
        self.pending = []
        self.chunks = 0
        # One writer preserves chunk order; at most two queued chunks bound memory.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='episode-save') if async_write else None
        self.futures = deque()

    def append(self, observation, action, reward, next_observation, terminated, truncated):
        if len(self.pending) >= self.chunk_size:
            self.flush()
        self.pending.append((observation.copy(), int(action), float(reward), next_observation.copy(), bool(terminated), bool(truncated)))
        self.metadata['steps'] += 1
        self.metadata['reward'] += float(reward)

    def flush(self):
        if not self.pending:
            return
        if self.executor and len(self.futures) >= 2:
            self.futures.popleft().result()
        rows = self.pending
        path = self.folder / f'{self.chunks:05d}.npz'
        if self.executor:
            self.futures.append(self.executor.submit(self._write_chunk, path, rows))
        else:
            self._write_chunk(path, rows)
        self.chunks += 1
        self.pending = []

    @staticmethod
    def _write_chunk(path, pending):
        rows = list(zip(*pending))
        names = ('observations', 'actions', 'rewards', 'next_observations', 'terminated', 'truncated')
        types = (np.uint8, np.int64, np.float32, np.uint8, bool, bool)
        np.savez_compressed(path, **{n:np.asarray(v,dtype=t) for n,v,t in zip(names,rows,types)})

    def close(self):
        """Join pending saves and propagate failures before publishing the manifest."""
        if self.executor:
            self.executor.shutdown(wait=True)
        while self.futures:
            self.futures.popleft().result()

    def finish(self, reason, info):
        if not self.metadata['steps']:
            self.close()
            self.folder.rmdir()
            return None
        if reason in ('manual_stop','window_closed') and self.pending:
            row = list(self.pending[-1])
            # Manual stop is terminal, so Q targets do not bootstrap across reset.
            row[4] = True
            self.pending[-1] = tuple(row)
        try:
            self.flush()
        finally:
            self.close()
        self.metadata.update(complete=True, reason=reason, chunks=self.chunks, result=info)
        (self.folder/'episode.json').write_text(json.dumps(self.metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        return self.folder


def completed_episode_count(directory):
    """Count saved episodes once at startup, excluding interrupted/incompatible data."""
    count = 0
    for manifest in Path(directory).rglob('episode.json'):
        try:
            metadata = json.loads(manifest.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if isinstance(metadata, dict) and metadata.get('complete') and metadata.get('protocol') == PROTOCOL and metadata.get('steps', 0) > 0:
            count += 1
    return count


def dataset_files(directory):
    files=[]
    for manifest in sorted(Path(directory).rglob('episode.json')):
        metadata=json.loads(manifest.read_text(encoding='utf-8'))
        if metadata.get('complete') and metadata.get('protocol') == PROTOCOL:
            shards=sorted(manifest.parent.glob('*.npz'))
            if len(shards) != metadata.get('chunks',len(shards)):
                raise ValueError(f'에피소드 데이터 파일이 누락되었습니다: {manifest}')
            count=0
            for shard in shards:
                with np.load(shard,allow_pickle=False) as data:
                    count+=len(data['actions'])
            if count != metadata.get('steps'):
                raise ValueError(f'에피소드 전이 개수가 일치하지 않습니다: {manifest}')
            files.extend(shards)
    if not files:
        raise ValueError('완료된 RL 데이터가 없습니다. 먼저 collect.py를 실행하세요.')
    return files


KEYS = ('observations', 'actions', 'rewards', 'next_observations', 'terminated', 'truncated')


def read_shard(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in KEYS}
    n = len(arrays['actions'])
    if n == 0 or any(len(value) != n for value in arrays.values()):
        raise ValueError(f'Empty or inconsistent replay shard: {path}')
    for key in ('observations', 'next_observations'):
        if arrays[key].shape != (n, 96, 96, 3) or arrays[key].dtype != np.uint8:
            raise ValueError(f'Invalid observations: {path}')
    for key in KEYS[1:3] + KEYS[4:]:
        if arrays[key].shape != (n,):
            raise ValueError(f'Invalid transition shape: {path}')
    actions = arrays['actions']
    if not np.issubdtype(actions.dtype, np.integer) or np.any((actions < 0) | (actions > 8)):
        raise ValueError(f'Invalid actions: {path}')
    if not np.isfinite(arrays['rewards']).all():
        raise ValueError(f'Invalid rewards: {path}')
    return arrays


def loaded_shards(files, workers):
    if workers == 0:
        for path in files:
            yield read_shard(path)
        return
    # Threads avoid spawn/fork and GPU-context issues on Windows/macOS/Linux.
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='replay-read') as pool:
        paths = iter(files)
        pending = deque()
        for _ in range(workers):
            path = next(paths, None)
            if path is not None:
                pending.append(pool.submit(read_shard, path))
        while pending:
            arrays = pending.popleft().result()
            path = next(paths, None)
            if path is not None:
                pending.append(pool.submit(read_shard, path))
            yield arrays


def epoch_batches(files, batch_size, workers=2, seed=0):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if type(workers) is not int or not 0 <= workers <= 16:
        raise ValueError('num_workers must be an integer from 0 to 16')
    rng = np.random.default_rng(seed)
    order = [files[int(i)] for i in rng.permutation(len(files))]
    pieces = {key: [] for key in KEYS}
    count = 0
    for arrays in loaded_shards(order, workers):
        indices = rng.permutation(len(arrays['actions']))
        offset = 0
        while offset < len(indices):
            take = min(batch_size-count, len(indices)-offset)
            selected = indices[offset:offset+take]
            for key in KEYS:
                pieces[key].append(arrays[key][selected])
            count += take
            offset += take
            if count == batch_size:
                yield {key: np.concatenate(value) for key, value in pieces.items()}
                pieces = {key: [] for key in KEYS}
                count = 0
    if count:
        yield {key: np.concatenate(value) for key, value in pieces.items()}
