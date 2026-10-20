"""Shared RL training/evaluation rules; imitation-learning rules remain unchanged."""
from collections import deque
import math
import numpy as np
from metrics import DrivingMetrics
from driving import DEFAULT_DRIVING

RL_PROTOCOL = 'driving-50hz-rl-v2'
RL_DRIVING = dict(DEFAULT_DRIVING, max_speed=50.0, throttle=0.5)
RL_ENGINE_POWER_MULTIPLIER = 1.0
RL_GRASS_PENALTY = 0.5  # per 50 Hz physics step
RL_GRASS_MAX_SPEED = 8.0
RL_TIME_PENALTY = 0.0
RL_FINISH_REWARD = 0.0


class RLDrivingMetrics(DrivingMetrics):
    """Count on-road coverage separately from completing a lap; grass remains penalized."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.termination_reason = None
        self.step_reward = self.episode_reward = 0.0
        self.display_episode_reward = self.display_reward_delta = 0.0
        self._recent_rewards = deque()
        self.needs_recovery = False
        self.grass_seconds = 0.0
        self.reward_boundaries = np.sort((self.starts - self.previous) % self.length)
        self.rewarded_tiles = 0
        self.display_rewarded_tiles = 0
        self._rewarded_segments = np.zeros(len(self.reward_boundaries), dtype=bool)

    def update(self, position, on_road, dt):
        from metrics import OFF_ROAD_LIMIT
        if self.finished or self.aborted:
            self.step_reward = 0.0
            return
        self.frames += 1
        self.elapsed = self.frames * dt
        along, error = self.project(position)
        delta = (along-self.previous+self.length/2) % self.length-self.length/2
        self.previous = along
        was_off_road = self.off_road
        previous_distance = self.distance
        self.distance += delta
        self.off_road = not on_road
        self.off_road_frames = self.off_road_frames+1 if self.off_road else 0
        self.off_road_seconds = self.off_road_frames*dt
        if self.off_road:
            self.grass_seconds += dt
        # Projection along a corner can move farther than the physical displacement.
        # Include road-width tolerance; large jumps still cannot create progress.
        invalid_jump = abs(delta) > RL_DRIVING['max_speed']*dt + 2*self.half_width
        self.aborted = invalid_jump or self.off_road_seconds >= OFF_ROAD_LIMIT-1e-9
        # Track the furthest route position even on grass, so skipped distance
        # never becomes delayed progress when the car returns to the road.
        reached = min(self.length, max(self.furthest, self.distance))
        gained = max(0., reached - self.furthest)
        self.furthest = reached
        if not self.aborted and on_road and not was_off_road:
            self.scored_distance += gained
            normalized_error = min(error/self.half_width,1.)
            self.center_integral += gained*(.2+.8*(1-normalized_error)**4)
        self.needs_recovery = False
        self.finished = not self.aborted and on_road and self.distance >= self.length-1e-7
        if self.finished:
            self.lap_time = self.elapsed
            self.termination_reason = 'finished'
        elif self.aborted:
            self.termination_reason = 'invalid_progress' if invalid_jump else 'grass_timeout'
        self.totals = np.array([100. if self.finished else 100*self.scored_distance/self.length,
                               100*self.center_integral/self.length,self.elapsed])
        # Reward actual on-road visits independently of the coverage score.
        # Never credit skipped grass segments or reward the same segment twice.
        new_tiles = 0
        if not self.aborted and on_road and delta > 0:
            offsets = (self.reward_boundaries - previous_distance % self.length) % self.length
            visited = offsets <= delta + 1e-9
            current_segment = (int(np.searchsorted(self.reward_boundaries, self.distance % self.length, side='right')) - 1) % len(visited)
            visited[current_segment] = True
            new_tiles = int(np.count_nonzero(visited & ~self._rewarded_segments))
            self._rewarded_segments |= visited
            self.rewarded_tiles += new_tiles
        center_bonus = .5 * max(0., 1. - error / self.half_width)
        self.step_reward = new_tiles * (1. + center_bonus) - RL_TIME_PENALTY
        if self.off_road:
            self.step_reward -= RL_GRASS_PENALTY
        if self.finished:
            self.step_reward += RL_FINISH_REWARD
        custom_reward = getattr(self, 'reward_function', None)
        if custom_reward is not None:
            context = dict(new_road_segments=new_tiles, center_alignment=2*center_bonus,
                           on_road=bool(on_road), finished=self.finished, aborted=self.aborted,
                           speed=getattr(self,'speed',0.), dt=dt, elapsed=self.elapsed)
            self.step_reward = float(custom_reward(context))
            if not math.isfinite(self.step_reward):
                raise ValueError('calculate_reward must return a finite number')
        self.episode_reward += self.step_reward
        # UI only: accumulate each step, but refresh the one-second change once per second.
        self._recent_rewards.append((self.elapsed, self.step_reward))
        while self._recent_rewards and self._recent_rewards[0][0] <= self.elapsed - 1. + 1e-9:
            self._recent_rewards.popleft()
        if self.elapsed+1e-9 >= self.next_update or self.finished or self.aborted:
            self.display_episode_reward = self.episode_reward
            self.display_rewarded_tiles = self.rewarded_tiles
            self.display_reward_delta = math.fsum(reward for _, reward in self._recent_rewards)
        if self.elapsed+1e-9 >= self.next_update or self.finished or self.aborted:
            self.display_totals = self.totals.copy()
            self.display_delta = self.totals-self.last_totals
            self.last_totals = self.totals.copy()
            self.next_update = np.floor(self.elapsed+1e-9)+1

    def result(self):
        return dict(super().result(), termination_reason=self.termination_reason,
                    reward=self.step_reward, episode_reward=self.episode_reward,
                    needs_recovery=self.needs_recovery, grass_seconds=self.grass_seconds,
                    recovery_progress=100*self.scored_distance/self.length)


def apply_grass_slowdown(car,dt):
    """Trusted physics penalty, independent of the submitted model's actions."""
    for body in [car.hull,*car.wheels]:
        velocity=np.asarray(body.linearVelocity,dtype=float)
        speed=float(np.linalg.norm(velocity))
        if speed > RL_GRASS_MAX_SPEED:
            velocity *= RL_GRASS_MAX_SPEED/speed
            body.linearVelocity=tuple(float(v) for v in velocity)


def footprint_positions(car):
    return [np.array(car.hull.position), *[np.array(w.position) for w in car.wheels]]


class RoadIndex:
    """Conservative spatial lookup for static road fixtures; TestPoint stays authoritative."""
    def __init__(self, road, cell_size=16.):
        import math
        self.cell_size = cell_size
        self.cells = {}
        # Box2D AABBs already enclose the shape. Padding also covers float conversion.
        for tile in road:
            for fixture in tile.fixtures:
                bounds = fixture.GetAABB(0)
                low = [math.floor((float(v)-.001)/cell_size) for v in bounds.lowerBound]
                high = [math.floor((float(v)+.001)/cell_size) for v in bounds.upperBound]
                for x in range(low[0], high[0]+1):
                    for y in range(low[1], high[1]+1):
                        self.cells.setdefault((x,y), []).append(fixture)

    def contains(self, point):
        import math
        point = tuple(float(x) for x in point)
        cell = tuple(math.floor(x/self.cell_size) for x in point)
        return any(fixture.TestPoint(point) for fixture in self.cells.get(cell, ()))


def road_contains(road, point):
    if isinstance(road, RoadIndex):
        return road.contains(point)
    point = tuple(float(x) for x in point)
    return any(fixture.TestPoint(point) for tile in road for fixture in tile.fixtures)


def swept_on_road(road, previous, current):
    # Check the center and all four wheel centers, including between physics steps.
    for start, end in zip(previous, current):
        count = max(1, int(np.ceil(np.linalg.norm(end-start) / .25)))
        for fraction in np.linspace(0., 1., count+1):
            if not road_contains(road, start+(end-start)*fraction):
                return False
    return True


def apply_rl_engine_power(car):
    """Scale engine input inside this RL car only; preserve gas ramp and braking."""
    if getattr(car, '_rl_power_installed', False):
        return
    original_step = car.step
    def powered_step(dt):
        gas = [wheel.gas for wheel in car.wheels]
        try:
            for wheel, value in zip(car.wheels, gas):
                wheel.gas = value * RL_ENGINE_POWER_MULTIPLIER
            original_step(dt)
        finally:
            for wheel, value in zip(car.wheels, gas):
                wheel.gas = value
    car.step = powered_step
    car._rl_power_installed = True
