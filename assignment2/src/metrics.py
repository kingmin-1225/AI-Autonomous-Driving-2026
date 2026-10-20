"""한 바퀴의 진행 거리, 중앙 유지 점수와 시뮬레이션 시간을 독립 집계한다."""
import numpy as np


OFF_ROAD_LIMIT = 5.0  # 연속 이탈 허용 시간 (시뮬레이션 초)


class DrivingMetrics:
    def __init__(self, track, position, half_width):
        self.points = np.asarray(track, dtype=float)[:, 2:4]
        self.vectors = np.roll(self.points, -1, axis=0) - self.points
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.starts = np.r_[0., np.cumsum(self.lengths)[:-1]]
        self.length = float(self.lengths.sum())
        self.half_width = half_width
        self.previous, _ = self.project(position)
        self.distance = self.furthest = self.scored_distance = self.center_integral = 0.
        self.off_road_frames = 0
        self.off_road_seconds = 0.
        self.off_road = False
        self.frames = 0
        self.elapsed = 0.
        self.finished = self.aborted = False
        self.lap_time = None
        self.totals = np.zeros(3)
        self.display_totals = np.zeros(3)
        self.display_delta = np.zeros(3)
        self.last_totals = np.zeros(3)
        self.next_update = 1.

    def project(self, position):
        relative = np.asarray(position) - self.points
        fractions = np.clip(np.sum(relative * self.vectors, axis=1) / self.lengths**2, 0, 1)
        errors = np.linalg.norm(relative - fractions[:, None] * self.vectors, axis=1)
        index = int(np.argmin(errors))
        return float(self.starts[index] + fractions[index] * self.lengths[index]), float(errors[index])

    def update(self, position, on_road, dt):
        if self.finished or self.aborted:
            return
        self.frames += 1
        self.elapsed = self.frames * dt
        was_off_road = self.off_road
        self.off_road = not on_road
        self.off_road_frames = self.off_road_frames + 1 if self.off_road else 0
        self.off_road_seconds = self.off_road_frames * dt
        self.aborted = self.off_road_seconds >= OFF_ROAD_LIMIT - 1e-9

        # 이탈 중에도 위치/최대 진행 구간을 소비하여 복귀 시 소급 득점하지 않는다.
        along, error = self.project(position)
        delta = (along - self.previous + self.length / 2) % self.length - self.length / 2
        self.previous = along
        self.distance += delta
        reached = min(self.length, max(self.furthest, self.distance))
        gained = reached - self.furthest
        if on_road and not was_off_road:
            self.scored_distance += gained
            normalized_error = min(error / self.half_width, 1.)
            self.center_integral += gained * (.2 + .8 * (1 - normalized_error)**4)
        self.furthest = reached
        self.finished = on_road and self.distance >= self.length - 1e-7
        if self.finished:
            self.lap_time = self.elapsed
        self.totals = np.array([100 * self.scored_distance / self.length,
                                100 * self.center_integral / self.length, self.elapsed])
        if self.elapsed + 1e-9 >= self.next_update or self.finished or self.aborted:
            self.display_totals = self.totals.copy()
            self.display_delta = self.totals - self.last_totals
            self.last_totals = self.totals.copy()
            self.next_update = np.floor(self.elapsed + 1e-9) + 1

    def result(self):
        return dict(progress=float(self.totals[0]), center=float(self.totals[1]),
                    elapsed=self.elapsed, lap_time=self.lap_time,
                    lap_finished=self.finished, lane_departure=self.off_road,
                    off_road_seconds=self.off_road_seconds,
                    recovery_remaining=max(0., OFF_ROAD_LIMIT - self.off_road_seconds),
                    lane_departure_timeout=self.aborted)
