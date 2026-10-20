"""단조 시계로 프레임 간격을 유지하고 지연 후 몰아서 그리지 않는다."""
import time


class FramePacer:
    def __init__(self, clock=None, sleep=None):
        self.clock = clock or time.perf_counter
        self.sleep = sleep or time.sleep
        self.last = None
        self.deadline = None

    def tick(self, fps):
        now = self.clock()
        if self.last is not None and fps > 0:
            deadline = self.deadline if self.deadline is not None else self.last + 1.0 / fps
            remaining = deadline - now
            while remaining > 0:
                self.sleep(remaining)
                now = self.clock()
                remaining = deadline - now
        elapsed = 0.0 if self.last is None else now - self.last
        # 짧은 sleep 오차는 누적하지 않고, 한 주기 이상 지연되면 기준을 재설정한다.
        if fps > 0:
            period = 1.0 / fps
            self.deadline = (deadline + period if self.last is not None and now - deadline < period
                             else now + period)
        else:
            self.deadline = None
        self.last = now
        return elapsed * 1000
