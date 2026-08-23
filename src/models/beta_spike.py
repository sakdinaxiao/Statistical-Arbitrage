import heapq
from collections import defaultdict, deque
import numpy as np


class _Median_engine:
    def __init__(self,window):
        self.window = window
        self.max = []
        self.min = []
        # use deque instead of list
        self.timeline = deque()
        self.graveyard = defaultdict(int)
        self.balance = 0

    def ghosthunt(self,heap,is_max=False):
        while heap:
            top_val = -heap[0] if is_max else heap[0]
            
            # kick ghost out
            if self.graveyard[top_val] > 0 :
                self.graveyard[top_val] -= 1
                heapq.heappop(heap)
            else:
                break

    def add(self,num):
        if len(self.timeline) == self.window:
            old_num = self.timeline.popleft()
            self.graveyard[old_num] += 1
            # calculate balance
            # self.max[0] is the NEGATED stored value; actual max-heap top is -self.max[0]
            if self.max and old_num <= -self.max[0]:
                self.balance -= 1
            else:
                self.balance += 1
        
        # it is not full now
        self.timeline.append(num)
        if not self.max or num <= -self.max[0]:
            heapq.heappush(self.max,-num)
            self.balance += 1
        else:
            heapq.heappush(self.min,num)
            self.balance -= 1

        # invariant: len(max) >= len(min), so balance in {0, 1}
        if self.balance > 1:
            item = heapq.heappop(self.max)
            heapq.heappush(self.min,-item)
            self.balance -= 2
        elif self.balance < 0:
            item = heapq.heappop(self.min)
            heapq.heappush(self.max,-item)
            self.balance += 2

    def median(self):
        self.ghosthunt(self.max,is_max=True)
        self.ghosthunt(self.min,is_max=False)

        if self.balance == 1:
            return -self.max[0]
        else:
            return (-self.max[0] + self.min[0])/2.0
        

class BetaChecker:
    def __init__(self, beta_spike_window, z_threshold=3.5):
        self.beta_spike_window = beta_spike_window
        self.z_threshold = z_threshold
        self.prev_beta = None
        self.med_engine = _Median_engine(beta_spike_window)   # rolling median of Δβ
        self.mad_engine = _Median_engine(beta_spike_window)   # rolling median of |Δβ − median|
        self.warmup_count = 0

    def beta_spike_check(self, beta):
        # Warmup / NaN guard
        if np.isnan(beta):
            return True
        if self.prev_beta is None:
            self.prev_beta = beta
            return True

        p = beta - self.prev_beta
        self.prev_beta = beta

        self.med_engine.add(p)
        current_median = self.med_engine.median()

        self.mad_engine.add(abs(p - current_median))
        mad = self.mad_engine.median()

        # Need a full beta_spike_window of Δβ before the MAD estimate is trustworthy
        self.warmup_count += 1
        if self.warmup_count < self.beta_spike_window:
            return False

        if mad == 0.0:
            return False   # no variance yet — can't compute, treat as healthy

        z = 0.6745 * (p - current_median) / mad
        return abs(z) >= self.z_threshold



