from collections import deque
import numpy as np


class REPLAY_BUFFER:
  def __init__(self, max_size):
    self.buffer = deque(maxlen = max_size)

  def add_experience(self, experience):
    self.buffer.append(experience)

  def get_batch(self, batch_size):
    batch_idx = np.random.choice(np.arange(len(self.buffer)), size = batch_size, replace = False)
    return [self.buffer[i] for i in batch_idx]


