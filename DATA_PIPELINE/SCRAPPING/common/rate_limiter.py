import time
class RateLimiter:
    def __init__(self, delay=1.5):
        self.delay = delay
        self.last_time = 0
    def wait(self):
        elapsed = time.time() - self.last_time
        if elapsed < self.delay: time.sleep(self.delay - elapsed)
        self.last_time = time.time()
