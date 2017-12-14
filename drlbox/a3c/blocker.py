
import time
from datetime import timedelta
import signal

class Blocker:

    need_block = True

    def __init__(self, sleep_unit=1, report_interval=300):
        signal.signal(signal.SIGINT, self.handler)
        signal.signal(signal.SIGTERM, self.handler)
        self.sleep_unit = sleep_unit
        self.report_interval = report_interval
        self.start_time = time.time()

    def handler(self, signum, frame):
        self.need_block = False

    def block(self):
        wait_idx = 0
        while self.need_block:
            wait_idx += 1
            if wait_idx >= self.report_interval:
                wait_idx = 0
                elapsed = int(time.time() - self.start_time)
                time_str = str(timedelta(seconds=elapsed))
                print('Elapsed time:', time_str)
            time.sleep(self.sleep_unit)

