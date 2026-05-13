import time
from functools import wraps
def retry_request(max_retries=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try: return func(*args, **kwargs)
                except: time.sleep(2**i)
            return None
        return wrapper
    return decorator
