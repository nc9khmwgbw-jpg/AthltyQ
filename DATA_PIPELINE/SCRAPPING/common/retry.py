import time
from functools import wraps
def retry_request(max_retries=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"⚠️ Tentative {i+1}/{max_retries} échouée : {str(e)}")
                    if i < max_retries - 1:
                        time.sleep(2 ** i)
            print(f"❌ Échec définitif après {max_retries} tentatives.")
            return None
        return wrapper
    return decorator
