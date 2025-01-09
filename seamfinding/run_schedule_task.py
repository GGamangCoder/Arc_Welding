# 쓰레드 - 데이터 획득
# interval(60sec)마다 함수 실행

import threading
import time
import test_final

def schedule_task(interval, task):
    def wrapper():
        task()
        threading.Timer(interval, wrapper).start()

    wrapper()


schedule_task(60, test_final.execute)