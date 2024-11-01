import threading
import time
import test_final

def schedule_task(interval, task):
    def wrapper():
        task()
        threading.Timer(interval, wrapper).start()

    wrapper()


schedule_task(60, test_final.execute)