import time

def begin_from(count):
    while True:
        print('Countdown -', count)
        if count == 0:
            break
        count -= 1
        time.sleep(1)
