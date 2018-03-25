import sys
import time

arr = ['/images/tick.png', '/images/nofeed.png', '/images/cross.png', '/images/error.png']

i = 0
while (True):
    time.sleep(3)
    print(arr[i])
    sys.stdout.flush()    
    i += 1