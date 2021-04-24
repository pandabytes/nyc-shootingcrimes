import time

def elapsedTime(func):
    ''' Print the time that a function takes to complete '''
    def wrapper(*args, **kwargs):
        timeStart = time.time()
        result = func(*args, **kwargs)
        elapsedTime = time.time() - timeStart
        if not ("quiet" in kwargs and kwargs["quiet"]):
        	print("Function \"{0}\" took {1:.2f} seconds to complete".format(func.__name__, elapsedTime))
        return result
    return wrapper