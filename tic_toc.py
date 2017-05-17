from time import time
_tstart_stack = []

def tic():
    _tstart_stack.append(time())

def toc():
    return (time() - _tstart_stack.pop())
