__all__ = ['singleton','cached']

def singleton(cls):
    instance_container = []
    def getinstance():
        if not len(instance_container):
            instance_container.append(cls())
        return instance_container[0]
    return getinstance

def cached(f):
    result = {}
    def wrapped_f(*x):
        if x not in result:
            result[x] = f(*x)
        return result[x]
    wrapped_f.__name__ = f.__name__
    return wrapped_f
