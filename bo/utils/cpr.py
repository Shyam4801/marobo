import cProfile
import pstats
import io
import functools

def cprofile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        # Create a stream object to capture the profiling results
        stream = io.StringIO()
        ps = pstats.Stats(pr, stream=stream)
        ps.print_stats()
        
        # Print the profiling results
        print(f"Profiling results for {func.__name__}:")
        print(stream.getvalue())
        
        return result
    
    return wrapper