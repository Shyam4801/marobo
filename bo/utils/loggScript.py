import logging
import functools

from typing import Union

log = logging.getLogger()


class MyLogger:
    def __init__(self, filename):
        logging.basicConfig(level=logging.DEBUG)
        # logging.setLevel(logging.INFO) 
        self.fh = logging.FileHandler(filename=filename)
        formatter = logging.Formatter(
                        fmt = '%(asctime)s :: %(message)s', datefmt = '%a, %d %b %Y %H:%M:%S'
                        )

        self.fh.setFormatter(formatter)
        

    def get_logger(self, name=None):
        # logging.addHandler(self.fh)
        return logging.getLogger(name).addHandler(self.fh)
   
def get_default_logger():
    return MyLogger("shydef.log").get_logger()


def log(_func=None, *, my_logger: Union[MyLogger, logging.Logger] = None):
    def decorator_log(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_default_logger()
            try:
                if my_logger is None:
                    first_args = next(iter(args), None)  # capture first arg to check for `self`
                    logger_params = [  # does kwargs have any logger
                        x
                        for x in kwargs.values()
                        if isinstance(x, logging.Logger) or isinstance(x, MyLogger)
                    ] + [  # # does args have any logger
                        x
                        for x in args
                        if isinstance(x, logging.Logger) or isinstance(x, MyLogger)
                    ]
                    if hasattr(first_args, "__dict__"):  # is first argument `self`
                        logger_params = logger_params + [
                            x
                            for x in first_args.__dict__.values()  # does class (dict) members have any logger
                            if isinstance(x, logging.Logger)
                            or isinstance(x, MyLogger)
                        ]
                    h_logger = next(iter(logger_params), MyLogger())  # get the next/first/default logger
                else:
                    h_logger = my_logger  # logger is passed explicitly to the decorator

                if isinstance(h_logger, MyLogger):
                    logger = h_logger.get_logger(func.__name__)
                else:
                    logger = h_logger

                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                logger.debug(f"function {func.__name__} called with args {signature}")
            except Exception:
                pass

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.exception(f"Exception raised in {func.__name__}. exception: {str(e)}")
                raise e
        return wrapper

    if _func is None:
        return decorator_log
    else:
        return decorator_log(_func)
    

 
# @log
# def foo(a, b, logger):
#     pass

# @log
# def bar(a, b=10, logger=None): # Named parameter
#     pass


# foo(10, 20, logger)  # OR foo(10, 20, MyLogger().get_logger())
# bar(10, b=20, logger = logger)
# logger = MyLogger("shy.log").get_logger()

# class Foo:
#     def __init__(self):
#         # self.lg = logger
#         pass

#     # @log
#     def sum(self, a, b=10):
#         return a + b
    
#     @log(my_logger=logger)
#     def perform(self,c, car):
#         logging.info('inside perform')
#         c = car.sample(2,3)
#         return c
    
# class car:
#     def __init__(self) -> None:
#         pass
#     # @log
#     def sample(self,a,b):
#         logging.info('inside sample ')


# car1 = car()

# Foo().perform(10, car1) 
# car1.sample(4,5)