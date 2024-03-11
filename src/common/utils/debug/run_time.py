import time

from loguru import logger

class log_mean_run_time:
    
    _record = dict()


    def __init__(self, mean_num=1):
        self.mean_num = mean_num

    def __call__(self, func):
        f_name = func.__name__
        
        if f_name not in self._record:
            self._record[f_name] = list()

        def inline_func(*argv, **kargv):
            start_time = time.time()
            
            ret = func(*argv, **kargv)
            
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            
            self._record[f_name].append(elapsed_time)
            if len(self._record[f_name]) >= self.mean_num:
                logger.debug(f'{f_name} running average time is {int(elapsed_time)}ms')
                self._record[f_name].clear()

            return ret
        
        return inline_func