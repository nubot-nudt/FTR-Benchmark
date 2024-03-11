import sys, os

from loguru import logger

import traceback
def init_logger():

    stack_trace = traceback.extract_stack()
    bottom_frame = stack_trace[0]
    launched_file = bottom_frame.filename
    name = os.path.basename(launched_file)[:-3]

    log_file = f'logs/run_log_{name}.log'

    logger.add(log_file, rotation='10MB', retention='72h', encoding='utf-8')