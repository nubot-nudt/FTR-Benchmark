import os
import sys

import requests
from loguru import logger

def pushpluse_send_msg(title, content, token=None):

    if token is None or len(token) == 0:
        token = os.getenv("PUSHPLUS_TOKEN")

    if token is None or len(token) == 0:
        logger.error('could not find PUSHPLUS_TOKEN')
        return None

    data = {
        'token': token,
        'title': title,
        'content': content,
    }
    return requests.post('http://www.pushplus.plus/send/', json=data).text

if __name__ == '__main__':
    print(pushpluse_send_msg(sys.argv[1], sys.argv[2]))