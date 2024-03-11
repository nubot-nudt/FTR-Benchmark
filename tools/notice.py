
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()

from utils.notice import pushpluse_send_msg

if __name__ == '__main__':
    print(pushpluse_send_msg(sys.argv[1], sys.argv[2]), '02d07b1b9a20494c8be4da9652859156')