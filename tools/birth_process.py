import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()
# ---------------------------------------------------

import math
from pprint import pprint

import fire
from tui.select_info import select_asset_name
from utils.common import birth

asset = select_asset_name()

birth_file, birth_info = birth.load_birth_file_and_info(asset)

class BirthProcessor():

    def stdize(self):

        for item in birth_info:
            sp = item['start_point']
            tp = item['target_orient']

            sp[1] = tp[1] = (sp[1] + tp[1]) / 2

            item['start_point'] = sp
            item['target_orient'] = tp
            item['start_orient'] = (0, 0, 0 if sp[0] < tp[0] else math.pi)

        birth.save(birth_file, birth_info)
        print(f'{asset} stdize finish')
        pprint(birth_info)

    def reverse(self):
        info = []
        for item in birth_info:
            sp = item['start_point']
            tp = item['target_point']

            sp[1] = tp[1] = (sp[1] + tp[1]) / 2

            info.append({
                'start_point': sp,
                'start_orient': (0, 0, 0 if sp[0] < tp[0] else math.pi),
                'target_point': tp,
                'target_orient': item['target_orient']
            })

            info.append({
                'start_point': tp,
                'start_orient': (0, 0, 0 if tp[0] < sp[0] else math.pi),
                'target_point': sp,
                'target_orient': item['start_orient']
            })

        birth.save(birth_file, info)
        print(f'{asset} stdize finish')
        pprint(info)




if __name__ == '__main__':
    fire.Fire(BirthProcessor)