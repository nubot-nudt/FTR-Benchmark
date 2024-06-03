import os
import shutil
import pathlib
import queue


def get_train_names():
    return _get_sub_names('cfg/train')


def get_task_names():
    return _get_sub_names('cfg/task')


def _get_sub_names(path):
    train_names = []
    q = queue.Queue()

    for i in os.listdir(path):
        q.put(i)

    while not q.empty():
        item: str = q.get()

        if item.endswith('.yaml'):
            train_names.append(item.removesuffix('.yaml'))
        elif os.path.isdir(f'{path}/{item}'):
            for i in os.listdir(f'{path}/{item}'):
                q.put(f'{item}/{i}')

    return train_names


def backup_task_nn(task):
    directory = f'runs/{task}'
    if not os.path.exists(directory):
        return

    root_dir = os.path.dirname(directory)
    name = os.path.basename(directory)

    for i in range(1, 100):
        new_dir = os.path.join(root_dir, f'{name}-{i}')
        if not os.path.exists(new_dir):
            shutil.move(directory, new_dir)
            return
    else:
        raise RuntimeError(f"could not backup {directory}")
