import os, shutil

def get_train_names():
    train_names = [i[:-5] for i in os.listdir('cfg/train')]
    return train_names

def get_task_names():
    task_names = [i[:-5] for i in os.listdir('cfg/task')]
    return task_names

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