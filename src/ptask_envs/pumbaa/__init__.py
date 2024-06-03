from loguru import logger


def initialize_task(config, env, init_sim=True, wrap=None):
    from ptask_envs.omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    sim_config = SimConfig(config)

    from ptask_envs import tasks

    task_class_list = [i for i in dir(tasks) if i.endswith('Task')]

    # 自动导入
    task_map = {i[:-4]: getattr(tasks, i) for i in task_class_list}

    cfg = sim_config.config

    if 'task' in cfg and 'task' in cfg['task'] and 'name' in cfg['task']['task']:
        task_name = cfg['task']['task']['name']
    elif 'name' in cfg['task']:
        task_name = cfg['task']['name']
    else:
        task_map = cfg['task_name']

    # logger.debug(f'task list: {task_class_list}')
    logger.info(f'loading task: {task_name} -> {tasks.__class__} ')

    task = task_map[task_name](
        name=task_name, sim_config=sim_config, env=env
    )

    if wrap is not None:
        task = wrap(task)

    env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=init_sim)

    return task
