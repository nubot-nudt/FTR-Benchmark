
robot_prim_path = f'pumbaa_wheel'

baselink_prim_path = f'{robot_prim_path}/chassis_link'

baselink_wheel_prim_path = f'{robot_prim_path}/wheel_list/wheel_*/[LR]*'
baselink_wheel_render_prim_path = f'{baselink_wheel_prim_path}/Render'
flipper_prim_path = f'{robot_prim_path}/flipper_list/[fr]*/[FR]*'
flipper_render_prim_path = f'{robot_prim_path}/flipper_list/*/*/FlipperRender'

L = 0.3

wheel_radius = 0.1
center_height = wheel_radius

flipper_joint_names = ['front_left_flipper_joint', 'front_right_flipper_joint', 'rear_left_flipper_joint',
                       'rear_right_flipper_joint']

wheel_joint_names = [
    *[f'L{i + 1}RevoluteJoint' for i in range(8)],
    *[f'R{i + 1}RevoluteJoint' for i in range(8)],
    *[f'LF{i + 1}RevoluteJoint' for i in range(5)],
    *[f'RL{i + 1}RevoluteJoint' for i in range(5)],
    *[f'LR{i + 1}RevoluteJoint' for i in range(5)],
    *[f'RR{i + 1}RevoluteJoint' for i in range(5)],
]
robot_length, robot_width = 0.56, 0.48
flipper_start_point = [(0.28, 0.26), (0.28, -0.26), (-0.28, 0.26), (-0.28, -0.26)]
flipper_end_point = [(0.62, 0.26), (0.62, -0.26), (-0.62, 0.26), (-0.62, -0.26)]
