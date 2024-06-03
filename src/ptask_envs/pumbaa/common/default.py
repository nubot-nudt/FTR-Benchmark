
robot_base_name = 'pumbaa_wheel'

robot_prim_path = f'pumbaa_wheel'

baselink_prim_path = f'{robot_prim_path}/chassis_link'

flipper_material_path = f'{robot_prim_path}/Looks/flipper_material'
wheel_material_path = f'{robot_prim_path}/Looks/wheel_material'

baselink_wheel_prim_path = f'{robot_prim_path}/wheel_list/wheel_*/[LR]*'
baselink_wheel_render_prim_path = f'{baselink_wheel_prim_path}/Render'
flipper_prim_path = f'{robot_prim_path}/flipper_list/[fr]*/[FR]*'
flipper_render_prim_path = f'{robot_prim_path}/flipper_list/*/*/FlipperRender'
flipper_fr_prim_path = f'{robot_prim_path}/flipper_list/front_right_wheel/FR*'
flipper_fl_prim_path = f'{robot_prim_path}/flipper_list/front_left_wheel/FL*'
flipper_rl_prim_path = f'{robot_prim_path}/flipper_list/rear_left_wheel/RL*'
flipper_rr_prim_path = f'{robot_prim_path}/flipper_list/rear_right_wheel/RR*'

L = 0.5

wheel_radius = 0.1
center_height = wheel_radius

flipper_joint_names = ['front_left_flipper_joint', 'front_right_flipper_joint', 'rear_left_flipper_joint',
                       'rear_right_flipper_joint']

flipper_joint_paths = [f'{baselink_prim_path}/{i}' for i in flipper_joint_names]

baselink_wheel_joint_names = [
    *[f'L{i + 1}RevoluteJoint' for i in range(8)],
    *[f'R{i + 1}RevoluteJoint' for i in range(8)],
]

flipper_wheel_joint_names = [
    *[f'LF{i + 1}RevoluteJoint' for i in range(5)],
    *[f'RL{i + 1}RevoluteJoint' for i in range(5)],
    *[f'LR{i + 1}RevoluteJoint' for i in range(5)],
    *[f'RR{i + 1}RevoluteJoint' for i in range(5)],
]

wheel_joint_names = [
    *baselink_wheel_joint_names,
    *flipper_wheel_joint_names,
]

robot_length, robot_width = 0.56, 0.48

flipper_joint_x, flipper_joint_y, flipper_length = 0.28, 0.26, 0.62-0.28
flipper_start_point = [
    (flipper_joint_x, flipper_joint_y),
    (flipper_joint_x, -flipper_joint_y),
    (-flipper_joint_x, flipper_joint_y),
    (-flipper_joint_x, -flipper_joint_y),
]
flipper_end_point = [
    ((flipper_joint_x + flipper_length), flipper_joint_y),
    ((flipper_joint_x + flipper_length), -flipper_joint_y),
    (-(flipper_joint_x + flipper_length), flipper_joint_y),
    (-(flipper_joint_x + flipper_length), -flipper_joint_y),
]

wheel_points = {
    'left_track': [(-0.28 + i * 0.08, 0.145) for i in range(8)],
    'right_track': [(-0.28 + i * 0.08, -0.145) for i in range(8)],
    'front_left_flipper': [(0.28 + i * 0.085, 0.26) for i in range(5)],
    'front_right_flipper': [(0.28 + i * 0.085, -0.26) for i in range(5)],
    'rear_left_flipper': [(-(0.28 + i * 0.085), 0.26) for i in range(5)],
    'rear_right_flipper': [(-(0.28 + i * 0.085), -0.26) for i in range(5)],
}
