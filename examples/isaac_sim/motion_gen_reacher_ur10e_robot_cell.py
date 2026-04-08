import torch
import argparse
import numpy as np
from isaacsim.simulation_app import SimulationApp

# 1. SETUP SIMULATION APP
parser = argparse.ArgumentParser()
parser.add_argument("--headless_mode", type=str, default=None)
parser.add_argument("--robot", type=str, default="ur10e.yml")
args = parser.parse_args()

simulation_app = SimulationApp({
    "headless": args.headless_mode is not None,
    "width": 1920,
    "height": 1080,
})

# 2. CORE IMPORTS (Must be after SimulationApp)
import carb
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.objects.cylinder import VisualCylinder
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage

# CuRobo Imports
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Cuboid as CuCuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

def create_safe_joint_state(pos, vel, names, tensor_args):
    """Factory to ensure acceleration/jerk are NEVER None to avoid 'NoneType' crash."""
    n = len(pos)
    return JointState(
        position=tensor_args.to_device(pos),
        velocity=tensor_args.to_device(vel),
        acceleration=torch.zeros(n, device=tensor_args.device, dtype=torch.float32),
        jerk=torch.zeros(n, device=tensor_args.device, dtype=torch.float32),
        joint_names=names
    )

def main():
    tensor_args = TensorDeviceType()
    my_world = World(stage_units_in_meters=1.0)
    usd_help = UsdHelper()
    
    # 3. SCENE SETUP
    my_world.stage.DefinePrim("/World", "Xform")
    
    # Load Visual Table (No Collision) - UPDATE THIS PATH
    USD_TABLE_PATH = "/home/shan/assets/my_table.usd" 
    table_prim_path = "/World/RobotCell/VisualTable"
    try:
        add_reference_to_stage(usd_path=USD_TABLE_PATH, prim_path=table_prim_path)
    except Exception as e:
        print(f"Warning: Could not load USD table: {e}")

    # 4. ROBOT & TARGET
    robot_cfg_path = get_robot_configs_path()
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)

    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.4, 0.3, 0.6]),
        size=0.05,
        color=np.array([1, 0, 0])
    )

    # Add visual obstacle (red cylinder) that robot must avoid
    obstacle = VisualCylinder(
        prim_path="/World/obstacle_cylinder",
        position=np.array([0.3, 0.0, 0.4]),  # Position between robot and target
        radius=0.08,  # 8cm radius
        height=0.8,   # 80cm height
        color=np.array([1.0, 0.2, 0.2])  # Red color
    )
    print("Added red cylinder obstacle for robot to avoid")

    # 5. CUROBO CONFIG (Logical Collision)
    cu_table = CuCuboid(
        name="logical_table",
        pose=[0, 0, 0.38, 1, 0, 0, 0], # Z lower than visual to avoid base collision
        dims=[1.2, 1.2, 0.76],
    )
    
    # Add logical collision obstacle (cylinder) for motion planning
    from curobo.geom.types import Cylinder as CuCylinder
    cu_obstacle = CuCylinder(
        name="obstacle_cylinder",
        pose=[0.3, 0.0, 0.4, 1, 0, 0, 0],  # Same position as visual obstacle
        radius=0.08,  # Same as visual
        height=0.8,   # Same as visual
    )
    
    world_cfg = WorldConfig(cuboid=[cu_table], cylinder=[cu_obstacle])

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
    )
    motion_gen = MotionGen(motion_gen_config)
    
    print("Warming up cuRobo... please wait.")
    motion_gen.warmup(enable_graph=True)

    add_extensions(simulation_app, args.headless_mode)
    plan_config = MotionGenPlanConfig(max_attempts=1, enable_graph=True)

    # 6. SIMULATION LOOP
    cmd_plan = None
    cmd_idx = 0
    articulation_controller = None

    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            continue

        step_index = my_world.current_time_step_index
        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()

        # Articulation Startup
        if step_index < 30:
            if step_index == 15:
                robot._articulation_view.initialize()
                # Check for name mismatch
                print(f"[DEBUG] Robot Joint Names in Isaac: {robot.dof_names}")
                print(f"[DEBUG] Robot Joint Names in cuRobo: {motion_gen.kinematics.joint_names}")
                
                default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
                idx_list = [robot.get_dof_index(x) for x in robot_cfg["kinematics"]["cspace"]["joint_names"]]
                robot.set_joint_positions(default_config, idx_list)
            continue

        # World Sync (Ignore Visual Assets)
        if step_index % 500 == 0:
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                reference_prim_path=robot_prim_path,
                ignore_substring=[robot_prim_path, table_prim_path, "/World/target", "/World/obstacle_cylinder", "/curobo"],
            ).get_collision_check_world()
            motion_gen.update_world(obstacles)

        # 7. PLANNING WITH DEFENSIVE TENSOR HANDLING
        if cmd_plan is None and step_index % 120 == 0:
            sim_js = robot.get_joints_state()
            
            # Create current state with explicit ZEROS for acceleration/jerk
            current_state = create_safe_joint_state(
                sim_js.positions, 
                sim_js.velocities, 
                robot.dof_names, 
                tensor_args
            )
            
            # Map names to solver order
            cu_js = current_state.get_ordered_joint_state(motion_gen.kinematics.joint_names)

            # Debugging check before solver call
            if cu_js.acceleration is None:
                print("[ERROR] Acceleration buffer lost during reordering!")
                continue

            cube_pos, cube_ori = target.get_world_pose()
            ik_goal = Pose(
                position=tensor_args.to_device(cube_pos),
                quaternion=tensor_args.to_device(cube_ori)
            )
            
            try:
                # Add batch dimension and plan
                result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
                
                if result.success.item():
                    print(f"Plan Success at step {step_index}")
                    cmd_plan = result.get_interpolated_plan()
                    cmd_plan = motion_gen.get_full_js(cmd_plan)
                    
                    # Indices for applying the action
                    idx_list = [robot.get_dof_index(x) for x in robot.dof_names if x in cmd_plan.joint_names]
                    cmd_plan = cmd_plan.get_ordered_joint_state([robot.dof_names[i] for i in idx_list])
                    cmd_idx = 0
                else:
                    print(f"Plan failed: {result.status}")
            except Exception as e:
                print(f"Solver Exception: {e}")

        # 8. EXECUTION
        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]
            art_action = ArticulationAction(
                joint_positions=cmd_state.position.cpu().numpy(),
                joint_velocities=cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            articulation_controller.apply_action(art_action)
            cmd_idx += 1
            if cmd_idx >= len(cmd_plan.position):
                cmd_plan = None
                print("Execution finished.")

    simulation_app.close()

if __name__ == "__main__":
    main()