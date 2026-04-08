# Initialize Isaac Sim app FIRST to avoid torch/usd runtime conflicts
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Standard Library
import time
import os
import sys

# Add the 'isaac_sim' folder to path so we can import built-in curobo helpers
sys.path.append(os.path.join(os.path.dirname(__file__), "isaac_sim"))

# Third Party
import numpy as np
import torch
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.utils.types import ArticulationAction
from isaac_sim.helper import add_robot_to_scene

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

def plot_traj(trajectory, dof):
    # Third Party
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(3, 1)
    q = trajectory[:, :dof]
    qd = trajectory[:, dof : dof * 2]
    qdd = trajectory[:, dof * 2 : dof * 3]

    for i in range(q.shape[-1]):
        axs[0].plot(q[:, i], label=str(i))
        axs[1].plot(qd[:, i], label=str(i))
        axs[2].plot(qdd[:, i], label=str(i))
    plt.legend()
    plt.savefig("test.png")
    plt.show()


def draw_points(rollouts: torch.Tensor):
    if rollouts is None:
        return
    
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw
        
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()
    cpu_rollouts = rollouts.cpu().numpy()
    b, h, _ = cpu_rollouts.shape
    point_list = []
    colors = []
    for i in range(b):
        point_list += [
            (cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]) for j in range(h)
        ]
        colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
    sizes = [10.0 for _ in range(b * h)]
    draw.draw_points(point_list, colors, sizes)


def demo_full_config_mpc():
    PLOT = True # Enabled plotting to show popup at end of run
    # Basic tensor device configuration indicating which device to use (e.g. CPU or GPU)
    tensor_args = TensorDeviceType()
    
    # =========================================================================
    # CONFIGURABLE PARAMETERS
    # =========================================================================
    # 1. world_file: Path or filename for the collision environment (YAML format).
    world_file = "collision_test.yml"
    
    # 2. robot_file: The robot configuration file.
    robot_file = "ur10e.yml"
    
    # 3. step_dt: The timestep (in seconds) used internally by the MPC solver.
    step_dt = 0.03
    
    # 4. goal_offset: How much to perturb the joint positions to generate a random 
    #    reachable goal pose using forward kinematics limit checks.
    goal_offset = 0.5
    
    # 5. pose_error_tolerance: Stopping criteria specifying the maximum allowed 
    #    error to consider reaching the goal successfully.
    pose_error_tolerance = 0.01
    
    # 6. max_steps: Hard limit to prevent the solver loop from running infinitely.
    max_steps = 1000

    # 7. robot_origin: Specify the robot's base position in world coordinates [x, y, z].
    robot_origin = np.array([0.0, 0.0, 0.9])
    # =========================================================================

    # -------------------------------------------------------------
    # SETUP ISAAC SIM ENVIRONMENT
    # -------------------------------------------------------------
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()

    # Load robot configuration details from specified YAML config
    robot_cfg_dict = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    
    # Spawn the Robot visually inside the Isaac Environment at the specified origin
    robot, robot_prim_path = add_robot_to_scene(robot_cfg_dict, my_world, position=robot_origin)
    articulation_controller = robot.get_articulation_controller()
    
    robot_cfg = RobotConfig.from_dict(robot_cfg_dict, tensor_args)

    # Initialize the MPC solver with settings like dt and rollouts to store
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_file,
        store_rollouts=True,
        step_dt=step_dt,
    )
    mpc = MpcSolver(mpc_config)

    # Retrieves the robot's default retract/home pose in joint space [1, num_dofs]
    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.unsqueeze(0)
    joint_names = mpc.joint_names

    # Compute Forward Kinematics to calculate the target Goal Pose in Cartesian space.
    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg + goal_offset, joint_names=joint_names)
    )
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    
    # Create the interactive target cube initialized at the computed target position
    target = cuboid.VisualCuboid(
        "/World/target",
        position=retract_pose.position.view(-1).cpu().numpy(),
        orientation=retract_pose.quaternion.view(-1).cpu().numpy(),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )
    
    # Sets the robot start position initially at the retract pose
    start_state = JointState.from_position(retract_cfg, joint_names=joint_names)

    # Package the start constraint, secondary goal constraint, and the main end-effector goal pose
    goal = Goal(
        current_state=start_state,
        goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        goal_pose=retract_pose,
    )
    
    # Formulate a set of goal buffers optimized for parallel MPC evaluation 
    goal_buffer = mpc.setup_solve_single(goal, 1)

    tstep = 0
    traj_list = []
    mpc_time = []
    
    # Assign the current goal buffers to the solver instance
    mpc.update_goal(goal_buffer)
    current_state = start_state
    
    # -------------------------------------------------------------
    # WARM UP THE SIMULATION & CACHE ACCESORS
    # -------------------------------------------------------------
    my_world.reset()
    for _ in range(10):
        my_world.step(render=True)
        
    idx_list = [robot.get_dof_index(x) for x in joint_names]
    
    past_pose = None
    
    # Control loop running MPC sequentially and Rendering 
    while simulation_app.is_running():
        draw_points(mpc.get_visual_rollouts())
        
        # Advance the world render loop one frame
        my_world.step(render=True)
        if not my_world.is_playing():
            continue
            
        # Dynamically read the cube's position from the Isaac Sim viewer
        cube_position, cube_orientation = target.get_world_pose()

        if past_pose is None:
            past_pose = cube_position + 1.0

        # If the user drags the target cube, update the goal immediately
        if np.linalg.norm(cube_position - past_pose) > 1e-3:
            # We subtract the robot origin because the MPC solver operates in the robot's local frame
            local_position = cube_position - robot_origin
            
            ik_goal = Pose(
                position=tensor_args.to_device(local_position),
                quaternion=tensor_args.to_device(cube_orientation),
            )
            goal_buffer.goal_pose.copy_(ik_goal)
            mpc.update_goal(goal_buffer)
            past_pose = cube_position
            
        st_time = time.time()
        
        # Advance the MPC step by solving for an optimal action sequence 
        result = mpc.step(current_state, 1)

        # Ensure GPU synchronization before calculating performance times
        torch.cuda.synchronize()
        if tstep > 5:
            mpc_time.append(time.time() - st_time)
            
        # Update our logical state 
        current_state.copy_(result.action)
        
        # Collect optimized states history
        traj_list.append(result.action.get_state_tensor())
        tstep += 1
            
        # -------------------------------------------------------------
        # APPLY ACTION ONTO THE VISUAL ROBOT
        # -------------------------------------------------------------
        cmd_state = result.action
        art_action = ArticulationAction(
            cmd_state.position.view(-1).cpu().numpy(),
            joint_indices=idx_list,
        )
        articulation_controller.apply_action(art_action)
            
    # Store graphical trajectory of executed multi-DOF joints using Matplotlib
    if PLOT:
        plot_traj(torch.cat(traj_list, dim=0).cpu().numpy(), dof=retract_cfg.shape[-1])
        
    simulation_app.close()

if __name__ == "__main__":
    demo_full_config_mpc()
