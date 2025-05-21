from omni.isaac.kit import SimulationApp

# Launch Isaac Sim with GUI
simulation_app = SimulationApp({"headless": False})

# Core Isaac Sim & USD imports
import omni.kit.commands
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.prims import GeometryPrim, RigidPrim
from isaacsim.asset.importer.urdf import _urdf
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import UsdLux, Sdf, UsdPhysics, Gf
from pxr import UsdPhysics, PhysxSchema, Gf, PhysicsSchemaTools, UsdGeom
from pxr import UsdGeom, Gf
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleArticulation
import numpy as np
import random

# semi-Random initial pose and orientation for the gripper at the start of a trial
def sample_random_pose_near(target_position: Gf.Vec3d, distance=0.5, yaw_jitter_deg=10.0):
    # Pick a random point on the circle
    angle = random.uniform(0, 2 * np.pi)
    dx = distance * np.cos(angle)
    dy = distance * np.sin(angle)
    position = Gf.Vec3d(
        target_position[0] + dx,
        target_position[1] + dy,
        target_position[2]
    )

    # Compute ideal yaw so gripper +X faces the box
    vec_to_box = Gf.Vec3d(
        target_position[0] - position[0],
        target_position[1] - position[1],
        0.0
    )
    ideal_yaw_rad = np.arctan2(vec_to_box[1], vec_to_box[0])
    ideal_yaw_deg = np.degrees(ideal_yaw_rad)

    # Add some jitter around that ideal yaw
    yaw_deg = ideal_yaw_deg + random.uniform(-yaw_jitter_deg, yaw_jitter_deg)

    # Build quaternion for that Z‑axis rotation
    q = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_deg).GetQuaternion()
    orientation = [
        q.GetImaginary()[0],
        q.GetImaginary()[1],
        q.GetImaginary()[2],
        q.GetReal()
    ]
    return position, orientation

def move_gripper_toward(gripper: SingleArticulation, target: Gf.Vec3d, step_size=0.002):
    current_pos, _ = gripper.get_world_pose()
    direction = np.array([target[0], target[1], target[2]]) - np.array(current_pos)
    distance = np.linalg.norm(direction)
    if distance < step_size:
        return False  # Reached
    direction = direction / distance  # Normalize
    new_pos = np.array(current_pos) + direction * step_size
    pose_mat = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*new_pos))
    gripper.set_world_pose(position=new_pos)
    return True


def reset_scene(gripper, cracker_box, target_position):
    # Reset the box 
    cracker_box.set_world_pose(
        position=[0.0, 0.0, 1.0],
        orientation=[0.0, 0.0, 0.0, 1.0]
    )

    # Sample a brand‑new pose
    new_pos, new_ori = sample_random_pose_near(target_position)

    # Restart the sim
    world.reset()
    gripper.initialize()

    # Teleport the gripper 
    gripper.set_world_pose(
        position=new_pos,
        orientation=new_ori
    )
    gripper.apply_action(open_action)

# Initialize simulation world
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

# Define physics scene and set gravity
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(0.0)

# Add ground plane
PhysicsSchemaTools.addGroundPlane(
    stage,
    "/World/groundPlane",
    "Z",
    15,
    Gf.Vec3f(0, 0, 0),
    Gf.Vec3f(0.7)
)

# Add a dome light because isaac starts fully dark for some reason?
dome_path = Sdf.Path("/World/DomeLight")
if not stage.GetPrimAtPath(dome_path):
    dome = UsdLux.DomeLight.Define(stage, dome_path)
    dome.CreateIntensityAttr(750.0)
    dome.CreateColorAttr((1.0, 1.0, 1.0))

# Function to import a URDF robot
def import_urdf_model(urdf_path: str, position=Gf.Vec3d(0.0, 0.0, 0.0), rotation_deg=90):
    import_config = _urdf.ImportConfig()
    import_config.convex_decomp = False # Disable convex decomposition for simplicity
    import_config.fix_base = False # Fix the base of the robot to the ground
    import_config.make_default_prim = False # Make the robot the default prim in the scene
    import_config.self_collision = False # Disable self-collision for performance
    import_config.distance_scale = 1.0 # Set distance scale for the robot
    import_config.density = 0.0 # Set density to 0 (use default values)
    import_config.merge_fixed_joints = True

    result, robot_model = omni.kit.commands.execute(
        "URDFParseFile",
        urdf_path=urdf_path,
        import_config=import_config,
    )
    # Update the joint drive parameters for better stiffness and damping (from the sample code)
    for joint in robot_model.joints:
        robot_model.joints[joint].drive.strength = 1047.19751  # High stiffness value
        robot_model.joints[joint].drive.damping = 52.35988    # Moderate damping value
    
    result, prim_path = omni.kit.commands.execute(
        "URDFImportRobot",
        urdf_robot=robot_model,
        import_config=import_config,
    )

    # Create and apply full transform (rotation + translation)
    rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation_deg)  # Rotate around Z axis
    mat = Gf.Matrix4d().SetRotate(rot)
    mat.SetTranslateOnly(position)

    xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(mat)
    # straight gangsta

    robot = SingleArticulation(prim_path=prim_path, name="panda")
    world.scene.add(robot)

    return robot

# Create CrackerBox object
create_prim(
    prim_path="/World/CrackerBox",
    prim_type="Xform",
    usd_path="/home/csrobot/Isaac/assets/ycb/003_cracker_box/textured.usd"
)

# Wrap it in RigidPrim so the item is affected by gravity
collision_box = RigidPrim(
    prim_path=f"/World/CrackerBox",
    name="cracker_collision",
    position=[0.0, 0.0, 1.0],
    scale=[1, 1, 1],
)

# Enable collision on the box by using the collision api
UsdPhysics.CollisionAPI.Apply(collision_box.prim)

# Add box to the simulation scene
world.scene.add(collision_box)

target_pos = Gf.Vec3d(0.0, 0.0, 1.0) 

start_pos, start_ori = sample_random_pose_near(target_pos)
gripper = import_urdf_model(
    "/home/csrobot/Isaac/assets/grippers/franka_panda/franka_panda.urdf",
    position=start_pos,
    rotation_deg=0 
)
gripper.set_world_pose(position=start_pos, orientation=start_ori)

# Start simulation loop
world.reset()
gripper.initialize()

close_action = ArticulationAction(joint_positions=np.array([0.0, 0.0])) # joint_indices=np.array()) 
open_action = ArticulationAction(joint_positions=np.array([0.04, 0.04])) # joint_indices=np.array())

grasped = False
trial = 0
max_trials = 100
count = 0

while trial < max_trials:
    world.step(render=True)

    if not grasped:
        #TODO: fix when the gripper grasps (based in distance from center of gripper to object contact point)
        if not move_gripper_toward(gripper, target_pos):
            gripper.apply_action(close_action)
            grasped = True
    else:
        gripper.apply_action(close_action)
        if count > 100:
            reset_scene(gripper, collision_box, target_pos)
            grasped = False
            count = 0
            trial += 1
            print(f"Trial {trial} complete. Resetting scene.")
    count += 1

# Close the simulator
simulation_app.close()
