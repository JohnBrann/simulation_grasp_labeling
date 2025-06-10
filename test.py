# from isaacsim import SimulationApp
# import os
# import numpy as np

# # Start Isaac Sim with GUI
# simulation_app = SimulationApp({"headless": False})

# from isaacsim.core.api import World
# from isaacsim.core.utils.stage import add_reference_to_stage
# from isaacsim.robot.manipulators.grippers import ParallelGripper
# from isaacsim.core.utils.types import ArticulationAction
# from pxr import UsdGeom, Gf, Sdf, UsdPhysics, UsdLux, PhysicsSchemaTools
# # from omni.physx.utils import PhysicsSchemaTools
# from isaacsim.core.prims import SingleArticulation

# # Create simulation world (1 unit = 1 meter)
# my_world = World(stage_units_in_meters=1.0)
# stage = my_world.stage

# # Reference the standalone Robotiq Hand-E USD
# assets_dir = "/home/csrobot/Isaac/assets/isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005/Assets/Isaac/4.5/Isaac/Robots/Robotiq/Hand-E"
# usd_path = os.path.join(assets_dir, "Robotiq_Hand_E_base.usd")
# prim_path = "/World/robotiq_gripper"
# add_reference_to_stage(
#     usd_path=usd_path,
#     prim_path=prim_path
# )

# # Lift the gripper for visibility
# prim = stage.GetPrimAtPath(prim_path)
# if prim and prim.IsValid():
#     xform = UsdGeom.Xformable(prim)
#     xform.ClearXformOpOrder()
#     xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.3))

# # Zero-gravity physics scene
# global_phys = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
# global_phys.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
# global_phys.CreateGravityMagnitudeAttr().Set(0.0)

# # Ground plane and lighting
# PhysicsSchemaTools.addGroundPlane(
#     stage,
#     "/World/groundPlane",
#     "Z",
#     15,
#     Gf.Vec3f(0.0),
#     Gf.Vec3f(0.7)
# )
# light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/directionalLight"))
# light.GetIntensityAttr().Set(500.0)
# light.GetColorAttr().Set(Gf.Vec3f(1.0))
# lxform = UsdGeom.Xformable(light.GetPrim())
# lxform.ClearXformOpOrder()
# lxform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 45.0, 0.0))

# # Reset and start simulation
# my_world.reset()
# # my_world.play()

# # Wrap the gripper USD prim as a SingleArticulation
# hand_art = SingleArticulation(
#     prim_path=prim_path,
#     name="robotiq_gripper_articulation"
# )
# hand_art.initialize()

# # Create ParallelGripper wrapper around the same prim
# gripper = ParallelGripper(
#     end_effector_prim_path=f"{prim_path}/base_link",
#     joint_prim_names=["Slider_1", "Slider_2"],
#     joint_opened_positions=np.array([0.0, 0.0]),
#     joint_closed_positions=np.array([0.02, -0.02]),
#     action_deltas=np.array([-0.02, 0.02])
# )
# # Initialize the gripper helper
# dof_names = hand_art.dof_names
# gripper.initialize(
#     articulation_apply_action_func=hand_art.apply_action,
#     get_joint_positions_func=hand_art.get_joint_positions,
#     set_joint_positions_func=hand_art.set_joint_positions,
#     dof_names=dof_names
# )

# # Precompute finger DOF indices and open pose
# dof_indices = np.array([hand_art.get_dof_index(n) for n in ["Slider_1", "Slider_2"]])
# open_positions = gripper.joint_opened_positions
# close_torque = np.array([0.5, 0.5])  # increase torque if needed

# # Control loop: 100 steps open, 100 steps close, repeat
# step = 0
# while simulation_app.is_running():
#     my_world.step(render=True)
#     # Read joint positions
#     pos = hand_art.get_joint_positions()
#     if step % 100 == 0:
#         print(f"Step {step}, Positions: {pos}")
#     phase = step % 200
#     if phase < 100:
#         # Position control: move to open pose via ArticulationAction
#         action = ArticulationAction(
#             joint_positions=open_positions,
#             joint_indices=dof_indices
#         )
#         hand_art.apply_action(action)
#     else:
#         # Torque control: directly apply efforts to joints
#         hand_art.set_joint_efforts(
#             close_torque,
#             joint_indices=dof_indices
#         )
#     step += 1

# # Clean up
# simulation_app.close()
# # simulation_app.close()



############################

from isaacsim import SimulationApp
import os
import numpy as np

# Start Isaac Sim with GUI
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.prims import RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdGeom, Gf, Sdf, UsdPhysics, UsdLux, PhysicsSchemaTools, PhysxSchema
from isaacsim.core.api.robots import Robot
import omni
from isaacsim.sensors.physics import ContactSensor
from isaacsim.sensors.physics import _sensor
# from omni.physx

# Create simulation world (1 unit = 1 meter)
my_world = World(stage_units_in_meters=1.0)
# stage = my_world.stage
stage = omni.usd.get_context().get_stage()

# Define physics scene and set gravity
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(0.0)

# Add ground plane
PhysicsSchemaTools.addGroundPlane(stage, "/World/groundPlane", "Z", 15,
                                Gf.Vec3f(0,0,0), Gf.Vec3f(0.7))

# Add a dome light because isaac starts fully dark for some reason?
dome_path = Sdf.Path("/World/DomeLight")
if not stage.GetPrimAtPath(dome_path):
    dome = UsdLux.DomeLight.Define(stage, dome_path)
    dome.CreateIntensityAttr(750.0)
    dome.CreateColorAttr((1.0, 1.0, 1.0))

create_prim(
    prim_path="/World/CrackerBox",
    prim_type="Xform",
    usd_path="/home/csrobot/Isaac/assets/ycb/056_tennis_ball/textured.usdc"
    # usd_path="/home/csrobot/Isaac/assets/ycb/003_cracker_box/textured.usd"
)

# Wrap it in RigidPrim so the item is affected by gravity
collision_object = RigidPrim(
    prim_path=f"/World/CrackerBox",
    name="cracker_collision",
    position=[0.0, 0.0, 1.0],
    orientation=[1.0, 0.0, 0.0, 0.0],
    # orientation=[1.0, 0.0, 0.0, 0.0],
    scale=[1, 1, 1],
)

# Enable collision on the box by using the collision api
UsdPhysics.CollisionAPI.Apply(collision_object.prim)

# Add box to the simulation scene
my_world.scene.add(collision_object)

# Reference the standalone Robotiq Hand-E USD
assets_dir = "/home/csrobot/Isaac/assets/isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005/Assets/Isaac/4.5/Isaac/Robots/Robotiq/Hand-E"
usd_path = os.path.join(assets_dir, "Robotiq_Hand_E_base.usd")
prim_path = "/robotiq_gripper"
add_reference_to_stage(
    usd_path=usd_path,
    prim_path=prim_path
)

# wrap the prim as a robot (articulation)
prim = Robot(
    prim_path=prim_path,
    name="Hand-E",
    position=[0.0, 1.0, 1.0],       # ← move the root 1 m up in world‐space
)

# add the Robot into the physics scene
my_world.scene.add(prim)
# reset the world (this calls initialize() + post_reset() for every prim in scene)
my_world.reset()

# now DOFs are available
print("DOF names:", prim.dof_names)

dof_names     = prim.dof_names
slider_idxs   = np.array([
    dof_names.index("Slider_1"),
    dof_names.index("Slider_2")
])
print("gripper joint indices:", slider_idxs)


# after you've referenced the USD and before any world.reset()
left_path  = Sdf.Path(f"{prim_path}/left_gripper")
right_path = Sdf.Path(f"{prim_path}/right_gripper")

left_prim  = stage.GetPrimAtPath(left_path)
right_prim = stage.GetPrimAtPath(right_path)

# on the left finger, filter out any contacts with the right finger
pairs_api = UsdPhysics.FilteredPairsAPI.Apply(left_prim)
rel       = pairs_api.CreateFilteredPairsRel()
rel.AddTarget(right_prim.GetPath())

# left finger
left_sensor = ContactSensor(
    prim_path=f"{prim_path}/left_gripper/LeftFingerContact",
    name="LeftFingerContact",
    frequency=60,              # up to 60 Hz sampling
    translation=np.array([0,0,0.1]),
    min_threshold=0.0,
    max_threshold=1e6,
    radius=-1,                 # use full link extent
)

# right finger
right_sensor = ContactSensor(
    prim_path=f"{prim_path}/right_gripper/RightFingerContact",
    name="RightFingerContact",
    frequency=60,
    translation=np.array([0,0,0]),
    min_threshold=0.0,
    max_threshold=1e6,
    radius=-1,
)

# Joint position controll
# for dof in ["Slider_1","Slider_2"]:
#     joint_prim = stage.GetPrimAtPath(f"{prim_path}/{dof}")
#     # attach the USD-side DriveAPI
#     drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
#     drive.CreateStiffnessAttr(1000.0)             # N/m
#     drive.CreateDampingAttr(   50.0)              # N-s/m
#     drive.CreateMaxForceAttr(500.0)  
#     drive.CreateTargetPositionAttr(0.0)
#     # drive.CreateEnablePositionDriveAttr(True)
#     # ensure PhysX sees it
#     PhysxSchema.PhysxJointAPI.Apply(joint_prim)


# Set threshold for gripper positions to allow for full closure
for dof in ["Slider_1","Slider_2"]:
    pj = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(Sdf.Path(f"{prim_path}/{dof}")))
    pj.GetLowerLimitAttr().Set(-0.025)  # double the stroke


for dof in ["Slider_1","Slider_2"]:
    jp    = stage.GetPrimAtPath(Sdf.Path(f"{prim_path}/{dof}"))
    drive = UsdPhysics.DriveAPI.Apply(jp, "linear")
    # switch to velocity drive
    # drive.CreateTypeAttr().Set(UsdPhysics.Tokens.velocity)
    # velocity drives only need damping (to avoid runaway)
    drive.CreateDampingAttr(50.0)           
    drive.CreateMaxForceAttr(200.0)         # max N·s/m

    # ensure PhysX joint schema is present
    PhysxSchema.PhysxJointAPI.Apply(jp)


# make two‐element arrays

# Kp = np.ones_like(slider_idxs, dtype=float) * 1000.0   # high stiffness
# Kd = np.ones_like(slider_idxs, dtype=float) *   10.0   # modest damping
close_positions = np.zeros(len(slider_idxs))           # [0.0, 0.0]
open_positions  = np.full(len(slider_idxs), 0.025)     # [0.025, 0.025]

# lower = np.array([-0.025, -0.025])
# upper = np.array([ 0.025,  0.025])

# close_action = ArticulationAction(
#     joint_positions=lower,
#     joint_efforts=  None,
#     joint_velocities=None,
#     joint_indices=slider_idxs
# )
# open_action  = ArticulationAction(
#     joint_positions=upper,
#     joint_efforts=  None,
#     joint_velocities=None,
#     joint_indices=slider_idxs
# )

close_vel = np.full(2, -0.03)
open_vel  = np.full(2,  0.03)

close_action = ArticulationAction(
    joint_velocities = close_vel,
    joint_indices    = slider_idxs
)
open_action = ArticulationAction(
    joint_velocities = open_vel,
    joint_indices    = slider_idxs
)
contact_iface = _sensor.acquire_contact_sensor_interface()
count = 0
while simulation_app.is_running():
    my_world.step(render=True)

    left_reading  = contact_iface.get_sensor_reading(left_sensor.prim_path,  use_latest_data=True)
    right_reading = contact_iface.get_sensor_reading(right_sensor.prim_path, use_latest_data=True)

    if left_reading.in_contact:
        print("Left finger touching something, force =", left_reading.value)
    if right_reading.in_contact:
        print("Right finger touching something, force =", right_reading.value)

    if count <= 100:
        prim.apply_action(open_action)
    elif count > 100 and count <= 200:
        prim.apply_action(close_action)
    elif count > 200:
        count = 0

    # if count == 100:              # after 100 frames, close
    #     for _ in range(60):
    #         prim.apply_action(open_action)
    #         my_world.step(render=True)
    #     # prim.apply_action(close_action)
    # if count == 200:              # after 200 frames, open
    #     for _ in range(60):
    #         prim.apply_action(close_action)
    #         my_world.step(render=True)

    count += 1


# close the robot fingers: panda_finger_joint1 (7) and panda_finger_joint2 (8) to 0.0

    # action = ArticulationAction(joint_positions=np.array([0.0, 0.0]), joint_indices=np.array([7, 8]))
    # prim.apply_action(action)
#     # Read joint positions
#     pos = hand_art.get_joint_positions()
#     if step % 1000 == 0:
#         print(f"Step {step}, Positions: {pos}")
#     phase = step % 2000
#     if phase < 1000:
#         # Position control: move to open pose via ArticulationAction
#         action = ArticulationAction(
#             joint_positions=open_positions,
#             joint_indices=dof_indices
#         )
#         hand_art.apply_action(action)
#     else:
#         # Torque control: directly apply efforts to joints
#         hand_art.set_joint_efforts(
#             close_torque,
#             joint_indices=dof_indices
#         )
#     step += 1

# # Clean up
# simulation_app.close()
# # simulation_app.close()
