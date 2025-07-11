# Omniverse Kit & Isaac Sim core
# Considerations for using a simulated LiDar
from omni.isaac.kit import SimulationApp
# Launch Isaac Sim with GUI
simulation_app = SimulationApp({"headless": False})
# Core Isaac Sim & USD imports
import omni.kit.commands
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.prims import RigidPrim
from isaacsim.asset.importer.urdf import _urdf
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import UsdLux, Sdf, UsdPhysics, Gf, PhysicsSchemaTools, UsdGeom, Usd, UsdShade, PhysxSchema, Tf
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleArticulation
import omni.replicator.core as rep
import omni
import asyncio
from isaacsim.sensors.physx import _range_sensor
from omni.kit.viewport.utility import get_active_viewport
from isaacsim.core.api.physics_context import PhysicsContext
from isaacsim.sensors.physics import ContactSensor
import numpy as np
import random
import math
import carb
from isaacsim.core.api.objects import DynamicCuboid, GroundPlane
from isaacsim.core.api.world import World
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.sensors.physx import ProximitySensor, register_sensor, clear_sensors
from isaacsim.core.api.robots import Robot
from omni.physx.scripts.utils import setCollider
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
# from isaacsim.core.api.action import ArticulationAction
# from omni.isaac.core.articulations import ArticulationView, ArticulationViewCfg, ControlMode
import os


from util_functions import visualize_point_sample, calculate_centroid, position_lidar, calculate_approach_pose, look_at_point, move_gripper_to_lidar, move_gripper_toward, check_reached_target, move_gripper_upward, reset_scene, get_mesh_points, sample_point_and_normal, import_urdf_model 

# ─────────────────── SETUP WORLD & SCENE ─────────────────────────────────────
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

# Physics scene & zero gravity
scene = UsdPhysics.Scene.Get(stage, Sdf.Path("/physicsScene"))
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)

# Add ground plane
PhysicsSchemaTools.addGroundPlane(stage, "/World/groundPlane", "Z", 15,
                                Gf.Vec3f(0,0,0), Gf.Vec3f(0.7))
# Dome light
dome_path = Sdf.Path("/World/DomeLight")
if not stage.GetPrimAtPath(dome_path):
    dome = UsdLux.DomeLight.Define(stage, dome_path)
    dome.CreateIntensityAttr(750.0)
    dome.CreateColorAttr((1.0, 1.0, 1.0))

# ─────────────────── OBJECT ─────────────────────────────────────────
# create_prim(
#     prim_path="/World/object",
#     prim_type="Xform",
#     # usd_path="/home/csrobot/Isaac/assets/ycb/056_tennis_ball/textured.usdc"
#     usd_path="/home/csrobot/Isaac/assets/ycb/009_gelatin_box.usd"
#     # usd_path="/home/csrobot/Isaac/assets/ycb/engine/engine.usd"
#     # usd_path="/home/csrobot/Isaac/assets/ycb/cube.usd"
# )
# object = RigidPrim(
#     prim_path="/World/object",
#     name="object_collision",
#     position=[0.0, 0.0, 1.0],
#     orientation=[1.0, 0.0, 0.0, 0.0],
#     scale=[1.0, 1.0, 1.0],
#     # scale=[0.025, 0.025, 0.025],
# )

material = PhysicsMaterial(
            prim_path="/World/objectMaterial",
            static_friction=1.0,
            dynamic_friction=0.8,
        )

object = DynamicCuboid(
                prim_path="/World/object",
                name="object",
                size=1.0,
                color=np.array([0.5, 0.5, 0]),
                position=[0.0, 0.0, 0.0], 
                scale= Gf.Vec3f(0.05, 0.05, 0.05),
                mass=0.1,
            )
object.apply_physics_material(material)


# ─────────────────── GRIPPER ────────────────────────────────────────
# gripper = import_urdf_model(
#     "/home/csrobot/Isaac/assets/grippers/franka_panda/franka_panda.urdf",
#     position=Gf.Vec3d(0, 0, 0.5), rotation_deg=0)
# Reference the standalone Robotiq Hand-E USD

assets_dir = "/home/csrobot/Isaac/assets/isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005/Assets/Isaac/4.5/Isaac/Robots/Robotiq/Hand-E"
usd_path = os.path.join(assets_dir, "Robotiq_Hand_E_convexDecomp_flattened.usd")
gripper_prim_path = "/robotiq_gripper"
add_reference_to_stage(
    usd_path=usd_path,
    prim_path=gripper_prim_path
)
# wrap the prim as a robot (articulation)
gripper = Robot(
    prim_path=gripper_prim_path,
    name="Hand-E",
    position=[0.0, 1.0, 1.0],
)

# applies zero gravity to the gripper
links = ["base_link", "left_gripper", "right_gripper"]
for link in links:
    prim_path = f"{gripper_prim_path}/{link}"
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Link prim not found: {prim_path!r}")
    physxAPI = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physxAPI.CreateDisableGravityAttr(True)
    # Give the gripper a ton of mass so it does not rotate when trying to pick something
    if link == "base_link":
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.GetMassAttr().Set(100000.0)
    if link == "left_gripper" or "right_gripper":
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.GetMassAttr().Set(10.0)

# add the Robot into the physics scene
world.scene.add(gripper)

# Create Gripper Pads and attach to gripper

rubber_material = PhysicsMaterial(
            prim_path="/World/rubberMaterial",
            static_friction=1.0,
            dynamic_friction=0.8,
        )

left_pad_path  = "/robotiq_gripper/left_gripper/left_gripper_pad"
right_pad_path = "/robotiq_gripper/right_gripper/right_gripper_pad"

DynamicCuboid(
                prim_path=left_pad_path,
                name=f"left_gripper_pad",
                size=1.0,
                color=np.array([0.5, 0, 0]),
                translation= Gf.Vec3d(0.0, 0.05, -0.025),
                scale= Gf.Vec3f(0.02, 0.02, 0.002),
                mass=1.0,
            ).apply_physics_material(rubber_material)
DynamicCuboid(
                prim_path=right_pad_path,
                name=f"right_gripper_pad",
                size=1.0,
                color=np.array([0.5, 0, 0]),
                translation= Gf.Vec3d(0.0, 0.05, 0.025),
                scale= Gf.Vec3f(0.02, 0.02, 0.002),
                mass=1.0,
            ).apply_physics_material(rubber_material)

# left_gripper_pad = UsdGeom.Cube.Define(stage, "/robotiq_gripper/left_gripper/left_gripper_pad")
# xformLeftApi = UsdGeom.XformCommonAPI(left_gripper_pad)
# xformLeftApi.SetScale    (Gf.Vec3f(0.01, 0.01, 0.0007))
# xformLeftApi.SetTranslate(Gf.Vec3d(0.0, 0.05, -0.025))

# right_gripper_pad = UsdGeom.Cube.Define(stage, "/robotiq_gripper/right_gripper/right_gripper_pad")
# xformRightApi = UsdGeom.XformCommonAPI(right_gripper_pad)
# xformRightApi.SetScale    (Gf.Vec3f(0.01, 0.01, 0.0007))
# xformRightApi.SetTranslate(Gf.Vec3d(0.0, 0.05, 0.025))

# # Define rubber material
# left_pad_path  = "/robotiq_gripper/left_gripper/left_gripper_pad"
# right_pad_path = "/robotiq_gripper/right_gripper/right_gripper_pad"
# # left_pad_path  = "/robotiq_gripper/left_gripper"
# # right_pad_path = "/robotiq_gripper/right_gripper"

# mat_path   = Sdf.Path("/World/Materials/RubberMat")
# rubber_mat = UsdShade.Material.Define(stage, mat_path)

# # 2) Author the core friction + restitution (USD‐physics API)
# mat_api = UsdPhysics.MaterialAPI.Apply(rubber_mat.GetPrim())
# mat_api.CreateStaticFrictionAttr().Set(1.0)   # static friction coefficient :contentReference[oaicite:0]{index=0}
# mat_api.CreateDynamicFrictionAttr().Set(0.9)  # dynamic friction coefficient :contentReference[oaicite:1]{index=1}
# mat_api.CreateRestitutionAttr().Set(0.1)      # restitution (bounciness) :contentReference[oaicite:2]{index=2}

# # 3) (Optional) Tweak combine modes / compliant‐contact in PhysX
# px_api = PhysxSchema.PhysxMaterialAPI.Apply(rubber_mat.GetPrim())
# px_api.CreateFrictionCombineModeAttr().Set("multiply")
# px_api.CreateRestitutionCombineModeAttr().Set("average")


# for pad_path in (left_pad_path, right_pad_path):
#     pad_prim    = stage.GetPrimAtPath(Sdf.Path(pad_path))
#     bind_api    = UsdShade.MaterialBindingAPI.Apply(pad_prim)
#     bind_api.Bind(
#         rubber_mat                       
#     )

enable_extension("isaacsim.sensors.physx")
world.reset()
world.step(render=False)

# ─────────────────── LIDAR SENSOR ───────────────────────────────────────────
cone = UsdGeom.Cone.Define(stage, "/World/lidar_cone")
cone.GetHeightAttr().Set(0.1)
cone.GetRadiusAttr().Set(0.02)
# Rotate cone to start
xf_lidar_cone = UsdGeom.XformCommonAPI(cone.GetPrim())
xf_lidar_cone.SetRotate(Gf.Vec3f(-90.0, 0.0, 0.0)) 

# Add Lidar
timeline = omni.timeline.get_timeline_interface()   
lidarInterface = _range_sensor.acquire_lidar_sensor_interface() 

omni.kit.commands.execute('AddPhysicsSceneCommand',stage = stage, path='/physicsScene')
result, lidar = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path="/Lidar",
            parent="/World/lidar_cone",
            min_range=0.2,
            max_range=1.0,
            draw_points=True,
            draw_lines=True,
            horizontal_fov=3.0,
            vertical_fov=3.0,
            horizontal_resolution=1,
            vertical_resolution=1,
            rotation_rate=0.0,
            high_lod=True,
            yaw_offset=0.0,
            enable_semantics=False
        )

# Have lidar come out the cones point
lidarPath = "/World/lidar_cone/Lidar"
lidar_prim = stage.GetPrimAtPath(lidarPath)
xf_lidar = UsdGeom.Xformable(lidar_prim)

# Translate Lidar up the cones height
cone_height = cone.GetHeightAttr().Get()
xf_lidar.ClearXformOpOrder()
mat = Gf.Matrix4d()
mat.SetTranslateOnly(Gf.Vec3d(0.0, 0.0, cone_height))
# Rotate lidar
rot = Gf.Rotation(Gf.Vec3d(0.0, -1.0, 0.0), 90.0)
mat.SetRotate(rot)
xf_lidar.AddTransformOp().Set(mat)

# Calculate centroid of object????????????????????
# centroid = calculate_centroid("/World/object")
centroid = np.array([0, 0, 1], dtype=float)
print(f"Centroid: {centroid}")
visualize_point_sample(centroid, stage, sphere_radius=0.005, color=(0,255,0))

# Make it so makrer does not collide with anything
marker_path = "/World/marker_sphere"
marker_prim = stage.GetPrimAtPath(marker_path)
if not marker_prim or not marker_prim.IsValid():
    raise RuntimeError(f"Could not find prim at {marker_path!r}")

# Apply a PhysX collider so ProximitySensor can “see” it.
UsdPhysics.CollisionAPI.Apply(marker_prim)       
PhysxSchema.PhysxCollisionAPI.Apply(marker_prim) 

xform = UsdGeom.Xformable(marker_prim)
xform.AddScaleOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

# Filter out any real contacts between this marker and your robot/object.
filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(marker_prim)
# filtered_pairs.CreateFilteredPairsRel().AddTarget("/panda")
filtered_pairs.CreateFilteredPairsRel().AddTarget("/robotiq_gripper")
filtered_pairs.CreateFilteredPairsRel().AddTarget("/World/object")

enable_extension("isaacsim.sensors.physx")
simulation_app.update()
simulation_app.update()

grasptarget_path = "/robotiq_gripper/base_link"
# grasptarget_path = "/panda/panda_hand"
grasptarget_prim = stage.GetPrimAtPath(grasptarget_path)
if not grasptarget_prim or not grasptarget_prim.IsValid():
    raise RuntimeError(f"No valid prim at {grasptarget_path}")

# apply collision API's to gripper
UsdPhysics.CollisionAPI.Apply(grasptarget_prim)
PhysxSchema.PhysxCollisionAPI.Apply(grasptarget_prim)

# Create prim offset for Proximity Sensor
offset_path = grasptarget_path + "/sensor_offset"
if stage.GetPrimAtPath(offset_path):
    stage.RemovePrim(offset_path)
offset_xform = UsdGeom.Xform.Define(stage, offset_path)
offset_prim = stage.GetPrimAtPath(offset_path)#

# ROBOTIQ HAND-E GRIPPER
offset_api   = UsdGeom.XformCommonAPI(offset_xform)
offset_api.SetTranslate(Gf.Vec3d(0.0, 0.01, 0.0)) #0.11 is TCP
offset_api.SetScale(Gf.Vec3f(1.0, 1.0, 1.0))

# FRANKA PANDA GRIPPER
# offset_api.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.066)) #0.11 is TCP
# offset_api.SetScale(Gf.Vec3f(1.0, 1.0, 1.0))

UsdPhysics.CollisionAPI.Apply(offset_prim)
PhysxSchema.PhysxCollisionAPI.Apply(offset_prim)

# Create Proximity Sensor
clear_sensors()
s = ProximitySensor(offset_prim)
register_sensor(s)

proximity_distance = None

def print_proximity_data(_):
    global proximity_distance
    data = s.get_data()
    if "/World/marker_sphere" in data:
        dist     = data["/World/marker_sphere"]["distance"]
        proximity_distance = dist
        duration = data["/World/marker_sphere"]["duration"]
        # print(f"Proximity to marker_sphere: distance={dist:.3f}, duration={duration:.3f}")
        # print(f"Proximity to marker_sphere: distance={dist:.3f}")
        return proximity_distance

world.add_physics_callback("print_sensor", print_proximity_data)
# simulation_app.update()
# simulation_app.update()

# ─────────────────── CONTACT SENSOR ──────────────────────────────────
left_pad_path  = "/robotiq_gripper/left_gripper/left_gripper_pad"
right_pad_path = "/robotiq_gripper/right_gripper/right_gripper_pad"

# left_pad_prim = stage.GetPrimAtPath(left_pad_path)
# contact_report = PhysxSchema.PhysxContactReportAPI.Apply(left_pad_prim)
# contact_report.CreateThresholdAttr(0.0)

# right_pad_prim = stage.GetPrimAtPath(right_pad_path)
# contact_report = PhysxSchema.PhysxContactReportAPI.Apply(right_pad_prim)
# contact_report.CreateThresholdAttr(0.0)


# 4) Create & configure the ContactSensor wrappers
# left_sensor = ContactSensor(
#     prim_path=f"{left_pad_path}/Contact_Sensor",
#     name="LeftPadSensor",
#     frequency=120,                   # sample at 120 Hz (every physics step)
#     translation=np.array([0.0, 0.0, 0.0]),
#     min_threshold=0.0,               # report all contacts ≥ 0 N·s
#     max_threshold=1e7,               # clamp impulses up to 10 000 000 N·s
#     radius=-1                        # use the pad’s full collision bounds
# )
# right_sensor = ContactSensor(
#     prim_path=f"{right_pad_path}/Contact_Sensor",
#     name="RightPadSensor",
#     frequency=120,
#     translation=np.array([0.0, 0.0, 0.0]),
#     min_threshold=0.0,
#     max_threshold=1e7,
#     radius=-1
# )

# simulation_app.update()
# simulation_app.update()

# ─────────────────── GRIPPER JOINTS & DRIVES ────────────────────────────────────────
dof_names     = gripper.dof_names
print(f'dof_names {dof_names}')
slider_idxs   = np.array([
    dof_names.index("Slider_1"),
    dof_names.index("Slider_2")
])
print("gripper joint indices:", slider_idxs)

# after you've referenced the USD and before any world.reset()
left_path  = Sdf.Path(f"{gripper_prim_path}/left_gripper")
right_path = Sdf.Path(f"{gripper_prim_path}/right_gripper")
left_prim  = stage.GetPrimAtPath(left_path)
right_prim = stage.GetPrimAtPath(right_path)

# on the left finger, filter out any contacts with the right finger
pairs_api = UsdPhysics.FilteredPairsAPI.Apply(left_prim)
rel       = pairs_api.CreateFilteredPairsRel()
rel.AddTarget(right_prim.GetPath())

# Set threshold for gripper positions to allow for full closure
for dof in ["Slider_1","Slider_2"]:
    pj = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(Sdf.Path(f"{gripper_prim_path}/{dof}")))
    pj.GetLowerLimitAttr().Set(-0.025)  # double the stroke

enable_extension("isaacsim.sensors.physx")
simulation_app.update()
simulation_app.update()

# Position Control
# for dof in ["Slider_1","Slider_2"]:
#     jp    = stage.GetPrimAtPath(Sdf.Path(f"{gripper_prim_path}/{dof}"))
#     drive = UsdPhysics.DriveAPI.Apply(jp, "linear")
#     drive.CreateStiffnessAttr(400.0)             # N/m
#     drive.CreateDampingAttr(200.0)
#     drive.CreateMaxForceAttr(5000.0)         # max N·s/m
#     drive.CreateTargetPositionAttr().Set(0.0)
#     PhysxSchema.PhysxJointAPI.Apply(jp)

# # Gripper Actions
# close_action = ArticulationAction(
#     joint_positions = np.array([-0.02, -0.02]),
#     joint_indices   = slider_idxs,
#     # joint_efforts = np.full(len(slider_idxs), -0.05, dtype=np.float32),
# )
# open_action = ArticulationAction(
#     joint_positions = np.array([ 0.024,  0.024]),
#     joint_indices   = slider_idxs,
#     # joint_efforts = np.full(len(slider_idxs), +0.05, dtype=np.float32),
# )

# Torque Control
for dof in ["Slider_1","Slider_2"]:
    jp    = stage.GetPrimAtPath(Sdf.Path(f"{gripper_prim_path}/{dof}"))
    drive = UsdPhysics.DriveAPI.Apply(jp, "linear")
    drive.CreateStiffnessAttr(0.0)             # N/m
    drive.CreateDampingAttr(0.0)
    drive.CreateMaxForceAttr(100.0)         # max N·s/m
    drive.CreateTargetPositionAttr().Set(0.0)
    PhysxSchema.PhysxJointAPI.Apply(jp)

# Gripper Actions
close_action = ArticulationAction(
    # joint_positions = np.array([-0.02, -0.02]),
    joint_indices   = slider_idxs,
    joint_efforts = np.full(len(slider_idxs), -1.55, dtype=np.float32),
)
open_action = ArticulationAction(
    # joint_positions = np.array([ 0.024,  0.024]),
    joint_indices   = slider_idxs,
    joint_efforts = np.full(len(slider_idxs), +0.5, dtype=np.float32),
)
simulation_app.update()
simulation_app.update()



# for prim in stage.Traverse():
#     if "PhysxCollisionAPI" in prim.GetAppliedSchemas():
#         UsdShade.MaterialBindingAPI.Apply(prim).Bind(rubber_mat)
#         print(f"🔗 bound rubber_mat to collision prim: {prim.GetPath()}")


# ─────────────────── INITIALIZE SIMULATION ──────────────────────────────────
trial = 0
max_trials = 100
count = 0

target_point = reset_scene(world, stage, gripper, object, centroid, open_action)

while trial < max_trials:
    world.step(render=True)
    if count == 0:
        print(f'\nTrial {trial}:')
    # Check if gripper reached target point
    reached_target = check_reached_target(proximity_distance, reach_threshold=0.015)


    if reached_target:
        print('Gripper Reached Target')
        world.step(render=True)
        print('Closing Gripper')


        # Check contact of finger pads with object
        # left_contact = left_sensor.get_current_frame()
        # right_contact = right_sensor.get_current_frame()
        # print(f'left: {left_contact}')
        # if left_contact['number_of_contacts'] == 1 and right_contact['number_of_contacts'] == 1:
        #     print('Both finger pads in contact with object')
            # Get current finger pose, maintain it, then close a little bit more
            # gripper_joint_pos = gripper.get_joint_positions()

        gripper.apply_action(close_action)

        for _ in range(100):
            world.step(render=True)

        move_gripper_upward(gripper, world, speed=0.01, duration=1.0)

        # TODO: apply shake movement
        # TODO: Check if object is still in gripper
        # TODO: Label grasp (result, sampled point, gripper orientation)
        
    elif not reached_target:
        if count % 50 == 0:
            print("Moving Toward Sampled Point...")
        move_gripper_toward(gripper, target_point)

    # TIMEOUT
    count += 1
    if count > 900: # Timeout
        print("TIMOUT: RESETTING SCENE\n")
        target_point = reset_scene(world, stage, gripper, object, centroid, open_action)
        count = 0
        reached_target = False

    
# Close the simulator
simulation_app.close()


# while trial < max_trials:
#     world.step(render=True)
#     for i in range(300):
#         print('open')
#         for i in range(400):
#                 if i == 100:
#                      gripper.apply_action(open_action)
#                 world.step(render=True)
#         print('close')
#         for i in range(400):
#                 if i == 100:
#                     gripper.apply_action(close_action)
#                 world.step(render=True)
#         print('open')
#         for i in range(400):
#                 if i == 100:
#                     gripper.apply_action(open_action)
#                 world.step(render=True)
#         print('close')
#         for i in range(400):
#                 if i == 100:
#                     gripper.apply_action(close_action)
#                 world.step(render=True)
#         print('reset')
#         # world.reset()
# # Close the simulator
# simulation_app.close()









