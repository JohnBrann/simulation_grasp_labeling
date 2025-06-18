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
# from isaacsim.core.api.action import ArticulationAction
import os

def visualize_point_sample(point: np.ndarray, stage, sphere_path="/World/marker_sphere", sphere_radius=0.005, color=(0,255,0)):
    # Sphere marker
    color_normalized = [c / 255.0 for c in color]
    sphere = UsdGeom.Sphere.Define(stage, sphere_path) # create sphere
    sphere.GetRadiusAttr().Set(sphere_radius) #radius
    xf_s = UsdGeom.Xformable(sphere.GetPrim()) # gets prim behind object that will let us transform it
    xf_s.ClearXformOpOrder() # removes old rotation and translation cache
    xf_s.AddTranslateOp().Set(Gf.Vec3d(*point.tolist())) # translates sphere to desired location
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color_normalized)])

def calculate_centroid(xform_path: str):
    points = get_mesh_points(xform_path)
    
    # Calulate Centroid locally
    if points.size == 0:
        raise ValueError("Points is empty")
    local_centroid = points.mean(axis=0) 

    # Translate local centroid based on world location of mesh
    local_vec = Gf.Vec3d(*local_centroid)
    stage = omni.usd.get_context().get_stage()
    xform_cache = UsdGeom.XformCache()
    prim = stage.GetPrimAtPath(xform_path)
    world_mat = xform_cache.GetLocalToWorldTransform(prim)

    # Tranform into world space
    world_vec = world_mat.Transform(local_vec)
    centroid = np.array([world_vec[0], world_vec[1], world_vec[2]], dtype=float)

    return centroid

# get a point for li
def position_lidar(point: np.ndarray, radius=0.5):
    # z = cos(theta) uniform in [-1,1]
    z = 2*np.random.rand() - 1
    phi = 2*np.pi*np.random.rand()
    r_xy = np.sqrt(1 - z*z) * radius
    x = r_xy * np.cos(phi)
    y = r_xy * np.sin(phi)
    return centroid + np.array([x, y, z*radius], dtype=float)

def calculate_approach_pose(point: np.ndarray, normal: np.ndarray, standoff: float = 0.5):
    # position a bit out along the normal
    n = normal / np.linalg.norm(normal)
    pos = point + n * standoff
    return Gf.Vec3d(*pos.tolist())


def look_at_point(cone_path: str, target_pos: np.ndarray, point: np.ndarray):
    stage = omni.usd.get_context().get_stage()
    
    prim  = stage.GetPrimAtPath(cone_path)
    xf    = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()

    mat = Gf.Matrix4d()

    # → rotation: align +Z → (point - target_pos)
    dir_vec  = Gf.Vec3d(*(point - target_pos))
    dir_norm = dir_vec.GetNormalized()

    forward = Gf.Vec3d(0.0, 0.0, 1.0)  # cone’s “tip” axis
    dot     = np.clip(Gf.Dot(forward, dir_norm), -1.0, 1.0)

    axis = Gf.Cross(forward, dir_norm)
    if axis.GetLength() < 1e-6:
        axis = Gf.Vec3d(1.0, 0.0, 0.0)
    else:
        axis = axis.GetNormalized()

    angle = np.degrees(np.arccos(dot))
    mat.SetRotate(Gf.Rotation(axis, angle))

    # move origin to target_pos
    mat.SetTranslateOnly(Gf.Vec3d(*target_pos))

    xf.AddTransformOp().Set(mat)

def move_gripper_to_lidar(gripper: SingleArticulation,
                          lidar_path: str = "/World/lidar_cone/Lidar"):

    stage = omni.usd.get_context().get_stage()
    xform_cache = UsdGeom.XformCache()
    prim = stage.GetPrimAtPath(lidar_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"No valid prim at {lidar_path}")

    # Get the full local→world transform
    world_mat = xform_cache.GetLocalToWorldTransform(prim)

    # Extract translation
    trans = world_mat.ExtractTranslation()  # Gf.Vec3d

    # 3) Extract rotation as quaternion
    base_quat = world_mat.ExtractRotation().GetQuat()  # Gf.Quatd

    angle_range = 90
    noise_angle = random.choice(range(-angle_range, angle_range, 10))
    noise_angle = 0

    # FOR HAND-E gripper
    wrist_offset = Gf.Rotation(Gf.Vec3d(0,0,1), noise_angle).GetQuat()
    pitch_offset = Gf.Rotation(Gf.Vec3d(1,0,0), -90.0).GetQuat()
    final_quat = pitch_offset * wrist_offset * base_quat

    # FOR Panda Gripper
    # wrist_offset = Gf.Rotation(Gf.Vec3d(1,0,0), noise_angle).GetQuat()
    # pitch_offset = Gf.Rotation(Gf.Vec3d(0,1,0), 90.0).GetQuat()
    # final_quat = wrist_offset * pitch_offset * base_quat

    # final_quat = pitch_quat * noise
    ori = [
        final_quat.GetImaginary()[0],
        final_quat.GetImaginary()[1],
        final_quat.GetImaginary()[2],
        final_quat.GetReal()
    ]   

    # 4) Teleport your gripper
    gripper.set_world_pose(
        position    = [trans[0], trans[1], trans[2]],
        orientation = ori
    )

#TODO: Use rays and contact points and distance from center of gripper to know when reach object and should grasp
def move_gripper_toward(gripper: SingleArticulation, target: Gf.Vec3d, step_size=0.0001):
    current_pos, _ = gripper.get_world_pose()
    direction = np.array([target[0], target[1], target[2]]) - np.array(current_pos)
    distance = np.linalg.norm(direction)
    # if proximity_distance < step_size:
    #     return True  # Reached
    direction = direction / distance  # Normalize
    new_pos = np.array(current_pos) + direction * step_size
    pose_mat = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*new_pos))
    gripper.set_world_pose(position=new_pos)
    # return False

def check_reached_target(proximity_distance, reach_threshold= 0.015):
    if proximity_distance == None:
        return False
    elif proximity_distance < reach_threshold:
        return True
    return False 


def apply_swing_shake(gripper: SingleArticulation, world: World,
    amplitude_rad: float = 0.15, num_steps: int = 40):
    
    # current world pose once, before shaking:
    base_pos, base_ori = gripper.get_world_pose()
    
    qx, qy, qz, qw = base_ori
    base_quat = Gf.Quatd(
        float(qw),  # real
        float(qx),  # imag.x
        float(qy),  # imag.y
        float(qz)   # imag.z
    )

    # Store pivot position so the gripper doesn’t drift
    pivot_pos = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]

    for i in range(num_steps):
        theta = amplitude_rad * math.sin(2 * math.pi * i / num_steps)
        deg = float(theta * (180.0 / math.pi))
        axis = Gf.Vec3d(0.0, 0.0, 1.0)
        yaw_delta_quat = Gf.Rotation(axis, deg).GetQuat()  # returns Gf.Quatd

        # Q_new = base_quat * yaw_delta_quat (both are Quatd)
        new_quat = Gf.Quatd(base_quat)  # ensure we have a Quatd copy
        new_quat = new_quat * yaw_delta_quat

        # Teleport the gripper to pivot_pos with orientation=new_quat
        gripper.set_world_pose(
            position = pivot_pos,
            orientation = [
                new_quat.GetImaginary()[0],  # qx
                new_quat.GetImaginary()[1],  # qy
                new_quat.GetImaginary()[2],  # qz
                new_quat.GetReal()           # qw
            ]
        )

        # Step physics so the shake is visible
        world.step(render=True)

    # Restore exactly to the original orientation Q0:
    restore_quat = [
        base_quat.GetImaginary()[0],
        base_quat.GetImaginary()[1],
        base_quat.GetImaginary()[2],
        base_quat.GetReal()
    ]
    gripper.set_world_pose(position=pivot_pos, orientation=restore_quat)
    world.step(render=True)

# TODO: check whether the fingers are in contact with the mesh after shaking
def check_finger_contact():
    return

def reset_scene(stage, gripper, model, centroid):
    # reset the box
    model.set_world_pose(
        position    = [0.0, 0.0, 1.0],
        orientation = [0.0, 0.0, 0.0, 1.0]
    )

    world.reset()
    
    # Sample point around object and look at point
    # view_pos = position_lidar(centroid, 0.4)

    # Sample single point and lidar pose
    sampled_point, normal = sample_point_and_normal("/World/object")
    view_pos = calculate_approach_pose(sampled_point, normal, standoff = 0.1)

    visualize_point_sample(sampled_point, stage, sphere_path="/World/marker_sphere", sphere_radius=0.005, color=(0,255,0))
    look_at_point("/World/lidar_cone", target_pos=view_pos, point=sampled_point)

    move_gripper_to_lidar(gripper)
    
    # open the gripper
    gripper.apply_action(open_action)
    world.step(render=False)

    return sampled_point

# TODO: Get points once and store them into a list 
def get_mesh_points(xform_path: str):
    stage = omni.usd.get_context().get_stage()

    # find the Mesh prim of the desired object
    root = stage.GetPrimAtPath(xform_path)
    mesh_prim = None
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            mesh_prim = prim
            break
    if mesh_prim is None:
        raise RuntimeError(f"No Mesh found under {xform_path}")

    mesh = UsdGeom.Mesh(mesh_prim)

    # get points and indices
    verts_attr = mesh.GetPointsAttr().Get()
    idxs_attr  = mesh.GetFaceVertexIndicesAttr().Get()
    if verts_attr is None or idxs_attr is None:
        raise RuntimeError(f"Mesh at {mesh_prim.GetPath()} has no points or indices")

    # converts points to npy, assuming all faces are triangles
    verts = np.array([[p[0], p[1], p[2]] for p in verts_attr])
    tris  = np.array(idxs_attr, dtype=int).reshape(-1, 3)

    # print(f'Verts {verts.shape}')
    return verts

# def sample_single_mesh_point(xform_path: str):
#     # get all points, make this more effficient?
#     points = get_mesh_points(xform_path)

#     if points.size == 0:
#         raise ValueError(f"No points found under {xform_path}")
#     # pick a random index
#     idx = random.randrange(points.shape[0])
#     local_pt = points[idx]  # [x, y, z] in mesh-local frame

#     #  Transform into world space
#     stage = omni.usd.get_context().get_stage()
#     xform_cache = UsdGeom.XformCache()
#     prim = stage.GetPrimAtPath(xform_path)
#     world_mat = xform_cache.GetLocalToWorldTransform(prim)
#     world_vec = world_mat.Transform(Gf.Vec3d(*local_pt))
#     point = np.array([world_vec[0], world_vec[1], world_vec[2]], dtype=float) 

#     return point

# Sample a random point on 
def sample_point_and_normal(xform_path: str):
    stage = omni.usd.get_context().get_stage()

    # find the Mesh prim of the desired object
    root = stage.GetPrimAtPath(xform_path)
    mesh_prim = None
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            mesh_prim = prim
            break
    if mesh_prim is None:
        raise RuntimeError(f"No Mesh found under {xform_path}")

    mesh = UsdGeom.Mesh(mesh_prim)

    # get points and indices
    verts_attr = mesh.GetPointsAttr().Get()
    idxs_attr  = mesh.GetFaceVertexIndicesAttr().Get()
    if verts_attr is None or idxs_attr is None:
        raise RuntimeError(f"Mesh at {mesh_prim.GetPath()} has no points or indices")

    # converts points to npy, assuming all faces are triangles
    verts = np.array([[p[0], p[1], p[2]] for p in verts_attr])
    tris  = np.array(idxs_attr, dtype=int).reshape(-1, 3)

    # compute triangles areas and sampling probabilities
    v0 = verts[tris[:,0]]; v1 = verts[tris[:,1]]; v2 = verts[tris[:,2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)
    probs = areas / areas.sum()

    # pick one triangle and a random barycentric point (get point on mesh)
    ti = np.random.choice(len(probs), p=probs)
    a, b, c = tris[ti]
    u, v = np.random.rand(), np.random.rand()
    if u+v > 1.0:
        u, v = 1-u, 1-v
    w = 1 - u - v
    local_pt = w*verts[a] + u*verts[b] + v*verts[c]

    # compute the face normal of selected triangle
    local_n = np.cross(verts[b]-verts[a], verts[c]-verts[a])
    local_n /= np.linalg.norm(local_n)

    # transform to world, get the point in respect to the world
    #   so we know where to put the visual markers
    xform_cache = UsdGeom.XformCache()
    mat = xform_cache.GetLocalToWorldTransform(mesh_prim)
    rot3d = mat.ExtractRotationMatrix()

    # world-space point (uses full 4×4)
    wp = mat.Transform(Gf.Vec3d(*local_pt))

    # world-space normal (rotate only via 3×3)
    rn_gf = rot3d * Gf.Vec3d(*local_n)
    
    # convert to npy and normalize
    world_pt = np.array([wp[i] for i in range(3)])
    world_n  = np.array([rn_gf[i] for i in range(3)])
    world_n /= np.linalg.norm(world_n)

    return world_pt, world_n

# TODO:function to see if the gripper grasp target area has made contact with the object
def grasp_threshold_reached():
    if threshold_reached:
        threshold_reached = True
    else:
        threshold_reached = False
    return threshold_reached

# function that if the object is grasped object is still within gripper
# TODO: move and shake the object aroudn to see if the object stays in the gripper or not
def grasp_stability_test():
    print(f"Shake test")

# TODO: label model a success or failure
def label_model(success):
    print(f"label the model in h5 format, look at ACRONYM")

# Function to import a URDF robot
def import_urdf_model(urdf_path: str, position=Gf.Vec3d(0.0, 0.0, 0.0), rotation_deg=90):
    import_config = _urdf.ImportConfig()
    import_config.convex_decomp = False # Disable convex decomposition for simplicity
    import_config.fix_base = True # Fix the base of the robot to the ground
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

    robot = SingleArticulation(prim_path=prim_path, name="panda")
    world.scene.add(robot)
    return robot

# ─────────────────── SETUP WORLD & SCENE ─────────────────────────────────────
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

# Physics scene & zero gravity
scene = UsdPhysics.Scene.Get(stage, Sdf.Path("/physicsScene"))
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(0.0)

# Add ground plane
PhysicsSchemaTools.addGroundPlane(stage, "/World/groundPlane", "Z", 15,
                                Gf.Vec3f(0,0,0), Gf.Vec3f(0.7))

# Dome light
dome_path = Sdf.Path("/World/DomeLight")
if not stage.GetPrimAtPath(dome_path):
    dome = UsdLux.DomeLight.Define(stage, dome_path)
    dome.CreateIntensityAttr(750.0)
    dome.CreateColorAttr((1.0, 1.0, 1.0))

# ─────────────────── OBJECT ─────────────────────────────────────────────
create_prim(
    prim_path="/World/object",
    prim_type="Xform",
    # usd_path="/home/csrobot/Isaac/assets/ycb/056_tennis_ball/textured.usdc"
    # usd_path="/home/csrobot/Isaac/assets/ycb/mustard/006_mustard_bottle.usd"
    # usd_path="/home/csrobot/Isaac/assets/ycb/engine/engine.usd"
    usd_path="/home/csrobot/Isaac/assets/ycb/009_gelatin_box.usd"
)
object = RigidPrim(
    prim_path="/World/object",
    name="cracker_collision",
    position=[0.0, 0.0, 1.0],
    orientation=[1.0, 0.0, 0.0, 0.0],
    scale=[1.0, 1.0, 1.0],
)
UsdPhysics.CollisionAPI.Apply(object.prim)
world.scene.add(object)


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



# finger_paths = [
#     "/robotiq_gripper/base_link",
#     "/robotiq_gripper/right_gripper",
#     "/robotiq_gripper/left_gripper",
#     "/robotiq_gripper/right_gripper/D_A03_ASM_DOIGTS_PARALLELES_1ROBOTIQ_HAND_E_DEFEATURE",
#     "/robotiq_gripper/right_gripper/D_A03_ASM_DOIGTS_PARALLELES_1ROBOTIQ_HAND_E_DEFEATURE_01",
#     "/robotiq_gripper/right_gripper/D_A03_ASM_DOIGTS_PARALLELES_1ROBOTIQ_HAND_E_DEFEATURE_02",
#     "/robotiq_gripper/left_gripper/D_A03_ASM_DOIGTS_PARALLELES_1ROBOTIQ_HAND_E_DEFEATURE",
#     "/robotiq_gripper/left_gripper/D_A03_ASM_DOIGTS_PARALLELES_1ROBOTIQ_HAND_E_DEFEATURE_01",
#     "/robotiq_gripper/left_gripper/D_A03_ASM_DOIGTS_PARALLELES_1ROBOTIQ_HAND_E_DEFEATURE_02",
#     "/robotiq_gripper/base_link/toothed_lock_washer_metric_robotiq_11ROBOTIQ_HAND_E_DEFEATURE",
#     "/robotiq_gripper/base_link/Group3ROBOTIQ_HAND_E_DEFEATURE",
#     "/robotiq_gripper/base_link/socket_head_cap_screw_iso_robotiq_6ROBOTIQ_HAND_E_DEFEATURE",
#     "/robotiq_gripper/base_link/toothed_lock_washer_metric_robotiq_13ROBOTIQ_HAND_E_DEFEATURE",
#     "/robotiq_gripper/base_link/Group3ROBOTIQ_HAND_E_DEFEATURE_01",
#     "/robotiq_gripper/base_link/toothed_lock_washer_metric_robotiq_10ROBOTIQ_HAND_E_DEFEATURE",
#     "/robotiq_gripper/base_link/socket_head_cap_screw_iso_robotiq_24ROBOTIQ_HAND_E_DEFEATURE",
#     "/robotiq_gripper/base_link/Group3ROBOTIQ_HAND_E_DEFEATURE_02",
#     "/robotiq_gripper/base_link/toothed_lock_washer_metric_robotiq_12ROBOTIQ_HAND_E_DEFEATURE",
#     "/robotiq_gripper/base_link/Group3ROBOTIQ_HAND_E_DEFEATURE_03",
#     "/robotiq_gripper/base_link/socket_head_cap_screw_iso_robotiq_7ROBOTIQ_HAND_E_DEFEATURE",
#     "/robotiq_gripper/base_link/socket_head_cap_screw_iso_robotiq_25ROBOTIQ_HAND_E_DEFEATURE",
#     "/robotiq_gripper/base_link/Group3ROBOTIQ_HAND_E_DEFEATURE_04",
#     "/robotiq_gripper/base_link/Group2ROBOTIQ_HAND_E_DEFEATURE",
#     "/robotiq_gripper/base_link/Group3ROBOTIQ_HAND_E_DEFEATURE_05",
#     "/robotiq_gripper/base_link/Group1ROBOTIQ_HAND_E_DEFEATURE",
# ]

# # def disable_root_collision(path):
# #     prim = stage.GetPrimAtPath(path)
# #     if not prim or not prim.IsValid():
# #         return
# #     # turn off the default Xform‐level collider
# #     api = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath()) or UsdPhysics.CollisionAPI.Apply(prim)
# #     api.GetCollisionEnabledAttr().Set(False)

# # def configure_mesh(prim):
# #     # 1) USD collision schema
# #     UsdPhysics.CollisionAPI.Apply(prim)
# #     # 2) PhysX collision schema
# #     physx_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
# #     # 3) swap to convex‐decomposition (legal on dynamics)
# #     setCollider(prim, "convexDecomposition")
# #     # 4) tighten margins: 1 mm skin, 0 mm rest
# #     physx_api.GetContactOffsetAttr().Set(1e-3)
# #     physx_api.GetRestOffsetAttr() .Set(0.0)

# # # --- A) disable the “ghost” colliders on the two root Xforms ---
# # disable_root_collision("/World/object")
# # disable_root_collision("/robotiq_gripper/base_link")

# # # --- B) walk only the actual mesh prims under both hierarchies ---
# # for root_path in ("/World/object", "/robotiq_gripper"):
# #     root = stage.GetPrimAtPath(root_path)
# #     if not root:
# #         continue
# #     for prim in Usd.PrimRange(root):
# #         if prim.IsA(UsdGeom.Mesh):
# #             configure_mesh(prim)





# for path in finger_paths:
#     prim = stage.GetPrimAtPath(path)
#     if not prim or not prim.IsValid():
#         print(f"⚠️ could not find prim at {path}")
#         continue

#     # 1) USD Collision API  
#     UsdPhysics.CollisionAPI.Apply(prim)

#     # 2) PhysX Collision API (returns the API object, so we can use it immediately)
#     physx_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)

#     # 3) switch to SDF (or "convexDecomposition")
#     setCollider(prim, "convexDecomposition")

#     # 4) zero out the padding on *this* API object
#     physx_api.GetContactOffsetAttr().Set(0.001)
#     physx_api.GetRestOffsetAttr() .Set(0.0)


# wrap the prim as a robot (articulation)
gripper = Robot(
    prim_path=gripper_prim_path,
    name="Hand-E",
    position=[0.0, 0.5, 0.5],       
)

# prim = stage.GetPrimAtPath(gripper_prim_path)
# rb = UsdPhysics.RigidBodyAPI.Apply(prim)
# rb.GetKinematicEnabledAttr().Set(True)

# FRANKA GRIPPER ROTATION
# Create and apply full transform (rotation + translation)
# rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), 0.0)  # Rotate around Z axis
# mat = Gf.Matrix4d().SetRotate(rot)
# # mat.SetTranslateOnly(position)

# xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
# xform.ClearXformOpOrder()
# xform.AddTransformOp().Set(mat)
#####################

# add the Robot into the physics scene
world.scene.add(gripper)


# ─────────────────── RUBBER PAD MATERIAL ────────────────────────────────────
mat_path   = Sdf.Path("/World/Materials/RubberPad")
rubber_mat = UsdShade.Material.Define(stage, mat_path)

# Preview Surface shader
shader_path = mat_path.AppendChild("PreviewShader")
shader = UsdShade.Shader.Define(stage, shader_path)
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
shader.CreateInput("metallic",      Sdf.ValueTypeNames.Float).Set(0.0)
shader.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(0.8)

# Connect shader → material
surface_output = shader.GetOutput("surface")
mat_output     = rubber_mat.CreateSurfaceOutput()
mat_output.ConnectToSource(surface_output)

# Physics: high friction, max‐combine
phys_api = UsdPhysics.MaterialAPI.Apply(rubber_mat.GetPrim())
phys_api.CreateStaticFrictionAttr(0.8)
phys_api.CreateDynamicFrictionAttr(0.8)
phys_api.CreateRestitutionAttr(0.0)
physx_api = PhysxSchema.PhysxMaterialAPI.Apply(rubber_mat.GetPrim())
physx_api.CreateFrictionCombineModeAttr().Set("max")


# Create Gripper Pads and attach to gripper
left_gripper_pad = UsdGeom.Cube.Define(stage, "/robotiq_gripper/left_gripper/left_gripper_pad")
xformLeftApi = UsdGeom.XformCommonAPI(left_gripper_pad)
xformLeftApi.SetScale    (Gf.Vec3f(0.01, 0.01, 0.0007))
xformLeftApi.SetTranslate(Gf.Vec3d(0.0, 0.05, -0.025))

right_gripper_pad = UsdGeom.Cube.Define(stage, "/robotiq_gripper/right_gripper/right_gripper_pad")
xformRightApi = UsdGeom.XformCommonAPI(right_gripper_pad)
xformRightApi.SetScale    (Gf.Vec3f(0.01, 0.01, 0.0007))
xformRightApi.SetTranslate(Gf.Vec3d(0.0, 0.05, 0.025))

# Bind material to gripper pads
for finger in ["left_gripper", "right_gripper"]:
    fp = f"{gripper_prim_path}/{finger}/{finger}_pad"
    prim = stage.GetPrimAtPath(fp)
    if prim and prim.IsValid():
        UsdShade.MaterialBindingAPI(prim).Bind(
            rubber_mat, UsdShade.Tokens.strongerThanDescendants
        )
        UsdPhysics.CollisionAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
    else:
        print(f"⚠️ Could not bind rubber pad to {fp!r}")

world.reset()
world.step(render=False)

# ─────────────────── JOINTS & DRIVES ────────────────────────────────────────
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


for dof in ["Slider_1","Slider_2"]:
    jp    = stage.GetPrimAtPath(Sdf.Path(f"{gripper_prim_path}/{dof}"))
    drive = UsdPhysics.DriveAPI.Apply(jp, "linear")
    drive.CreateStiffnessAttr(1000.0)             # N/m
    drive.CreateDampingAttr(50.0)           
    drive.CreateMaxForceAttr(70.0)         # max N·s/m
    drive.CreateTargetVelocityAttr(0.001) 
    PhysxSchema.PhysxJointAPI.Apply(jp)

efforts = np.full(len(slider_idxs), -0.005, dtype=np.float32)
close_action = ArticulationAction(
    joint_efforts = efforts,
    joint_indices = slider_idxs
)

open_action = ArticulationAction(
    joint_efforts = np.full(len(slider_idxs), +0.01, dtype=np.float32),
    joint_indices = slider_idxs
)

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

# Calculate centroid of object
centroid = calculate_centroid("/World/object")
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
simulation_app.update()
simulation_app.update()

# ─────────────────── CONTACT SENSOR ──────────────────────────────────
# left_pad_path  = "/robotiq_gripper/left_gripper/left_gripper_pad"
# right_pad_path = "/robotiq_gripper/right_gripper/right_gripper_pad"

# # 4) Create & configure the ContactSensor wrappers
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

# # 5) Register sensors with the world
# world.scene.add_sensor(left_sensor)
# world.scene.add_sensor(right_sensor)


# ─────────────────── INITIALIZE SIMULATION ──────────────────────────────────
world.reset()
world.step()


world.play()

# Trials!!!!
trial = 0
max_trials = 100
count = 0
reached_target = False

target_point = reset_scene(stage, gripper, object, centroid)

while trial < max_trials:
    world.step(render=True)
    # print(f'proximity distance: {proximity_distance}') 
    reached_target = check_reached_target(proximity_distance, reach_threshold=0.01)
    
   
    if reached_target:
        print('reached target')
        # world.step(render=False)
        world.step(render=True)
        for _ in range(400):
            gripper.apply_action(close_action)
            # left_contacts  = left_contact_sensor.get_contacts()  # list of Contact objects
            # right_contacts = right_contact_sensor.get_contacts()

            # if left_contacts:
            #     print(f"Left pad contacts:", left_contacts)
            # if right_contacts:
            #     print(f"Right pad contacts:", right_contacts)

            world.step(render=True)

        target_point = reset_scene(stage, gripper, object, centroid)
        # if object_in_gripper():
        reached_target = False
        count = 0
        print('end reached target')
    else:
        move_gripper_toward(gripper, target_point)
    
    # depth = lidarInterface.get_linear_depth_data(lidarPath)
    # print("depth", depth)  
    
    if count > 2000:
        target_point = reset_scene(stage, gripper, object, centroid)
        count = 0
    count += 1
# Close the simulator
simulation_app.close()

