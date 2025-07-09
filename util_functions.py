from omni.isaac.kit import SimulationApp

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

def move_gripper_upward(
    gripper,        # your SingleArticulation / Robot
    world,          # your omni.isaac.core.World
    speed=0.1,      # meters per second
    duration=2.0,   # total seconds to move
    render=True     # whether to render each step
):
    """
    Moves the gripper straight upward at `speed` m/s for `duration` seconds.
    """
    # estimate timestep (secs) per world.step()
    # Isaac Sim default physics dt is 1/60, but you can also query:
    # dt = world.physics_context.get_physics_dt()
    dt = 1.0 / 60.0

    steps = int(duration / dt)
    steps = 500
    for _ in range(steps):
        # 1) get current pose
        pos, ori = gripper.get_world_pose()      # pos is a list [x,y,z]
        # 2) increment Z
        pos[2] += speed * dt
        # 3) teleport gripper
        gripper.set_world_pose(position=pos, orientation=ori)
        # 4) advance sim
        world.step(render=render)

# TODO: check whether the fingers are in contact with the mesh after shaking
def check_finger_contact():
    return

def reset_scene(world, stage, gripper, model, centroid, open_action):

    model.set_linear_velocity([0.0,0.0,0.0])
    model.set_angular_velocity([0.0,0.0,0.0])

    gripper.set_linear_velocity([0.0,0.0,0.0])
    gripper.set_angular_velocity([0.0,0.0,0.0])

    gripper.set_world_pose(
        position    = [0.0, 15.0, 15.0],
        orientation = [0.0, 0.0, 0.0, 1.0]
    )

    model.set_world_pose(
        position    = [0.0, 0.0, 1.0],
        orientation = [0.0, 0.0, 0.0, 1.0]
    )
    world.step(render=False)

    # Sample single point and lidar pose
    sampled_point, normal = sample_point_and_normal("/World/object")
    view_pos = calculate_approach_pose(sampled_point, normal, standoff = 0.1)

    visualize_point_sample(sampled_point, stage, sphere_path="/World/marker_sphere", sphere_radius=0.005, color=(0,255,0))
    look_at_point("/World/lidar_cone", target_pos=view_pos, point=sampled_point)

    move_gripper_to_lidar(gripper)
    
    # open the gripper
    gripper.apply_action(open_action)

    gripper.set_linear_velocity([0.0,0.0,0.0])
    gripper.set_angular_velocity([0.0,0.0,0.0])
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
# def sample_point_and_normal(xform_path: str):
#     stage = omni.usd.get_context().get_stage()

#     # find the Mesh prim of the desired object
#     root = stage.GetPrimAtPath(xform_path)
#     mesh_prim = None
#     for prim in Usd.PrimRange(root):
#         if prim.IsA(UsdGeom.Mesh):
#             mesh_prim = prim
#             break
#     if mesh_prim is None:
#         raise RuntimeError(f"No Mesh found under {xform_path}")

#     mesh = UsdGeom.Mesh(mesh_prim)

#     # get points and indices
#     verts_attr = mesh.GetPointsAttr().Get()
#     idxs_attr  = mesh.GetFaceVertexIndicesAttr().Get()
#     if verts_attr is None or idxs_attr is None:
#         raise RuntimeError(f"Mesh at {mesh_prim.GetPath()} has no points or indices")

#     # converts points to npy, assuming all faces are triangles
#     verts = np.array([[p[0], p[1], p[2]] for p in verts_attr])
#     tris  = np.array(idxs_attr, dtype=int).reshape(-1, 3)

#     # compute triangles areas and sampling probabilities
#     v0 = verts[tris[:,0]]; v1 = verts[tris[:,1]]; v2 = verts[tris[:,2]]
#     areas = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)
#     probs = areas / areas.sum()

#     # pick one triangle and a random barycentric point (get point on mesh)
#     ti = np.random.choice(len(probs), p=probs)
#     a, b, c = tris[ti]
#     u, v = np.random.rand(), np.random.rand()
#     if u+v > 1.0:
#         u, v = 1-u, 1-v
#     w = 1 - u - v
#     local_pt = w*verts[a] + u*verts[b] + v*verts[c]

#     # compute the face normal of selected triangle
#     local_n = np.cross(verts[b]-verts[a], verts[c]-verts[a])
#     local_n /= np.linalg.norm(local_n)

#     # transform to world, get the point in respect to the world
#     #   so we know where to put the visual markers
#     xform_cache = UsdGeom.XformCache()
#     mat = xform_cache.GetLocalToWorldTransform(mesh_prim)
#     rot3d = mat.ExtractRotationMatrix()

#     # world-space point (uses full 4×4)
#     wp = mat.Transform(Gf.Vec3d(*local_pt))

#     # world-space normal (rotate only via 3×3)
#     rn_gf = rot3d * Gf.Vec3d(*local_n)
    
#     # convert to npy and normalize
#     world_pt = np.array([wp[i] for i in range(3)])
#     world_n  = np.array([rn_gf[i] for i in range(3)])
#     world_n /= np.linalg.norm(world_n)

#     return world_pt, world_n

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

    # get vertex positions, normals, and indices
    verts_attr  = mesh.GetPointsAttr().Get()
    normals_attr = mesh.GetNormalsAttr().Get()
    idxs_attr   = mesh.GetFaceVertexIndicesAttr().Get()
    if not verts_attr or not normals_attr or not idxs_attr:
        raise RuntimeError(f"Mesh at {mesh_prim.GetPath()} missing points, normals, or indices")

    # convert to numpy arrays
    verts     = np.array([[p[0], p[1], p[2]] for p in verts_attr])
    norms     = np.array([[n[0], n[1], n[2]] for n in normals_attr])
    tris      = np.array(idxs_attr, dtype=int).reshape(-1, 3)

    # compute triangle areas for sampling
    v0 = verts[tris[:,0]]; v1 = verts[tris[:,1]]; v2 = verts[tris[:,2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)
    probs = areas / areas.sum()

    # sample a triangle by area and get barycentric coordinates
    ti = np.random.choice(len(probs), p=probs)
    ia, ib, ic = tris[ti]
    u, v = np.random.rand(), np.random.rand()
    if u + v > 1.0:
        u, v = 1 - u, 1 - v
    w = 1 - u - v

    # interpolate position and authored normals
    local_pt = w * verts[ia] + u * verts[ib] + v * verts[ic]
    local_n  = w * norms[ia] + u * norms[ib] + v * norms[ic]
    local_n /= np.linalg.norm(local_n)

    # transform to world space\    
    xform_cache = UsdGeom.XformCache()
    mat4 = xform_cache.GetLocalToWorldTransform(mesh_prim)
    # world-space point
    wp = mat4.Transform(Gf.Vec3d(*local_pt))
    world_pt = np.array([wp[i] for i in range(3)])

    # correctly transform normals under scale/rotation: inverse-transpose of linear part
    M = np.array([[mat4[i][j] for j in range(3)] for i in range(3)], dtype=float)
    world_n = np.linalg.inv(M).T.dot(local_n)
    world_n /= np.linalg.norm(world_n)

    # ensure outward orientation
    centroid = calculate_centroid(xform_path)
    if np.dot(world_n, world_pt - centroid) < 0:
        world_n = -world_n

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