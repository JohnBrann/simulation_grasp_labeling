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
from pxr import UsdLux, Sdf, UsdPhysics, Gf, PhysicsSchemaTools, UsdGeom, Usd, UsdShade
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleArticulation
import omni.replicator.core as rep
import omni
import asyncio
from isaacsim.sensors.physx import _range_sensor 
from omni.kit.viewport.utility import get_active_viewport
import numpy as np
import random

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

    wrist_offset = Gf.Rotation(Gf.Vec3d(1,0,0), noise_angle).GetQuat()
    pitch_offset = Gf.Rotation(Gf.Vec3d(0,1,0), 90.0).GetQuat()
    final_quat = wrist_offset * pitch_offset * base_quat

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
def move_gripper_toward(gripper: SingleArticulation, target: Gf.Vec3d, step_size=0.0005):
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

def reset_scene(stage, gripper, model, centroid):
    # reset the box
    model.set_world_pose(
        position    = [0.0, 0.0, 1.0],
        orientation = [0.0, 0.0, 0.0, 1.0]
    )

    world.reset()
    gripper.initialize()
    world.step(render=False)

    # Sample point around object and look at point
    view_pos = position_lidar(centroid, 0.4)
    # print(f'view pos: {view_pos}')
    # view_pos = np.array([-0.04163335, -0.05821681, 0.81332028])

    # Sample single point and lidar pose
    sampled_point = sample_single_mesh_point("/World/CrackerBox")
    # print(f"sampled point {sampled_point}")
    # sampled_point = np.array([-0.04691, -0.081479, 1.17597795])

    visualize_point_sample(sampled_point, stage, sphere_path="/World/marker_sphere", sphere_radius=0.005, color=(0,255,0))
    look_at_point("/World/lidar_cone", target_pos=view_pos, point=sampled_point)

    move_gripper_to_lidar(gripper)
    
    # open the gripper
    gripper.apply_action(open_action)

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

def sample_single_mesh_point(xform_path: str):
    # get all points, make this more effficient?
    points = get_mesh_points(xform_path)

    if points.size == 0:
        raise ValueError(f"No points found under {xform_path}")
    # pick a random index
    idx = random.randrange(points.shape[0])
    local_pt = points[idx]  # [x, y, z] in mesh-local frame

    #  Transform into world space
    stage = omni.usd.get_context().get_stage()
    xform_cache = UsdGeom.XformCache()
    prim = stage.GetPrimAtPath(xform_path)
    world_mat = xform_cache.GetLocalToWorldTransform(prim)
    world_vec = world_mat.Transform(Gf.Vec3d(*local_pt))
    point = np.array([world_vec[0], world_vec[1], world_vec[2]], dtype=float) 

    return point

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

#### SETUP WORLD ####
# Initialize simulation world
world = World(stage_units_in_meters=1.0)
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

    # Create CrackerBox object
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
world.scene.add(collision_object)

gripper = import_urdf_model(
    "/home/csrobot/Isaac/assets/grippers/franka_panda/franka_panda.urdf",
    position=Gf.Vec3d(0, 0, 0.5), rotation_deg=0)

close_action = ArticulationAction(joint_positions=np.array([0.0, 0.0])) # joint_indices=np.array()) 
open_action = ArticulationAction(joint_positions=np.array([0.04, 0.04])) # joint_indices=np.array())


# Create Lidar cone
cone = UsdGeom.Cone.Define(stage, "/World/lidar_cone")
cone.GetHeightAttr().Set(0.1)
cone.GetRadiusAttr().Set(0.02)
# Rotate cone to start
xf_lidar_cone = UsdGeom.XformCommonAPI(cone.GetPrim())
xf_lidar_cone.SetRotate(Gf.Vec3f(-90.0, 0.0, 0.0)) 

# Add Lidar
timeline = omni.timeline.get_timeline_interface()   
lidarInterface = _range_sensor.acquire_lidar_sensor_interface() 

omni.kit.commands.execute('AddPhysicsSceneCommand',stage = stage, path='/World/PhysicsScene')
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
centroid = calculate_centroid("/World/CrackerBox")
print(f"Centroid: {centroid}")
visualize_point_sample(centroid, stage, sphere_radius=0.005, color=(0,255,0))

# Trials!!!!
trial = 0
max_trials = 100
count = 0

target_point = reset_scene(stage, gripper, collision_object, centroid)

while trial < max_trials:
    world.step(render=True)
    move_gripper_toward(gripper, target_point)
    # depth = lidarInterface.get_linear_depth_data(lidarPath)
    # print("depth", depth)   

    if count > 700:
            target_point = reset_scene(stage, gripper, collision_object, centroid)
            count = 0
            trial += 1
            print(f"Trial {trial} complete. Resetting scene.")
    count += 1
# Close the simulator
simulation_app.close()
