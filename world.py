# Omniverse Kit & Isaac Sim core
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
from pxr import UsdLux, Sdf, UsdPhysics, Gf, PhysicsSchemaTools, UsdGeom, Usd
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleArticulation

import numpy as np
import random


# Spawn a sphere at random sampled point on object and a cone aligned to normal of point
def visualize_sample(point: np.ndarray, normal: np.ndarray, stage,
                     sphere_path="/World/marker_sphere",
                     cone_path  ="/World/normal_cone",
                     sphere_radius=0.01,
                     cone_length =0.1,
                     cone_radius =0.005):
    # Sphere marker
    sphere = UsdGeom.Sphere.Define(stage, sphere_path) # create sphere
    sphere.GetRadiusAttr().Set(sphere_radius) #radius
    xf_s = UsdGeom.Xformable(sphere.GetPrim()) # gets prim behind object that will let us transform it
    xf_s.ClearXformOpOrder() # removes old rotation and translation cache
    xf_s.AddTranslateOp().Set(Gf.Vec3d(*point.tolist())) # translates sphere to desired location

    # Cone marker
    cone = UsdGeom.Cone.Define(stage, cone_path)
    cone.GetHeightAttr().Set(cone_length)
    cone.GetRadiusAttr().Set(cone_radius)

    # compute rotation normal
    z = np.array([0.0, 0.0, 1.0]) # up direction
    n = normal / np.linalg.norm(normal) # normalize surface normal
    dot = float(np.dot(z, n)) # dot product 
    if abs(dot) < 0.9999:
        axis = np.cross(z, n)
        angle = np.degrees(np.arccos(dot))
        rot  = Gf.Rotation(Gf.Vec3d(*axis.tolist()), angle)
    else:
        rot = Gf.Rotation(Gf.Vec3d(1,0,0), 0 if dot>0 else 180)

    # Plae sphere into place with above normal calcutions
    xf_mat = Gf.Matrix4d().SetRotate(rot)
    xf_mat.SetTranslateOnly(Gf.Vec3d(*point.tolist()))
    xf_c   = UsdGeom.Xformable(cone.GetPrim())
    xf_c.ClearXformOpOrder()
    xf_c.AddTransformOp().Set(xf_mat)

# At the beginning of an attempt, the gripper is spawned at a location n distance from the point sampled from the mesh
# in direction based on the normal of the mesh that we calculate
def sample_approach_pose(point: np.ndarray,
                         normal: np.ndarray,
                         standoff: float = 0.5,
                         roll_deg: float = 0.0):
    """
    Place the gripper at point + normal*standoff, with its local +Z axis
    aligned to -normal (so the fingers approach along the surface normal),
    then optionally roll around that axis.
    Returns (Gf.Vec3d position, [x,y,z,w] orientation).
    """
    # position a bit out along the normal
    n = normal / np.linalg.norm(normal)
    pos = point + n * standoff

    # find rotation that sends +Z → -n (approach vector)
    target = Gf.Vec3d(*(-n).tolist())
    # if your normal is exactly +Z or -Z, the constructor still works:
    rot_to_n = Gf.Rotation(Gf.Vec3d(0,0,1), target)

    # apply a roll about that same axis
    # if abs(roll_deg) > 1e-3:
    #     roll_rot = Gf.Rotation(target, roll_deg)
    #     rot_to_n = roll_rot * rot_to_n

    # extract quaternion
    q = rot_to_n.GetQuaternion()
    ori = [q.GetImaginary()[0],
           q.GetImaginary()[1],
           q.GetImaginary()[2],
           q.GetReal()]

    return Gf.Vec3d(*pos.tolist()), ori



#TODO: Use rays and contact points and distance from center of gripper to know when reach object and should grasp
def move_gripper_toward(gripper: SingleArticulation, target: Gf.Vec3d, step_size=0.001):
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

def reset_scene(gripper, cracker_box):
    # reset the box
    cracker_box.set_world_pose(
        position    = [0.0, 0.0, 1.0],
        orientation = [0.0, 0.0, 0.0, 1.0]
    )

    # sample a new point+normal
    point, normal = sample_point_and_normal("/World/CrackerBox")
    print(f"Resampled point: {point}, normal: {normal}")
    visualize_sample(point, normal, stage)

    # compute a rough “standoff” pose along the normal
    rough_pos, rough_ori = sample_approach_pose(
        point, normal,
        standoff = 0.3,
        roll_deg = random.uniform(-5, 5)
    )

    # restart sim, teleport to that rough pose
    world.reset()
    gripper.initialize()
    gripper.set_world_pose(
        position    = rough_pos,
        orientation = rough_ori
    )

    # get the static root→TCP transform
    tcp_path           = f"{gripper.prim_path}/joints/panda_hand/panda_grasptarget"
    tcp_prim           = stage.GetPrimAtPath(tcp_path)
    xform_cache        = UsdGeom.XformCache()
    static_root_to_tcp = xform_cache.GetLocalToWorldTransform(tcp_prim)

    # build a Matrix4d for our rough root→world
    rot = Gf.Rotation(Gf.Quatd(
        rough_ori[3],  # w
        rough_ori[0],  # x
        rough_ori[1],  # y
        rough_ori[2],  # z
    ))
    root_to_world = Gf.Matrix4d()
    root_to_world.SetRotate(rot)
    root_to_world.SetTranslateOnly(rough_pos)

    # desired world TCP = (rough root→world) × (static root→TCP)
    world_to_tcp = root_to_world * static_root_to_tcp

    # solve for new_root→world so that TCP lands exactly:
    #    new_root→world = world→TCP × inv(static_root→TCP)
    new_root_to_world = world_to_tcp * static_root_to_tcp.GetInverse()

    # extract new root pos & ori
    new_pos = new_root_to_world.ExtractTranslation()
    new_rot = new_root_to_world.ExtractRotation().GetQuaternion()
    new_ori = [
        new_rot.GetImaginary()[0],
        new_rot.GetImaginary()[1],
        new_rot.GetImaginary()[2],
        new_rot.GetReal()
    ]

    # teleport
    gripper.set_world_pose(
        position    = new_pos,
        orientation = new_ori
    )

    # open the gripper
    gripper.apply_action(open_action)
    return point, normal


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

gripper = import_urdf_model(
    "/home/csrobot/Isaac/assets/grippers/franka_panda/franka_panda.urdf",
    position=Gf.Vec3d(0, 0, 0.5), rotation_deg=0)

close_action = ArticulationAction(joint_positions=np.array([0.0, 0.0])) # joint_indices=np.array()) 
open_action = ArticulationAction(joint_positions=np.array([0.04, 0.04])) # joint_indices=np.array())

point, normal = reset_scene(gripper, collision_box)

world.reset()
gripper.initialize()


grasped = False
trial = 0
max_trials = 100
count = 0

while trial < max_trials:
    world.step(render=True)
    move_gripper_toward(gripper, point)
    if count > 250:
            point, normal = reset_scene(gripper, collision_box)
            count = 0
            trial += 1
            print(f"Trial {trial} complete. Resetting scene.")
    count += 1

# Close the simulator
simulation_app.close()
