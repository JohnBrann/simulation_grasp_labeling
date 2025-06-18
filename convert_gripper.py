# Omniverse Kit & Isaac Sim core
# Considerations for using a simulated LiDar
from omni.isaac.kit import SimulationApp

# Launch Isaac Sim with GUI
simulation_app = SimulationApp({"headless": False})
import os
# Core Isaac Sim & USD imports
import omni.kit.commands
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.prims import RigidPrim
from isaacsim.asset.importer.urdf import _urdf
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import UsdLux, Sdf, UsdPhysics, Gf, PhysicsSchemaTools, UsdGeom, Usd, UsdShade, PhysxSchema, Tf, UsdUtils
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

assets_dir = "/home/csrobot/Isaac/assets/isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005/Assets/Isaac/4.5/Isaac/Robots/Robotiq/Hand-E"
input_usd  = os.path.join(assets_dir, "Robotiq_Hand_E_base.usd")
output_usd = os.path.join(assets_dir, "Robotiq_Hand_E_convexDecomp_flattened.usd")

stage = Usd.Stage.Open(input_usd)

# 3) for every prim, apply the mesh‐collision API and set convexDecomposition
for prim in stage.Traverse():
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    # this will create the API if needed and return a valid object
    mesh_api.GetApproximationAttr().Set("convexDecomposition")

stage.Export(output_usd)

print(f"✅ Wrote flattened USD to {output_usd}")
