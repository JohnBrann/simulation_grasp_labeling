from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.physics_context import PhysicsContext

PhysicsContext()
GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]))
DynamicCuboid(prim_path="/World/Cube",
    position=np.array([-.5, -.2, 1.0]),
    scale=np.array([.5, .5, .5]),
    color=np.array([.2,.3,0.]))

from isaacsim.sensors.physics import ContactSensor
import numpy as np

sensor = ContactSensor(
    prim_path="/World/Cube/Contact_Sensor",
    name="Contact_Sensor",
    frequency=60,
    translation=np.array([0, 0, 0]),
    min_threshold=0,
    max_threshold=10000000,
    radius=-1
)

import omni
from pxr import PhysxSchema

stage = omni.usd.get_context().get_stage()
parent_prim = stage.GetPrimAtPath("/World/Cube")
contact_report = PhysxSchema.PhysxContactReportAPI.Apply(parent_prim)
# Set a minimum threshold for the contact report to zero
contact_report.CreateThresholdAttr(0.0)


value = sensor.get_current_frame()

print(value)
