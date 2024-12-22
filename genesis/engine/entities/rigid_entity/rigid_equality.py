import taichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.repr_base import RBC


@ti.data_oriented
class RigidEqConnect(RBC):
    """
    A Rigid Connect defines a 3D connection between two bodies at a point (ball joint)
    """

@ti.data_oriented
class RigidEqJoint(RBC):
    """
    A Rigid Joint Equality couples the values of two scalar joints with cubic
    """

# Other equality constraints from mjcf
# weld
# tendon
# flex