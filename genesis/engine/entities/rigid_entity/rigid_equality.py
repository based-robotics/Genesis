import taichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.repr_base import RBC


@ti.data_oriented
class RigidEquality(RBC):
    """
    A Rigid Equality
    """

    def __init__(
        self,
        entity,
        active0,
        data,
        id,
        name,
        obj1id,
        obj2id,
        solimp,
        solref,
        _type,
    ):
        self._uid = gs.UID()
        self._entity = entity
        self._solver = entity.solver

        self._active0 = active0
        self._data = data
        self._id = id
        self._name = name
        self._obj1id = obj1id
        self._obj2id = obj2id
        self._solimp = solimp
        self._solref = solref
        self._type = _type

    # TODO: here should go all functions which compute some dampings and forces and something else
