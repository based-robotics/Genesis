import taichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.repr_base import RBC
from genesis import EQ_TYPE


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
        link1id,
        link2id,
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
        self._link1id = link1id
        self._link2id = link2id
        self._solimp = solimp
        self._solref = solref
        self._type = EQ_TYPE(_type)

    # TODO: here should go all functions which compute some dampings and forces and something else

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def dim(self) -> int:
        if self._type == EQ_TYPE.CONNECT:
            return 6
        if self._type == EQ_TYPE.WELD:
            return 3
        if self._type == EQ_TYPE.JOINT:
            return 1

    @property
    def uid(self) -> gs.UID:
        """Get the unique identifier."""
        return self._uid

    @property
    def entity(self):
        """Get the entity."""
        return self._entity

    @property
    def solver(self):
        """Get the solver."""
        return self._solver

    @property
    def active0(self):
        """Get the initial active state."""
        return self._active0

    @property
    def data(self):
        """Get the data."""
        return self._data

    @property
    def id(self):
        """Get the ID."""
        return self._id

    @property
    def name(self) -> str:
        """Get the name."""
        return self._name

    @property
    def link1id(self):
        """Get the first linkect ID."""
        return self._link1id

    @property
    def link2id(self):
        """Get the second linkect ID."""
        return self._link2id

    @property
    def solimp(self):
        """Get the solver impedance parameters."""
        return self._solimp

    @property
    def solref(self):
        """Get the solver reference parameters."""
        return self._solref

    @property
    def type(self):
        """Get the type."""
        return self._type
