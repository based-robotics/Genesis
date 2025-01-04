import numpy as np
import taichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.repr_base import RBC
from genesis import EQ_TYPE


@ti.data_oriented
class RigidEquality(RBC):
    """
    A Rigid Equality constraint between links
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
        sol_params,
        _type,
    ):
        self._uid = gs.UID()
        self._entity = entity
        self._solver = entity.solver

        self._active0 = active0
        self._data = np.array(data, dtype=gs.np_float)
        self._id = id
        self._name = name
        self._link1id = link1id
        self._link2id = link2id
        self._sol_params = np.array(sol_params, dtype=gs.np_float)
        self._type = EQ_TYPE(_type)

    # TODO: here should go all functions which compute some dampings and forces and something else

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def link1_jac(self, envs_idx=None) -> np.ndarray:
        return self._solver.get_links_jac([self.link1id], [envs_idx], dof_start=self._entity.dof_start)

    @gs.assert_built
    def link2_jac(self, envs_idx=None) -> np.ndarray:
        return self._solver.get_links_jac([self.link2id], [envs_idx], dof_start=self._entity.dof_start)

    @property
    def dim(self):
        """Get constraint dimensionality"""
        if self._type == EQ_TYPE.CONNECT:
            return 3  # 3D position constraint
        elif self._type == EQ_TYPE.WELD:
            return 6  # 3D position + 3D orientation
        elif self._type == EQ_TYPE.JOINT:
            return 1  # 1D joint constraint
        else:
            return 0

    @property
    def sol_params(self):
        """Get solver parameters for impedance calculation"""
        return self._sol_params

    @property
    def data(self):
        """Get constraint data"""
        return self._data

    @property
    def type(self):
        """Get constraint type"""
        return self._type

    @property
    def link1id_local(self):
        """Get first link's local ID within entity"""
        return self._link1id - self._entity._link_start

    @property
    def link2id_local(self):
        """Get second link's local ID within entity"""
        return self._link2id - self._entity._link_start

    @property
    def link1id(self):
        """Get first link's global ID in solver"""
        return self._link1id

    @property
    def link2id(self):
        """Get second link's global ID in solver"""
        return self._link2id

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
    def id(self):
        """Get the ID."""
        return self._id

    @property
    def name(self) -> str:
        """Get the name."""
        return self._name

    @property
    def solimp(self):
        """Get the solver impedance parameters."""
        return self._solimp

    @property
    def solref(self):
        """Get the solver reference parameters."""
        return self._solref

    @property
    def jac_body1(self):
        """Get Jacobian for first body"""
        return self.link1_jac()

    @property
    def jac_body2(self):
        """Get Jacobian for second body"""
        return self.link2_jac()

    @property
    def invweight(self):
        """Get inverse weight for constraint"""
        return self._solver.links_info[self.link1id].invweight[0] + self._solver.links_info[self.link2id].invweight[0]

    @property
    def n_constraints(self):
        """Get number of scalar constraints this equality represents"""
        return self.dim

    @property
    def is_built(self):
        """
        Returns whether the entity the joint belongs to is built.
        """
        return self.entity.is_built

    def get_error(self, i_b):
        """Get constraint error for batch index i_b"""
        if self._type == EQ_TYPE.CONNECT:
            # Get global positions of anchor points
            p_b1 = self._solver.links_state[self.link1id, i_b].pos
            p_b2 = self._solver.links_state[self.link2id, i_b].pos
            R_b1 = gu.quat_to_R(self._solver.links_state[self.link1id, i_b].quat)
            R_b2 = gu.quat_to_R(self._solver.links_state[self.link2id, i_b].quat)

            anchor1, anchor2 = self._data[0:3], self._data[3:6]
            pos1 = R_b1 @ anchor1 + p_b1
            pos2 = R_b2 @ anchor2 + p_b2
            return pos1 - pos2

        elif self._type == EQ_TYPE.WELD:
            # TODO: Implement WELD error calculation
            pass

        elif self._type == EQ_TYPE.JOINT:
            # TODO: Implement JOINT error calculation
            pass

        return None
