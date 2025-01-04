import numpy as np
import mujoco as mj
from mujoco import viewer
import argparse
import os
import pickle as pkl

import numpy as np

import genesis as gs
import genesis.engine.entities as gsen


def main():
    gripper_path = "mujoco_menagerie/robotiq_2f85_v4/scene.xml"
    mj_model = mj.MjModel.from_xml_path(gripper_path)

    mj_data = mj.MjData(mj_model)

    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(2.5, 0.0, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
        ),
        viewer_options=viewer_options,
        show_viewer=False,
        show_FPS=False,
    )

    ########################## entities ##########################
    scene.add_entity(
        gs.morphs.Plane(),
    )
    scene.add_entity(
        morph=gs.morphs.MJCF(
            file="mujoco_menagerie/robotiq_2f85_v4/2f85.xml",
            pos=(0.0, 0, 0.02),
        ),
    )

    ########################## build ##########################
    scene.build()
    print("-" * 20, "Jacobians", "-" * 20)
    print("mjx:")

    jacp1, jacr1 = np.zeros((3, mj_model.nv)), np.zeros((3, mj_model.nv))
    jacp2, jacr2 = np.zeros((3, mj_model.nv)), np.zeros((3, mj_model.nv))
    for i in range(1):
        id1, id2 = mj_model.eq_obj1id[0], mj_model.eq_obj2id[0]
        print(id1, id2)
        print(
            mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_BODY, id1),
            mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_BODY, id2),
        )
        mj.mj_step(mj_model, mj_data)
        mj.mj_jacBody(mj_model, mj_data, jacp1, jacr1, id1)
        mj.mj_jacBody(mj_model, mj_data, jacp2, jacr2, id2)
        with np.printoptions(suppress=True, linewidth=500):
            print("b1")
            print(jacp1)
            print("b2")
            print(jacp2)
    print("genesis:")
    scene.step()


if __name__ == "__main__":
    main()
