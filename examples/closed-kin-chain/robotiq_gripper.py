import argparse
import os
import pickle as pkl

import numpy as np

import genesis as gs
import genesis.engine.entities as gsen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
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
            dt=0.01,
        ),
        viewer_options=viewer_options,
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    gripper: gsen.RigidEntity = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="mujoco_menagerie/robotiq_2f85_v4/2f85.xml",
            pos=(0.0, 0, 0.02),
        ),
    )

    ########################## build ##########################
    scene.build()
    for i in range(1000):
        scene.step()


if __name__ == "__main__":
    main()
