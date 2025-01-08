import argparse
import os
import pickle as pkl

import numpy as np

import genesis as gs
import genesis.engine.entities as gsen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu, debug=args.debug)

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
        show_viewer=args.vis,
        show_FPS=False,
    )

    print(scene.rigid_solver._options)

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    gripper: gsen.RigidEntity = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="mujoco_menagerie/robotiq_2f85_v4/2f85.xml",
            pos=(0.0, 0, 0.0),
        ),
    )

    ########################## build ##########################
    scene.build()
    for i in range(len(scene.rigid_solver.links)):
        print(f"Entity with id {scene.rigid_solver.links[i].idx}: {scene.rigid_solver.links[i].name}")

    j1_id = gripper.get_joint("left_driver_joint").dof_start
    j2_id = gripper.get_joint("right_driver_joint").dof_start

    for i in range(100000000000):
        gripper.control_dofs_force(np.array([np.sin(i / 20) * 30, np.sin(i / 20) * 30]), [j1_id, j2_id])
        scene.step()


if __name__ == "__main__":
    main()
