import numpy as np
import mujoco as mj
from mujoco import viewer


def main():
    gripper_path = "mujoco_menagerie/robotiq_2f85_v4/scene.xml"
    mj_model = mj.MjModel.from_xml_path(gripper_path)

    mj_data = mj.MjData(mj_model)
    # mj_viewer = viewer.launch_passive(
    #     mj_model,
    #     mj_data,
    #     show_left_ui=False,
    #     show_right_ui=False,
    # )
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
        with np.printoptions(suppress=True, linewidth=500):
            print(jacp1)
        mj.mj_jacBody(mj_model, mj_data, jacp2, jacr2, id2)
        with np.printoptions(suppress=True, linewidth=500):
            print(jacp2)


if __name__ == "__main__":
    main()
