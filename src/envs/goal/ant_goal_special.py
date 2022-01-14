import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from pathlib import Path

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
xml_file = str(Path(__file__).resolve().parent / "assets" / "ant_goal.xml")


class AntGoalSpEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)
    
    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.25 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (ob, 0, done, dict(),)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        nq = self.model.nq
        noise = 0.2

        qpos = self.init_qpos.copy()
        qpos[2] = .26
        qpos[3:7] = np.array([0, 0, 1, 0])
        qpos[7:15] = np.array([0.,  1,   0.,   -1.,   0.,   -1.,   0.,  1.])
        qpos += np.random.randn(nq) * noise
        qvel = self.init_qvel.copy() + np.random.randn(self.model.nv) * noise
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * .5