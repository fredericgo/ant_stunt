import numpy as np
#import tensorflow_graphics.geometry.transformation as tfg_transformation
from common.quaternion_util import axis_angle_to_quaternion, quaternion_multiply, quaternion_apply
import torch


class SkeletonGeometry:
    def __init__(self, env):
        self._n_joints = env.model.njnt
        self._n_bodies = env.model.nbody
        self._build_skeleton(env)
        
    def _build_skeleton(self, env):
        self._n_bodies = env.model.nbody
        self._n_joints = env.model.njnt
        
        # get the highest body taht has another joint where the current joint is attached to.
        self._topology = []
        self._offsets = []
        for j in range(self._n_joints):
            offset = env.model.jnt_pos[j].copy()
            b = env.model.jnt_bodyid[j]
            offset += env.model.body_pos[b]
            pa = env.model.body_parentid[b]

            wid = env.model.body_weldid[pa]
            while env.model.body_weldid[wid] != wid:
                offset += env.model.body_pos[wid]
                wid = env.model.body_weldid[wid]
            pa = env.model.body_jntadr[wid]
            self._topology.append((j, pa))
            self._offsets.append(offset)
        self._offsets = np.stack(self._offsets).astype(np.float32)
        self._types = env.model.jnt_type
        self._axis = env.model.jnt_axis.astype(np.float32)
        self._jnt_qposadr = env.model.jnt_qposadr
    
    def get_rotations(self, s):
        """
        Input:
            s[A1, A2, ..., nq]: state
        Output:
            M[A1, A2, ..., nq, 4]: quaternions
        """
        # 先通通轉成quaternion好了
        transform = []
        
        for j in range(self._n_joints):
            typ = self._types[j]
            if typ in [0, 1]:
                # 目前只支援root是free joint之後再修改
                start = 1
                q = s[...,start:(start+4)]
                #print(q)
                #q = tf.gather(q, [1, 2, 3, 0], axis=-1)
                transform.append(q)

            elif typ == 3:
                start = self._jnt_qposadr[j] -2
                axis = self._axis[j]
                angle = s[..., start:(start+1)]
                axis_angle = angle * axis
                q = axis_angle_to_quaternion(axis_angle)
                transform.append(q)
            else:
                raise ValueError
        return transform
    
    def get_joint_locations(self, s):
        q = self.get_rotations(s)
        results = []
        r0 = np.zeros(q[0].shape[:-1] + (3,), dtype=np.float32)
        results.append(r0)

        for i, pi in self._topology:
            if pi == -1:
                assert i == 0
                continue
            q[i] = quaternion_multiply(q[pi], q[i])
            p = self._offsets[i]
            res = quaternion_apply(q[i], p)
            res += results[pi]
            results.append(res)
            
        results = np.stack(results)    
        return results