
import torch

class FlipperControl():

    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self._positions = torch.zeros((self.num_envs, 4), device=self.device)

    def _clip_positions(self):
        self._positions = torch.clip(self._positions, -180, 180)

    @property
    def positions(self):
        return self._positions

    def zero(self, index=None):
        if index is not None:
            self._positions[index] = torch.zeros((self.num_envs, 4), device=self.device)[index]
        else:
            self._positions = torch.zeros((self.num_envs, 4), device=self.device)

    def set_pos_dt_with_max(self, pos_dt, max_pos=60, index=None):
        self.set_pos_dt(pos_dt, index)
        self._positions = torch.clip(self._positions, -max_pos, max_pos)
        self._clip_positions()

    def set_pos_dt(self, pos_dt, index=None):
        if index is not None:
            self._positions[index] += pos_dt
        else:
            self._positions += pos_dt

        self._clip_positions()

    def set_pos(self, pos, index=None):
        if index is not None:
            self._positions[index] = pos
        else:
            self._positions = pos
        self._clip_positions()

    def set_pos_with_dt(self, pos, dt=2):
        for i in range(len(pos)):
            for j in range(4):
                if self._positions[i][j] > pos[i][j]:
                    self._positions[i][j] = torch.max(pos[i][j], self._positions[i][j] - dt)
                else:
                    self._positions[i][j] = torch.min(pos[i][j], self._positions[i][j] + dt)
        self._clip_positions()

    def set_pos_with_dt_at_index(self, pos, index, dt=2):
        for j in range(4):
            if self._positions[index][j] > pos[j]:
                self._positions[index][j] = torch.max(pos[j], self._positions[index][j] - dt)
            else:
                self._positions[index][j] = torch.min(pos[j], self._positions[index][j] + dt)
        self._clip_positions()