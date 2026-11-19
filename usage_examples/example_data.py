"""Synthetic data for the README usage examples: one year of daily values at six stations.
"""

import numpy as np

rng = np.random.default_rng(2026)
n_days = 365
t = np.arange(n_days)
season = np.sin(2 * np.pi * (t - 110) / 365)  # peaks in late July

stnslist = ['A01', 'B02', 'C03', 'D04', 'E05', 'F06']
depth = [12, 18, 25, 31, 40, 52]  # bottom depth [m]

TEMP_SURFACE = {'gain': [1.4, 0.75, 2.6, 0.45, 1.1, 0.6], 'lag': [8, 20, 25, 30, 60, 70],
                'bias': [-0.4, 0.8, 1.2, -1.0, 0.5, 0.9], 'noise': [0.4, 0.8, 1.0, 1.2, 1.5, 0.5]}
TEMP_BOTTOM = {'gain': [1.0, 0.55, 2.2, 0.9, 0.3, 1.5], 'lag': [3, 25, 10, 100, 40, 35],
               'bias': [0.3, -1.5, 2.0, -0.8, 1.5, -3.0], 'noise': [0.5, 1.0, 1.5, 3.0, 1.0, 1.0]}


def _model(obs, gain, lag, bias, noise):
    anom = np.roll(obs - obs.mean(), lag)
    return obs.mean() + gain * anom + bias + rng.normal(0, noise, obs.size)


# temperature time series, shape (n,) [degC]
TempBottomObs = [12 + 8 * season + rng.normal(0, 1.0, n_days) for _ in stnslist]
TempSurfaceObs = [9 + (8 - 0.1 * z) * np.roll(season, z) + rng.normal(0, 0.7, n_days)
                  for z in depth]
TempBottomMod = [_model(o, *p) for o, p in zip(TempBottomObs, zip(*TEMP_BOTTOM.values()))]
TempSurfaceMod = [_model(o, *p) for o, p in zip(TempSurfaceObs, zip(*TEMP_SURFACE.values()))]

CURRENT_SURFACE = {'gain': [1.15, 1.2, 1.1, 1.25, 1.18, 1.12], 'lag': [10, 14, 8, 12, 16, 11],
                   'noise': [1.0, 1.1, 0.9, 1.2, 1.0, 1.1]}
CurrentSurfaceObs, CurrentSurfaceMod = [], []
for gain, lag, noise in zip(*CURRENT_SURFACE.values()):
    tide = 2 * np.pi * t / 14.77  # spring-neap cycle, daily samples
    obs = np.column_stack([0.3 * np.cos(tide), 0.1 * np.sin(tide)])
    obs = obs + rng.normal(0, 0.05, (n_days, 2))
    turn = np.deg2rad(lag)  # model current direction rotated by the lag in degrees
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    mod = gain * obs @ rot.T + [0.03, -0.02] + rng.normal(0, 0.06 * noise, (n_days, 2))
    CurrentSurfaceObs.append(obs)
    CurrentSurfaceMod.append(mod)
