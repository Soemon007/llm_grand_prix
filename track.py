"""LLM Grand Prix: the race track and the car physics.

World units are pixels on a 1400 x 340 canvas. One sim step = 1/60 s.
"""
import numpy as np
from scipy.interpolate import CubicSpline

W, H = 1400, 340
TRACK_WIDTH = 32.0
DT = 1 / 60

# Control points of the centre line, driven clockwise on screen.
CONTROL = np.array([
    (600, 282), (850, 285), (1100, 282), (1250, 276), (1330, 242),
    (1352, 182), (1322, 124), (1252, 102), (1188, 122), (1146, 158),
    (1086, 172), (1036, 146), (1012, 100), (962, 68), (862, 60),
    (762, 66), (694, 90), (670, 134), (710, 166), (750, 188), (756, 212),
    (728, 234), (660, 240), (566, 222), (504, 230), (482, 258), (520, 284),
], dtype=float)


def _build_centerline(n=2400):
    pts = np.vstack([CONTROL, CONTROL[:1]])
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    cs = CubicSpline(d, pts, bc_type="periodic")
    dense = cs(np.linspace(0, d[-1], 20000, endpoint=False))
    seg = np.linalg.norm(np.diff(np.vstack([dense, dense[:1]]), axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    total = s[-1]
    target = np.linspace(0, total, n, endpoint=False)
    x = np.interp(target, s, np.r_[dense[:, 0], dense[0, 0]])
    y = np.interp(target, s, np.r_[dense[:, 1], dense[0, 1]])
    return np.stack([x, y], axis=1), total


CENTER, LAP_LENGTH = _build_centerline()
from scipy.ndimage import gaussian_filter1d
CENTER = gaussian_filter1d(CENTER, 20, axis=0, mode="wrap")   # round off kinks
LAP_LENGTH = float(np.sum(np.linalg.norm(np.diff(np.vstack([CENTER, CENTER[:1]]), axis=0), axis=1)))
N = len(CENTER)
_tan = np.roll(CENTER, -1, axis=0) - np.roll(CENTER, 1, axis=0)
TANGENT = _tan / np.linalg.norm(_tan, axis=1, keepdims=True)
NORMAL = np.stack([-TANGENT[:, 1], TANGENT[:, 0]], axis=1)
_ang = np.unwrap(np.arctan2(TANGENT[:, 1], TANGENT[:, 0]))
CURVATURE = np.gradient(_ang) / (LAP_LENGTH / N)
LEFT_EDGE = CENTER + NORMAL * TRACK_WIDTH / 2
RIGHT_EDGE = CENTER - NORMAL * TRACK_WIDTH / 2


def nearest_index(pos, hint=None):
    if hint is None:
        idx = np.arange(N)
    else:
        idx = (hint + np.arange(-40, 120)) % N
    d = np.sum((CENTER[idx] - pos) ** 2, axis=1)
    return int(idx[np.argmin(d)])


class Car:
    """Kinematic car. Controls: steer in [-1, 1], throttle in [-1, 1]."""

    WHEELBASE = 14.0
    MAX_STEER = np.radians(42)
    MAX_SPEED = 420.0      # px/s
    HALF_WIDTH = 4.5       # car body is 16 x 9 px
    IMPACT_KEEP = 0.5      # hitting a barrier keeps only half your speed...
    SCRAPE = 600.0         # ...and scraping along it slows you a further 600 px/s^2
    ACCEL = 260.0
    BRAKE = 520.0
    GRIP = 2600.0          # max lateral acceleration before sliding wide

    def __init__(self, start_index=0, lateral=0.0, back=0.0):
        i = (start_index - int(back / (LAP_LENGTH / N))) % N
        self.pos = CENTER[i] + NORMAL[i] * lateral
        self.heading = float(np.arctan2(TANGENT[i, 1], TANGENT[i, 0]))
        self.speed = 0.0
        self.idx = i
        self.progress = -back       # px travelled along the lap
        self.hit_wall = False      # touching a barrier this step
        self.wall_hits = 0         # number of separate impacts so far
        self.impact_pos = None     # where the latest new impact happened (for sparks)

    def step(self, steer, throttle):
        steer = float(np.clip(steer, -1, 1)) * self.MAX_STEER
        throttle = float(np.clip(throttle, -1, 1))
        if throttle >= 0:
            self.speed += throttle * self.ACCEL * DT
        else:
            self.speed += throttle * self.BRAKE * DT
        self.speed = float(np.clip(self.speed, 0, self.MAX_SPEED))

        yaw = self.speed * np.tan(steer) / self.WHEELBASE
        if self.speed > 1:
            max_yaw = self.GRIP / self.speed       # tyres can't turn faster than this
            yaw = float(np.clip(yaw, -max_yaw, max_yaw))
        self.heading += yaw * DT
        self.pos = self.pos + self.speed * DT * np.array([np.cos(self.heading), np.sin(self.heading)])

        new_idx = nearest_index(self.pos, self.idx)
        delta = (new_idx - self.idx + N // 2) % N - N // 2
        self.progress += delta * LAP_LENGTH / N
        self.idx = new_idx
        self._barriers(new_idx)

    def _barriers(self, i):
        """Solid walls on both track edges: no car can ever leave the track."""
        lateral = float(np.dot(self.pos - CENTER[i], NORMAL[i]))
        limit = TRACK_WIDTH / 2 - self.HALF_WIDTH
        self.impact_pos = None
        if abs(lateral) <= limit:
            self.hit_wall = False
            return
        side = np.sign(lateral)
        self.pos = self.pos - NORMAL[i] * (lateral - side * limit)      # pushed back off the wall
        v = self.speed * np.array([np.cos(self.heading), np.sin(self.heading)])
        vn = float(np.dot(v, NORMAL[i]))
        if vn * side > 0:                                                # moving into the wall: lose that part
            v = v - NORMAL[i] * vn
            if np.linalg.norm(v) > 1e-6:
                self.heading = float(np.arctan2(v[1], v[0]))
            self.speed = float(np.linalg.norm(v))
        if not self.hit_wall:                                            # a fresh impact
            self.speed *= self.IMPACT_KEEP
            self.wall_hits += 1
            self.impact_pos = CENTER[i] + NORMAL[i] * side * TRACK_WIDTH / 2
        else:                                                            # scraping along it
            self.speed = max(0.0, self.speed - self.SCRAPE * DT)
        self.hit_wall = True

    def observe(self, n_rays=7, n_waypoints=10, spacing=40.0):
        """What a driver sees. Everything is relative to the car."""
        c, s = np.cos(self.heading), np.sin(self.heading)
        to_car = np.array([[c, s], [-s, c]])       # world -> car frame (x forward, y right)
        step = int(spacing / (LAP_LENGTH / N))
        wp_idx = (self.idx + step * np.arange(1, n_waypoints + 1)) % N
        waypoints = (CENTER[wp_idx] - self.pos) @ to_car.T
        lateral = float(np.dot(self.pos - CENTER[self.idx], NORMAL[self.idx]))
        track_ang = np.arctan2(TANGENT[self.idx, 1], TANGENT[self.idx, 0])
        heading_err = float((self.heading - track_ang + np.pi) % (2 * np.pi) - np.pi)
        angles = np.linspace(-np.pi / 2, np.pi / 2, n_rays)
        rays = np.array([self._ray(self.heading + a) for a in angles])
        return {
            "speed": self.speed,                  # px/s, 0 .. 420
            "rays": rays,                         # distance to the barrier, -90° .. +90°, max 200
            "waypoints": waypoints,               # next centre-line points, car frame (x fwd, y right)
            "lateral_offset": lateral,            # px from centre line (+ = right of centre)
            "heading_error": heading_err,         # rad, car heading minus track direction
            "curvature_ahead": CURVATURE[wp_idx], # 1/px at each waypoint
            "hit_wall": self.hit_wall,            # touching a barrier right now
        }

    def _ray(self, ang, max_dist=200.0, step=4.0):
        d = np.array([np.cos(ang), np.sin(ang)])
        i = self.idx
        for k in np.arange(step, max_dist, step):
            p = self.pos + d * k
            i = nearest_index(p, i - 60)
            if abs(np.dot(p - CENTER[i], NORMAL[i])) > TRACK_WIDTH / 2:
                return k
        return max_dist
