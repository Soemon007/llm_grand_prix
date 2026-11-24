"""LLM Grand Prix: the race track and the car physics.

World units are pixels on a 1400 x 340 canvas. One sim step = 1/60 s.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

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
_TREE = cKDTree(CENTER)
RIGHT_EDGE = CENTER - NORMAL * TRACK_WIDTH / 2


def nearest_index(pos, hint=None):
    if hint is None:
        idx = np.arange(N)
    else:
        idx = (hint + np.arange(-40, 120)) % N
    d = np.sum((CENTER[idx] - pos) ** 2, axis=1)
    return int(idx[np.argmin(d)])


class Car:
    """A top-down car with momentum. Controls: steer in [-1, 1], throttle in [-1, 1].

    - Steering turns the front wheels gradually, and you get less steering lock the faster you go.
    - The car has momentum: the tyres can only bend its path so hard (GRIP). Turn too
      sharply at speed and it slides wide instead of turning.
    - Rolling resistance and air drag slow it down when you lift off the throttle.
    """

    WHEELBASE = 14.0
    MAX_STEER = np.radians(40)      # full lock when standing still...
    STEER_FALLOFF = 250.0           # ...shrinks with speed: lock / (1 + (speed / 250)^2)
    STEER_RATE = np.radians(240)    # how fast the wheels turn (deg/s)
    MAX_SPEED = 420.0               # px/s
    ENGINE = 240.0                  # px/s^2 at full throttle
    BRAKE = 520.0                   # px/s^2 at full brake
    ROLLING = 15.0                  # px/s^2 rolling resistance
    DRAG = 0.0012                   # air drag: DRAG * speed^2 px/s^2
    GRIP = 1700.0                   # max sideways acceleration from the tyres (px/s^2)
    SLIDE_SCRUB = 1.5               # speed lost per second per radian of sliding
    HALF_WIDTH = 4.5                # car body is 16 x 9 px
    IMPACT_KEEP = 0.5               # hitting a wall keeps only half your speed...
    SCRAPE = 600.0                  # ...and scraping along it slows you a further 600 px/s^2

    def __init__(self, start_index=0, lateral=0.0, back=0.0):
        i = (start_index - int(back / (LAP_LENGTH / N))) % N
        self.idx = i
        self.progress = -back           # px travelled along the lap
        self.hit_wall = False           # touching a wall this step
        self.wall_hits = 0              # number of separate impacts so far
        self.impact_pos = None          # where the latest new impact happened (for sparks)
        self.place(i, lateral)

    def place(self, i, lateral=0.0):
        """Put the car at rest on track point i, pointing along the track."""
        self.pos = CENTER[i] + NORMAL[i] * lateral
        self.heading = float(np.arctan2(TANGENT[i, 1], TANGENT[i, 0]))
        self.vel = np.zeros(2)
        self.speed = 0.0
        self.steer_angle = 0.0
        self.hit_wall = False

    @property
    def slip(self):
        """Angle between where the car points and where it is actually going (rad)."""
        if self.speed < 1:
            return 0.0
        move = np.arctan2(self.vel[1], self.vel[0])
        return float((move - self.heading + np.pi) % (2 * np.pi) - np.pi)

    def step(self, steer, throttle):
        throttle = float(np.clip(throttle, -1, 1))
        lock = self.MAX_STEER / (1 + (self.speed / self.STEER_FALLOFF) ** 2)
        target = float(np.clip(steer, -1, 1)) * lock
        self.steer_angle += float(np.clip(target - self.steer_angle, -self.STEER_RATE * DT, self.STEER_RATE * DT))

        fwd = np.array([np.cos(self.heading), np.sin(self.heading)])
        v_fwd = float(np.dot(self.vel, fwd))

        # engine pushes along the car; brakes, rolling resistance and drag slow the car down
        if throttle > 0:
            self.vel = self.vel + fwd * throttle * self.ENGINE * DT
        speed = float(np.linalg.norm(self.vel))
        slow = self.ROLLING + self.DRAG * speed ** 2 + (-throttle * self.BRAKE if throttle < 0 else 0.0)
        slow += self.SLIDE_SCRUB * abs(self.slip) * speed
        if speed > 0:
            new_speed = min(max(0.0, speed - slow * DT), self.MAX_SPEED)
            self.vel = self.vel * (new_speed / speed)

        # the car rotates according to its steering (only while rolling forwards)
        yaw = max(v_fwd, 0.0) * np.tan(self.steer_angle) / self.WHEELBASE
        self.heading += yaw * DT

        # the tyres bend the car's path towards where it points, but only so hard
        speed = float(np.linalg.norm(self.vel))
        if speed > 1:
            move = np.arctan2(self.vel[1], self.vel[0])
            gap = (self.heading - move + np.pi) % (2 * np.pi) - np.pi
            turn = float(np.clip(gap, -self.GRIP / speed * DT, self.GRIP / speed * DT))
            move += turn
            self.vel = speed * np.array([np.cos(move), np.sin(move)])
        self.speed = speed

        self.pos = self.pos + self.vel * DT
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
        vn = float(np.dot(self.vel, NORMAL[i]))
        if vn * side > 0:                                                # moving into the wall: lose that part
            self.vel = self.vel - NORMAL[i] * vn
            if np.linalg.norm(self.vel) > 1e-6:                          # glance off: the car now points along the wall
                self.heading = float(np.arctan2(self.vel[1], self.vel[0]))
        if not self.hit_wall:                                            # a fresh impact
            self.vel = self.vel * self.IMPACT_KEEP
            self.wall_hits += 1
            self.impact_pos = CENTER[i] + NORMAL[i] * side * TRACK_WIDTH / 2
        else:                                                            # scraping along it
            sp = float(np.linalg.norm(self.vel))
            if sp > 0:
                self.vel = self.vel * (max(0.0, sp - self.SCRAPE * DT) / sp)
        self.speed = float(np.linalg.norm(self.vel))
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
        rays = self._rays(n_rays)
        return {
            "speed": self.speed,                  # px/s, 0 .. 420
            "rays": rays,                         # distance to the barrier, -90° .. +90°, max 200
            "waypoints": waypoints,               # next centre-line points, car frame (x fwd, y right)
            "lateral_offset": lateral,            # px from centre line (+ = right of centre)
            "heading_error": heading_err,         # rad, car heading minus track direction
            "curvature_ahead": CURVATURE[wp_idx], # 1/px at each waypoint
            "hit_wall": self.hit_wall,            # touching a barrier right now
        }

    def _rays(self, n_rays, max_dist=200.0, step=4.0):
        """Distance to the barrier along n_rays rays from -90° to +90° (all rays at once)."""
        ang = self.heading + np.linspace(-np.pi / 2, np.pi / 2, n_rays)
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        k = np.arange(step, max_dist, step)
        pts = self.pos + dirs[:, None, :] * k[None, :, None]
        _, idx = _TREE.query(pts.reshape(-1, 2))
        idx = idx.reshape(n_rays, -1)
        lateral = np.abs(np.einsum("rkd,rkd->rk", pts - CENTER[idx], NORMAL[idx]))
        outside = lateral > TRACK_WIDTH / 2
        first = outside.argmax(axis=1)
        return np.where(outside.any(axis=1), k[first], max_dist)

    def _ray(self, ang, max_dist=200.0, step=4.0):
        d = np.array([np.cos(ang), np.sin(ang)])
        i = self.idx
        for k in np.arange(step, max_dist, step):
            p = self.pos + d * k
            i = nearest_index(p, i - 60)
            if abs(np.dot(p - CENTER[i], NORMAL[i])) > TRACK_WIDTH / 2:
                return k
        return max_dist
