"""Drive the race car yourself!

    python play.py

Controls
    ↑          accelerate
    ↓          brake
    ← / →      steer
    SPACE      put the car back on the track (+2 s penalty)
    R          restart
    ESC        quit

Finish 5 laps. Your lap times are saved to my_laps.csv after every lap.
"""
import csv

import numpy as np
import pygame

import track as T

LAPS = 5
RESET_PENALTY = 2.0          # seconds added when you press SPACE
OUTPUT = "my_laps.csv"

VIEW_X0 = 440                # the track lives in x = 440..1400 of the world
HUD_H = 64                   # height of the info bar at the top

GRASS = (62, 118, 56)
GRASS_STRIPE = (70, 128, 62)
ASPHALT = (58, 60, 66)
WALL = (204, 204, 198)
WALL_DARK = (44, 44, 50)
RED = (232, 54, 66)
WHITE = (245, 245, 245)
CAR = (232, 54, 66)
HUD_BG = (11, 14, 24)
TEXT = (240, 242, 248)
MUTED = (122, 132, 156)
GOLD = (255, 196, 46)


def fmt(t):
    return "--:--.--" if t is None else f"{int(t // 60)}:{t % 60:05.2f}"


class Game:
    """All the game logic. Drawing is kept separate so it is easy to read."""

    def __init__(self):
        self.restart()

    def restart(self):
        self.car = T.Car(start_index=0)
        self.steer = 0.0
        self.clock = 0.0             # race time, starts at the green light
        self.countdown = 3.0         # seconds of red lights before the start
        self.lap_start = 0.0
        self.laps = []               # finished lap times
        self.penalty = 0.0           # penalty collected during the current lap
        self.sparks = []             # little particles when you hit a wall
        self.skids = []              # tyre marks left when the car slides
        self.finished = False

    def update(self, steer_input, throttle_input, reset=False):
        """Advance the game by one frame (1/60 s)."""
        if self.finished:
            return
        if self.countdown > 0:
            self.countdown -= T.DT
            return

        self.steer = steer_input              # the car itself turns its wheels gradually

        if reset:
            self.reset_car()

        self.car.step(self.steer, throttle_input)
        self.clock += T.DT
        if self.car.impact_pos is not None:
            self.add_sparks(self.car.impact_pos)
        if abs(self.car.slip) > 0.12 and self.car.speed > 60:
            self.skids = (self.skids + [self.car.pos.copy()])[-400:]

        # a lap is done each time the car's distance passes another full lap
        if self.car.progress >= (len(self.laps) + 1) * T.LAP_LENGTH:
            lap_time = self.clock - self.lap_start + self.penalty
            self.laps.append(lap_time)
            self.lap_start = self.clock
            self.penalty = 0.0
            save_laps(self.laps)
            if len(self.laps) == LAPS:
                self.finished = True

        self.sparks = [(p, v, life - T.DT) for p, v, life in self.sparks if life > T.DT]
        self.sparks = [(p + v * T.DT, v, life) for p, v, life in self.sparks]

    def reset_car(self):
        """Put the car back on the centre line, pointing the right way."""
        self.car.place(self.car.idx)
        self.steer = 0.0
        self.penalty += RESET_PENALTY

    def add_sparks(self, pos):
        rng = np.random.default_rng()
        for _ in range(12):
            ang = rng.uniform(0, 2 * np.pi)
            v = rng.uniform(40, 140) * np.array([np.cos(ang), np.sin(ang)])
            self.sparks.append((np.array(pos, dtype=float), v, rng.uniform(0.15, 0.35)))

    @property
    def current_lap_time(self):
        return self.clock - self.lap_start + self.penalty


def save_laps(laps):
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lap", "lap_time"])
        for n, t in enumerate(laps, start=1):
            writer.writerow([n, round(t, 3)])


# ---------------------------------------------------------------- drawing
class Renderer:
    def __init__(self, screen, scale):
        self.screen = screen
        self.scale = scale
        self.font_big = pygame.font.SysFont(None, int(44 * min(scale, 1.2)))
        self.font = pygame.font.SysFont(None, int(28 * min(scale, 1.2)))
        self.track = self.draw_track()

    def px(self, p):
        """World coordinates -> screen pixels."""
        return (int((p[0] - VIEW_X0) * self.scale), int(p[1] * self.scale) + HUD_H)

    def pts(self, arr):
        return [self.px(p) for p in arr]

    def draw_track(self):
        s = self.scale
        w = int((T.W - VIEW_X0) * s)
        surf = pygame.Surface((w, int(T.H * s) + HUD_H))
        surf.fill(GRASS)
        for x0 in range(-surf.get_height(), w, int(70 * s)):
            h = surf.get_height()
            pygame.draw.polygon(surf, GRASS_STRIPE, [(x0, HUD_H), (x0 + 35 * s, HUD_H),
                                                      (x0 + 35 * s + h, h + HUD_H), (x0 + h, h + HUD_H)])
        hw = T.TRACK_WIDTH / 2
        loop = np.vstack([T.CENTER, T.CENTER[:1]])
        for side in (1, -1):
            outer = loop + np.vstack([T.NORMAL, T.NORMAL[:1]]) * side * (hw + 2.5)
            pygame.draw.lines(surf, WALL_DARK, True, self.pts(outer), max(2, int(5 * s)))
        pygame.draw.polygon(surf, ASPHALT, self.pts(np.vstack([T.LEFT_EDGE, T.RIGHT_EDGE[::-1]])))
        pygame.draw.lines(surf, ASPHALT, True, self.pts(T.CENTER), int(T.TRACK_WIDTH * s))
        for p in self.pts(T.CENTER[::4]):           # round off the fat line's corners
            pygame.draw.circle(surf, ASPHALT, p, int(hw * s))
        for side in (1, -1):
            edge = T.CENTER + T.NORMAL * side * (hw + 0.8)
            pygame.draw.lines(surf, WALL, True, self.pts(edge), max(1, int(2 * s)))
        for i in range(0, T.N, 6):                  # kerbs on tight corners
            k = T.CURVATURE[i]
            if abs(k) < 0.018:
                continue
            side = -np.sign(k)
            a = T.CENTER[i] + T.NORMAL[i] * side * (hw - 2)
            b = T.CENTER[(i + 6) % T.N] + T.NORMAL[(i + 6) % T.N] * side * (hw - 2)
            pygame.draw.line(surf, RED if (i // 6) % 2 else WHITE, self.px(a), self.px(b), max(2, int(3 * s)))
        rows = 6                                     # chequered start/finish line
        for r in range(rows):
            for c in range(2):
                lat = -hw + T.TRACK_WIDTH * (r + 0.5) / rows
                p = T.CENTER[0] + T.NORMAL[0] * lat + T.TANGENT[0] * (c * 3 - 1.5)
                half = T.TRACK_WIDTH / rows / 2
                corners = [p + T.NORMAL[0] * half * sx + T.TANGENT[0] * 1.5 * sy
                           for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                pygame.draw.polygon(surf, WHITE if (r + c) % 2 else (16, 16, 16), self.pts(corners))
        pygame.draw.rect(surf, HUD_BG, (0, 0, w, HUD_H))
        return surf

    def draw_car(self, car):
        c, s = np.cos(car.heading), np.sin(car.heading)
        fwd, side = np.array([c, s]), np.array([-s, c])
        body = [car.pos + fwd * 8 * a + side * 4.5 * b for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        pygame.draw.polygon(self.screen, CAR, self.pts(body))
        cockpit = [car.pos + fwd * (2 + 3 * a) + side * 2.5 * b for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        pygame.draw.polygon(self.screen, (20, 24, 34), self.pts(cockpit))

    def text(self, msg, pos, font=None, colour=TEXT, center=False):
        img = (font or self.font).render(msg, True, colour)
        rect = img.get_rect(center=pos) if center else img.get_rect(topleft=pos)
        self.screen.blit(img, rect)

    def draw(self, game):
        self.screen.blit(self.track, (0, 0))
        for p in game.skids:
            pygame.draw.circle(self.screen, (36, 37, 41), self.px(p), max(1, int(1.6 * self.scale)))
        self.draw_car(game.car)
        for p, _, life in game.sparks:
            pygame.draw.circle(self.screen, (255, 214, 90), self.px(p), max(1, int(2 * self.scale * life / 0.35)))

        # info bar
        w = self.screen.get_width()
        lap_no = min(len(game.laps) + 1, LAPS)
        best = min(game.laps) if game.laps else None
        last = game.laps[-1] if game.laps else None
        self.text(f"LAP {lap_no}/{LAPS}", (16, 18), self.font_big)
        self.text(f"TIME  {fmt(game.current_lap_time if game.countdown <= 0 and not game.finished else None)}",
                  (int(w * 0.22), 22))
        self.text(f"LAST  {fmt(last)}", (int(w * 0.44), 22), colour=MUTED)
        self.text(f"BEST  {fmt(best)}", (int(w * 0.64), 22), colour=GOLD)
        self.text(f"{game.car.speed:3.0f} px/s", (int(w * 0.85), 22), colour=MUTED)

        mid = (w // 2, HUD_H + (self.screen.get_height() - HUD_H) // 2)
        if game.countdown > 0:                       # start lights
            lit = 5 - int(np.ceil(game.countdown / 0.6))
            for n in range(5):
                colour = RED if n < max(lit, 1) else (44, 48, 60)
                pygame.draw.circle(self.screen, colour, (mid[0] - 80 + 40 * n, mid[1]), 14)
            self.text("Get ready...", (mid[0], mid[1] + 40), center=True)
        elif game.clock < 1.0:
            for n in range(5):
                pygame.draw.circle(self.screen, (60, 220, 110), (mid[0] - 80 + 40 * n, mid[1]), 14)
        if game.finished:
            box = pygame.Surface((520, 190), pygame.SRCALPHA)
            box.fill(HUD_BG + (235,))
            self.screen.blit(box, box.get_rect(center=mid))
            self.text("CHEQUERED FLAG!", (mid[0], mid[1] - 55), self.font_big, GOLD, center=True)
            self.text(f"Best lap: {fmt(min(game.laps))}", (mid[0], mid[1] - 10), self.font_big, center=True)
            self.text(f"Saved {LAPS} laps to {OUTPUT}", (mid[0], mid[1] + 30), colour=MUTED, center=True)
            self.text("R = race again     ESC = quit", (mid[0], mid[1] + 62), colour=MUTED, center=True)


def main():
    pygame.init()
    info = pygame.display.Info()
    scale = min(1.3, (info.current_w - 60) / (T.W - VIEW_X0), (info.current_h - HUD_H - 120) / T.H)
    size = (int((T.W - VIEW_X0) * scale), int(T.H * scale) + HUD_H)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("LLM Grand Prix: drive it yourself")
    renderer = Renderer(screen, scale)
    game = Game()
    clock = pygame.time.Clock()

    running = True
    while running:
        reset = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.restart()
                elif event.key == pygame.K_SPACE:
                    reset = True
        keys = pygame.key.get_pressed()
        steer = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        throttle = keys[pygame.K_UP] - keys[pygame.K_DOWN]
        game.update(steer, throttle, reset)
        renderer.draw(game)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    if game.laps:
        print(f"{len(game.laps)} lap(s) saved to {OUTPUT}. Best lap: {fmt(min(game.laps))}")


if __name__ == "__main__":
    main()
