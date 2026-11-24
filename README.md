# 🏁 LLM Grand Prix

Train an AI racer from scratch, then race it against ghost cars driven by bots that ChatGPT, Claude and Gemini designed.

This repo grows week by week. Right now it has everything for **Week 1: Warm-up Lap**.

## Setup

You need **Python 3.10 or newer**.

```bash
git clone <this repo's URL>
cd llm_grand_prix
pip install -r requirements.txt
python check.py
```

`check.py` should print a version for every library and end with "All good!".

> **Install trouble?**
> - Windows: if `pip` isn't found, use `py -m pip install -r requirements.txt`.
> - Mac: if `pip` isn't found, use `python3 -m pip install -r requirements.txt`.
> - Mac (Homebrew) "externally-managed-environment": make a virtual environment first with `python3 -m venv .venv` and `source .venv/bin/activate`.
> - pygame won't install: run `pip install pygame-ce` instead. It works the same.

## Week 1: drive it yourself 🏎️

```bash
python play.py
```

| Key | Action |
| --- | --- |
| ↑ | Accelerate |
| ↓ | Brake |
| ← → | Steer |
| SPACE | Put the car back on the track (+2 s penalty) |
| R | Restart |
| ESC | Quit |

Drive **5 laps**. The track has solid walls, and hitting them costs speed. Your lap times are saved to `my_laps.csv` after every lap.

Then plot your laps:

```bash
python plot_laps.py
```

Your best lap is your first benchmark. By Week 5, your AI should beat you.

## Files

| File | What it is |
| --- | --- |
| `play.py` | The game: drive the car with your keyboard |
| `track.py` | The race track and the car physics |
| `plot_laps.py` | Plots the laps saved in `my_laps.csv` |
| `check.py` | Checks your setup |
| `requirements.txt` | The libraries to install |
