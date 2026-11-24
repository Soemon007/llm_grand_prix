"""Week 1, walkthrough 1.4: plot the laps you drove with play.py."""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("my_laps.csv")
print(df)
print("Best lap:", df["lap_time"].min())

plt.plot(df["lap"], df["lap_time"], marker="o")
plt.axhline(df["lap_time"].min(), linestyle="--", label="Best lap = your benchmark")
plt.xlabel("Lap")
plt.ylabel("Lap time (s)")
plt.legend()
plt.show()
