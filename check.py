"""Week 1 setup check: run `python check.py`. Every line should print a version."""
import numpy, scipy, matplotlib, pandas, pygame, gymnasium
print("numpy", numpy.__version__, "| scipy", scipy.__version__)
print("matplotlib", matplotlib.__version__, "| pandas", pandas.__version__)
print("pygame", pygame.version.ver, "| gymnasium", gymnasium.__version__)
print("All good! You're ready to race.")
