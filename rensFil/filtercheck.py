import numpy as np
from numpy import pi, exp
import matplotlib.pyplot as plt


omega = np.linspace(0,pi, 1001)
z = exp(1j * omega)
z = 1/z
H = (0.281 - 0.281 * (z ** 2)) / (1 - 1.3981 * z + 0.438 * (z ** 2))
absH = np.abs(H)
Hdb = 20 * np.log10(absH / np.max(absH))
freqs = omega / pi * 8000
plt.plot(freqs, Hdb)
plt.ylim(-80,0)
plt.show()