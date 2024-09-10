import skfuzzy as fuzzy
import numpy as np
import matplotlib.pyplot as plt
x=np.arange(41)
mfx=fuzzy.trapmf(x,[0,5,15,20])
mfy=fuzzy.trimf(x,[15,20,25])
mfa=fuzzy.trapmf(x,[20,25,30,40])
plt.figure(figsize=(10, 8))

plt.plot(x, mfx, label='Temperatura 1')
plt.plot(x, mfy, label='Temperatura 2')
plt.plot(x, mfa, label='Temperatura 4')


plt.title('Funciones de Membresía')
plt.xlabel('x')
plt.ylabel('Membresía')
plt.legend()
plt.grid(True)
plt.show()