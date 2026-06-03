import matplotlib.pyplot as plt 
import pandas as pd 
df=pd.read_csv("data.csv")
df.columns=["unico","doble","uniforme"]
df.plot(kind="box")
plt.title("Comparacion: torneo")
plt.show()

