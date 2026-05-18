import matplotlib.pyplot as plt 
import pandas as pd 
df=pd.read_csv("data.csv")
df.columns=[2,4,8,16]
df.plot(kind="box")
plt.title("Comparacion: ruleta vs torneo")
plt.show()

