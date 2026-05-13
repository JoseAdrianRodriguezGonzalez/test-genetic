import matplotlib.pyplot as plt 
import pandas as pd 

df=pd.read_csv("data.csv")
df[["ruleta","torneo"]].plot(kind="box")
plt.title("Comparacion: ruleta vs torneo")
plt.show()

