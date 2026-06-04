import matplotlib.pyplot as plt 
import pandas as pd 
for n in range(1,4):
    print(f"funcion {n}")
    df=pd.read_csv(f"data_{n}.csv")
    mean=df["col"].mean()
    std=df["col"].std()
    print(f"mu={mean}")
    print(f"std={std}")
