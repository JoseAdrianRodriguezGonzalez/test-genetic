import skfuzzy as fuzzy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#definicion del conjunto de datos
data = {
    'X': [0.842105263, 0.571052632, 0.560526316, 0.878947368, 0.581578947, 
          0.834210526, 0.884210526, 0.797368421, 0.1, 0.744736842],
    'Y': [0.191699605, 0.205533597, 0.320158103, 0.239130435, 0.365612648, 
          0.201581028, 0.223320158, 0.278656126, 0.177865613, 0.120553336],
    'Target': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
}
df=pd.DataFrame(data);
#Funcines de membreeía

def plots(x,mfx,mfy,mfa):
    plt.figure(figsize=(10, 8))
    plt.plot(x, mfx, label='Temperatura 1')
    plt.plot(x, mfy, label='Temperatura 2')
    plt.plot(x, mfa, label='Temperatura 3')
    plt.title('Funciones de Membresía')
    plt.xlabel('x')
    plt.ylabel('Membresía')
    plt.legend()
    plt.grid(True)
    plt.show()
def generatingFunctions():
    x = np.arange(0, 1.31, 0.01)
#Definicion de la clases difusas
    values=[]
    mfx=fuzzy.trimf(x,[0,0.4,0.7])
    values.append(mfx)
    mfy=fuzzy.trimf(x,[0.4,0.7,1])
    values.append(mfy)
    mfa=fuzzy.trimf(x,[0.7,1,1.3])
    values.append(mfa)
    return x,values
def membershipFunc(x,value,datas):
    members=[]
    low=fuzzy.interp_membership(x,value[0],datas);
    members.append(low)
    medium=fuzzy.interp_membership(x,value[1],datas);
    members.append(medium)
    high=fuzzy.interp_membership(x,value[2],datas);     
    members.append(high)
    return members
x,val=generatingFunctions()
#plots(x,val[0],val[1],val[2])
F_x=membershipFunc(x,val,df.X)
F_y=membershipFunc(x,val,df.Y)

print(F_y)