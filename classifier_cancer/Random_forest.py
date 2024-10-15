import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df=pd.read_csv('./classifier_cancer/data.csv')
#print(df.head())
#Change the categorical values
diganosis=df['diagnosis']
diganosis=diganosis.astype('category')
df['diagnosis']=diganosis.cat.codes

df.drop(columns='id',axis=1)
plt.figure(figsize=(10,20))
sns.heatmap(df.corr(),annot=True, fmt=".2f", cmap='coolwarm')
plt.title('heat map')
plt.show()

"""
Concave points worst
area worst
perimeter worst
radius worst
concave points mean
concavity mean
perimeter mean 
radius mean
"""
selected_features = [
    'concave points_worst', 'area_worst', 'perimeter_worst', 'radius_worst',
    'concave points_mean', 'concavity_mean', 'perimeter_mean', 'radius_mean'
]

# Separar las características (X) y la variable objetivo (y)
X = df[selected_features]
y = df['diagnosis']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

rf = RandomForestClassifier(n_estimators=100, random_state=0)

# Entrenar el modelo
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Calcular la exactitud (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f'Exactitud: {accuracy:.4f}')

# Matriz de confusión
print(confusion_matrix(y_test, y_pred))

# Informe de clasificación
print(classification_report(y_test, y_pred))