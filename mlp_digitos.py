import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.model_selection 

numeros = sklearn.datasets.load_digits()
imagenes = numeros['images']  # Hay 1797 digitos representados en imagenes 8x8
n_imagenes = len(imagenes)
X = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
Y = numeros['target']

#Dividimos el training y test en 50-50
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.5)

#Estandarizamos los datos
scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Creamos un arreglo que me vaya guardando tanto F1 como loss
F1_Loss=np.zeros((4,20))
#Creamos un arreglo que me vaya guardando cada uno de los coeficientes de la red, lo utilizaremos después para la neurona deseable
Coeficientes=[]

#Debido a que nos interesa una sola capa de neuronas, pero variando el número de neuronas utilizamos un for
for nun in range(1,21):
    mlp = sklearn.neural_network.MLPClassifier(activation='logistic', 
                                           hidden_layer_sizes=(nun), 
                                           max_iter=200)
    mlp.fit(X_train, Y_train)
    F1_Loss[0,nun-1]= sklearn.metrics.f1_score(Y_test, mlp.predict(X_test), average='macro')
    F1_Loss[1,nun-1]= mlp.loss_
    Coeficientes.append(mlp.coefs_[0])


#Graficamos F1 y Loss
fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
axs[0].scatter(np.arange(1,21), F1_Loss[0])
axs[0].set_title('F1 score')
axs[0].set_ylim(0,1)
axs[1].scatter(np.arange(1,21), F1_Loss[1])
axs[1].set_title('Loss')
fig.savefig('loss_f1.png')
plt.show()
#Ahora vamos a utilizar derivadas para saber cuando se estabiliza el F1
#El método que usaremos es el de buscar la primera dereivada menor a el valor aceptable 0.04

deri=np.absolute(np.gradient(F1_Loss[0]))
plt.scatter(np.arange(1,21),np.absolute(np.gradient(F1_Loss[0])))
plt.show()
num_neu_opti=0
while(deri[num_neu_opti]>0.007):
    num_neu_opti+=1
print(num_neu_opti)
print(np.absolute(np.gradient(F1_Loss[0]).min()))

#Como tenemos el número de neuronas óptimas ahora las graficamos
scale = np.max(mlp.coefs_[0])
#Usamos los coeficientes del número de nuronas óptimas
cof= Coeficientes[num_neu_opti+1]

fig, axs = plt.subplots(1, num_neu_opti, figsize=(9, 3), sharey=True)
for ifi in range (num_neu_opti):
    fig.suptitle('Neuronas Óptimas')
    axs[ifi].imshow(cof[:,ifi].reshape(8,8),cmap=plt.cm.RdBu, 
                   vmin=-scale, vmax=scale)
    axs[ifi].set_title(ifi)
fig.savefig('neuronas.png')   
