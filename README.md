# Social-Network-Ads-Dataset-Machine-Learning

[BR]: Notebooks Jupyter com a análise do dataset, o processamento dos seus dados e as Inteligências Artificiais treinadas com o dataset, usando a técnica de Machine Learning. Usando as bibliotecas do Python: Numpy, Pandas, Matplotlib, Seaborn, ibmlearn, Scikit-learn, pickle e joblib.

[US]: Jupyter Notebooks with the analysis of the dataset, its data processing, and the Artificial Intelligence trained with this dataset, using Machine Learning's technique. Using the Python's librarys: Numpy, Pandas, Matplotlib, Seaborn, ibmlearn, Scikit-learn, pickle and joblib.

## **Análise dos modelos *rf3* e *dtree***

Escolhi fazer a análise dos modelos *rf3* e *dtree* pelo fato deles serem os modelos com maior e menor score, respectivamente.

Obs: infelizmente, quando gerei os modelos, eu só salvei os modelos de cada método com melhores scores por isso só apresento o arquivo do modelo *knn*, *gnb2*, *dtree3*, *rf3* e *v*. 

### **Pré-processamento**

Em ambos foi realizado um pré-processamento dos dados, sendo a retirada da coluna 'User ID', e a transformação de dados categóricos em dados numéricos na coluna 'Gender', pelo `LabelEncoder`.

Porém, só no modelo *rf3* foram feitas mais mudanças no dataset. Inicialmente foi feita uma padronização dos dados "não target" de treinamento (X), utilizando o `StandartScaler()`. Depois os dados foram divididos em dados de treinamento e de teste, pelo `train_test_split()`. Em seguida, devido a existência de desbalanceamento, foi utilizada a técnica de Undersampling, para balancear os dados pertencentes pelas duas classes, utilizando `RandomUnderSampler`.

### **Treinamento dos modelos**

O modelo com mais score utiliza o método de RandomForest de Machine Learning de Classificação Supervisionada, esse método faz uso de várias DecisionTree's para encontrar a com melhores resultados. Para começar, instanciei a classe RandomForestClassifier():

```
rf3 = RandomForestClassifier()
```

Depois utilizei o método fit(), para treinar o modelo, utilizando os dados de treinamento:

```
rf3.fit(X_res3, y_res3)
```


E por fim verifiquei o score do modelo com o método score(), utilizando os dados de teste:

```
rf3.score(X_test3, y_test3)
```

Recebendo a resposta:


> 0.9659090909090909

O modelo com menor score utiliza o método de DecisionTree de Machine Learning de Classificação Supervisionada, esse método cria uma série de condições, que, quando acabam, devolvem uma resposta, ou seja a classe daquela amostra. Para começar, instanciei a classe DecisionTreeClassifier():

```
dtree = DecisionTreeClassifier()
```

Depois utilizei o método fit(), para treinar o modelo, utilizando os dados de treinamento:

```
dtree.fit(X_train, y_train)
```


E por fim verifiquei o score do modelo com o método score(), utilizando os dados de teste:

```
dtree.score(X_test, y_test)
```

Recebendo a resposta:


> 0.8409090909090909

### **Validação e métricas**

O *dtree* apesar de ter o menor score, mesmo que esse já seja alto, apresenta outras métricas quando utilizamos `classification_report()`.

> **Precisão** - *Classe '0'* = 0.98 || *Classe '1'* = 0.94

> **Recall** - *Classe '0'* = 0.96 || *Classe '1'* = 0.97

> **F1-score** - *Classe '0'* = 0.97 || *Classe '1'* = 0.96

E essas são ainda melhores, já que na **Precisão**, que é calculada pela divisão da somatória dos Verdadeiros Positivos pela somatória da Condição positiva prevista, tem uma média de 0.96. A **recall**, que é calculada pela divisão da somatória dos Verdadeiros Positivos pela somatória da Condição Prevista, tendo uma média de 0.965. E a **f1-score**, calculada pela multiplicação de 2 pela divisão da multiplicação da precisão com a recall pela soma da precisão com a recall, que também tem uma média de 0.965.

O *rf3* tem o maior score, comparando com o menor existe uma diferença de 0.12, quanto as outras métricas apresentadas pela função `classification_report()`, temos:

> **Precisão** - *Classe '0'* = 1.0 || *Classe '1'* = 0.92

> **Recall** - *Classe '0'* = 0.94 || *Classe '1'* = 1.0

> **F1-score** - *Classe '0'* = 0.97 || *Classe '1'* = 0.96

E essas são ainda melhores, já que na **Precisão**, tem uma média de 0.96, mas na classe '0' tem precisão de 1. A **recall**, tem uma média de 0.97, mas na classe '1' tem recall de 1. E a **f1-score**, que tem uma média de 0.965 (igual a *dtree*).

Sendo assim, pelo fato da *rf3* apresentar um maior score (acurácia), e ser um pouco melhor para a classe '0', é recomendável a utilização dela para uso geral ou para foco na classe '0'. Porém, se tiver um foco na classe '1', ambas apresentam resultados muito semelhantes, assim sendo de preferência pessoal.
