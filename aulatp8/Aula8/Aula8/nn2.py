import sklearn as skl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

%matplotlib inline

device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("./sbsppdaa24/train_radiomics_hipocamp.csv")

# Drop unique identifier columns
df.drop(columns=["Mask", "ID", "Image"], inplace=True)

# Drop non-numeric columns except for 'Transition'
columns_to_drop = [col for col in df.columns if df[col].dtype == 'object' and col != 'Transition']
df.drop(columns=columns_to_drop, inplace=True)

# Apply MinMax scaling to columns
from sklearn.preprocessing import MinMaxScaler
float_cols = df.select_dtypes(include=['float', 'int']).columns
scaler = MinMaxScaler()
df[float_cols] = scaler.fit_transform(df[float_cols])

df.head()

from sklearn.preprocessing import LabelEncoder

cols = ['Transition']
le = LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])
df.head()

df = df.loc[:, df.nunique() > 1]

# Separando variáveis
df_X = df.drop(columns=['Transition'])
df_y = df['Transition']

t_X = pd.DataFrame(df_X)
filename = "train_nn2.csv"
t_X.to_csv(filename, index=False, encoding='utf-8')

t_y = pd.DataFrame(df_y)
filename = "res_nn2.csv"
t_y.to_csv(filename, index=False, encoding='utf-8')

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor

# Classe personalizada para carregar o dataset a partir de arquivos CSV
class CSVDataset(Dataset):
    def __init__(self, path):
        # Lê o arquivo CSV que contém as features (entradas)
        df_X = pd.read_csv("train_nn2.csv", header=0)  # Carrega o arquivo de treino (features)
        
        # Lê o arquivo CSV que contém os rótulos (saídas/targets)
        df_y = pd.read_csv("res_nn2.csv", header=0)  # Carrega o arquivo de resultado
        
        # Converte os dados de features (entradas) para um array numpy
        self.X = df_X.values
        
        # Converte os dados de rótulos para um array numpy e ajusta os valores (subtrai 1)
        self.y = df_y.values[:, 0] - 1  # Subtrai 1 para ajustar os índices dos rótulos (de 1 para 0)
        
        # Converte os dados de features para o tipo 'float64' (precisão para modelos de ML)
        self.X = self.X.astype('float32')
        
        # Converte os rótulos para tensores do PyTorch (com tipo 'long' para inteiros)
        self.y = torch.tensor(self.y, dtype=torch.long, device=device)
        
        # Imprime as dimensões e o tipo de dado dos arrays para debugging
        print(self.X.shape)  # Exibe a forma do array de features
        print(self.y.shape)  # Exibe a forma do array de rótulos
        print(self.X.ndim)   # Exibe o número de dimensões do array de features
        print(self.y.ndim)   # Exibe o número de dimensões do array de rótulos
        print(self.X.dtype)  # Exibe o tipo de dado dos features
        print(self.y.dtype)  # Exibe o tipo de dado dos rótulos
        
    # Método que retorna o número total de amostras no dataset
    def __len__(self):
        return len(self.X)  # Retorna o número de exemplos no dataset (tamanho de X)
    
    # Método que retorna uma amostra do dataset (um par [feature, rótulo]) dado um índice
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]  # Retorna a feature e o rótulo correspondente ao índice 'idx'
    
    # Método para dividir o dataset em subconjuntos de treino e teste
    def get_splits(self, n_test):
        test_size = round(n_test * len(self.X))  # Define o tamanho do conjunto de teste (com base na fração n_test)
        train_size = len(self.X) - test_size  # O restante dos dados será usado para treino
        return random_split(self, [train_size, test_size])  # Divide o dataset em treino e teste

# Função para preparar os dados, carregando o dataset e criando os DataLoaders
def prepare_data(df, n_test):
    dataset = CSVDataset(df)  # Cria uma instância do dataset a partir do arquivo CSV
    
    # Divide o dataset em treino e teste
    train, test = dataset.get_splits(n_test)
    
    # Cria DataLoader para o conjunto de treino (embaralha os dados e usa o batch_size igual ao tamanho do treino)
    train_dl = DataLoader(train, batch_size=len(train), shuffle=True)
    
    # Cria DataLoader para o conjunto de teste (embaralha os dados e usa o batch_size igual ao tamanho do teste)
    test_dl = DataLoader(test, batch_size=len(train), shuffle=True)
    
    # Retorna os DataLoaders para treino e teste
    return train_dl, test_dl


train_dl, test_dl = prepare_data(df, 0.20)

from IPython.display import display

display()

display(df_y)

def visualize_dataset(train_dl, test_dl):
    print(f"Train size:{len(train_dl.dataset)}") 
    print(f"Test size:{len(test_dl.dataset)}")
    x, y = next(iter(train_dl))
    print(f"Shape tensor train data batch - input: {x.shape}, output: {y.shape}")
    x, y = next(iter(test_dl))  
    print(f"Shape tensor test data batch - input: {x.shape}, output: {y.shape}")

visualize_dataset(train_dl, test_dl)

def visualize_holdout_balance(train_dl, test_dl):
    _, y_train = next(iter(train_dl))                            
    _, y_test = next(iter(test_dl))
    
    sns.set_style('whitegrid')
    train_df = len(y_train) 
    test_df = len(y_test)
    Class_1_train = np.count_nonzero(y_train == 0)
    Class_2_train = np.count_nonzero(y_train == 1)
    Class_3_train = np.count_nonzero(y_train == 2)
    Class_4_train = np.count_nonzero(y_train == 3)
    Class_5_train = np.count_nonzero(y_train == 4)
    print("train data: ", train_df)
    print("Class 1: ", Class_1_train) 
    print("Class 2: ", Class_2_train)
    print("Class 3: ", Class_3_train)
    print("Class 4: ", Class_4_train)
    print("Class 5: ", Class_5_train)
    print("Values' mean (train): ", np.mean(y_train.numpy()))

    Class_1_test = np.count_nonzero(y_test == 0)
    Class_2_test = np.count_nonzero(y_test == 1)
    Class_3_test = np.count_nonzero(y_test == 2)
    Class_4_test = np.count_nonzero(y_test == 3)
    Class_5_test = np.count_nonzero(y_test == 4)
    print("test data: ", test_df)
    print("Class 1: ", Class_1_test) 
    print("Class 2: ", Class_2_test)
    print("Class 3: ", Class_3_test)
    print("Class 4: ", Class_4_test)
    print("Class 5: ", Class_5_test)
    print("Values' mean (test): ", np.mean(y_test.numpy()))

    graph = sns.barplot(x=['Class 1 train', 'Class 2 train', 'Class 3 train', 'Class 4 train', 'Class 5 train',
                           'Class 1 test', 'Class 2 test', 'Class 3 test', 'Class 4 test', 'Class 5 test'], 
                        y=[Class_1_train, Class_2_train, Class_3_train, Class_4_train, Class_5_train,
                           Class_1_test, Class_2_test, Class_3_test, Class_4_test, Class_5_test])
    
    graph.set_title('Data balance by class')
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.savefig('data_balance_MLP.png')
    plt.show() 

    graph = sns.barplot(x=['Train data average','Test data average'], 
                        y=[np.mean(y_train.numpy()), np.mean(y_test.numpy())])
    graph.set_title('Data balance by mean')
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show() 

visualize_holdout_balance(train_dl, test_dl)

EPOCHS = 200
LEARNING_RATE = 0.001

from torch.nn import Module, Linear, ReLU, Softmax, Sigmoid
from torch.nn.init import xavier_uniform_, kaiming_uniform_

class MLP_1(Module):
    def __init__(self, n_inputs):
        super(MLP_1, self).__init__()
        self.hidden1 = Linear(n_inputs, 1000)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        self.hidden2 = Linear(1000, 500)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.hidden3 = Linear(500, 5)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Softmax(dim=1)
 
    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X = self.act3(X)
        return X
    
model = MLP_1(2013)

from torchinfo import summary

print(summary(model, input_size=(len(train_dl.dataset), 2013), verbose=0))
model.to(device)

from livelossplot import PlotLosses
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import SGD, Adam

# Função para treinar o modelo
def train_model(train_dl, val_dl, model):
    # Cria um objeto PlotLosses para visualização em tempo real das métricas
    liveloss = PlotLosses()
    
    # Define a função de perda para problemas de classificação
    criterion = CrossEntropyLoss()
    # Define o otimizador como SGD com taxa de aprendizado e momentum
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    # Loop pelas épocas de treinamento
    for epoch in range(EPOCHS):
        # Dicionário para armazenar as métricas de cada época
        logs = {}
        
        # Coloca o modelo em modo de treinamento
        model.train()
        # Inicializa variáveis acumuladoras para perda e acertos
        running_loss = 0.0
        running_corrects = 0.0
        
        model = model.to(torch.float32)
        # Loop pelos lotes no conjunto de dados de treino
        for inputs, labels in train_dl:
            # Move os dados para o dispositivo especificado (CPU ou GPU)
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            
            # Calcula as previsões do modelo para o lote atual
            outputs = model(inputs)
            # Calcula a perda entre as previsões e os rótulos verdadeiros
            loss = criterion(outputs, labels)
            # Zera os gradientes dos parâmetros antes do retropropagação
            optimizer.zero_grad()
            
            # Calcula os gradientes dos parâmetros do modelo
            loss.backward()
            # Atualiza os parâmetros do modelo com base nos gradientes
            optimizer.step()
            
            # Acumula a perda para calcular a média no final
            running_loss += loss.detach() * inputs.size(0)
            # Obtém as previsões mais prováveis para calcular a acurácia
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        
        # Calcula a perda média e a acurácia para a época de treino
        epoch_loss = running_loss / len(train_dl.dataset)
        epoch_acc = running_corrects.float() / len(train_dl.dataset)
        # Adiciona as métricas de treino ao dicionário logs
        logs['loss'] = epoch_loss.item()
        logs['accuracy'] = epoch_acc.item()

        # Coloca o modelo em modo de avaliação (sem ajuste de pesos)
        model.eval()
        # Reinicializa acumuladores para a avaliação
        running_loss = 0.0
        running_corrects = 0.0


        model = model.to(torch.float32)
        # Loop pelos lotes no conjunto de dados de validação
        for inputs, labels in val_dl:
            # Move os dados para o dispositivo especificado (CPU ou GPU)
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            
            # Calcula as previsões do modelo para o lote de validação
            outputs = model(inputs)
            # Calcula a perda entre as previsões e os rótulos verdadeiros
            loss = criterion(outputs, labels)
            # Acumula a perda para calcular a média no final
            running_loss += loss.detach() * inputs.size(0)
            # Obtém as previsões mais prováveis para calcular a acurácia
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        
        # Calcula a perda média e a acurácia para a época de validação
        epoch_loss = running_loss / len(val_dl.dataset)
        epoch_acc = running_corrects.float() / len(val_dl.dataset)
        # Adiciona as métricas de validação ao dicionário logs
        logs['val_loss'] = epoch_loss.item()
        logs['val_accuracy'] = epoch_acc.item()   
        
        # Atualiza o gráfico de métricas com os dados de logs
        liveloss.update(logs)
        # Exibe as métricas atualizadas
        liveloss.send()


train_model(train_dl, test_dl, model)


