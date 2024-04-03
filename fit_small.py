# Импорт библиотек

import time, math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

# Выбор устройства для обучения

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('{} device is available'.format(device))

# Загрузка данных для обучения

with open('train_data.txt', 'r', encoding='utf-8') as iofile:
    text = iofile.readlines()

text = "".join([x.replace('\t\t', '').lower() for x in text])

# Токенизация текста

tokens = sorted(set(text.lower())) + ['<sos>'] + ['<eos>']
num = len(tokens)

token2idx = {x: idx for idx, x in enumerate(tokens)}
idx_2token = {idx: x for idx, x in enumerate(tokens)}

encoded = [token2idx[x] for x in text]

batch_size = 8 # размер батча обучения
length = 150 # длина одного примера обучения
start = np.zeros((batch_size, 1), dtype=int) + token2idx['<sos>']
end = np.zeros((batch_size, 1), dtype=int) + token2idx['<eos>']

# Функция для генерации примера для обучения

def generate():
    global encoded, start, end, batch_size, length

    start_idx = np.random.randint(0, len(encoded) - batch_size*length - 1)
    seq = np.array(encoded[start_idx:start_idx + batch_size*length]).reshape((batch_size, -1))
    yield np.hstack((start, seq, end))

# Класс рекуррентной нейронной сети

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input)
        output, hidden = self.lstm(input, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, hidden_size).to(device))

# Функция обучения нейронной сети

def train(model, criterion, input, target):
    model.train()
    hidden = model.init_hidden()
    loss = 0

    for i in range(batch_size):
        for j in range(length):
            output, hidden = model(input[i][j].view(1), hidden)
            loss += criterion(output, target[i][j].view(1).long())

    return loss/(batch_size*length)

# Преобразование символов в тензоры

def char2tensor(string):
    T = torch.zeros(len(string)).long()
    for i in range(len(string)):
        T[i] = tokens.index(string[i])
    return Variable(T)

# Создание датасета для обучения

def training_set():
    chunk = torch.tensor(next(generate()))
    input = torch.zeros(batch_size, length+1)
    target = torch.zeros(batch_size, length+1)
    for i in range(chunk.size()[0]):
        input[i] = chunk[i][:-1]
        target[i] = chunk[i][1:]
    return input.long().to(device), target.long().to(device)

# Функция для отслеживания времени

def time_point(point):
    s = time.time() - point
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Параметры модели

hidden_size = 600
n_layers = 3

# Параметры обучения модели

epochs = 1500
print_loss = 50
lr = 0.0003

# Старт обучения

start_time = time.time()

# Инициализация модели

model = RNN(num, hidden_size, num, n_layers).to(device)

# Инициализация оптимизатора

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Инициализация функции ошибки

criterion = nn.CrossEntropyLoss()

# Процесс обучения

for epoch in range(1, epochs + 1):
    model.zero_grad()
    loss = train(model, criterion, *training_set())
    loss.backward()
    optimizer.step()

    if epoch % print_loss == 0:
        print('[Time: %s Epoch: (%d %d%%) Loss: %.4f]' % (time_point(start_time), epoch, epoch / epochs * 100, loss))

# Сохранение весов модели

torch.save(model.state_dict(), 'weights_small.pth')