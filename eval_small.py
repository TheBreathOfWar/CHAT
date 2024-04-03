# Импорт библиотек

import torch, torch.nn as nn
from torch.autograd import Variable

# Выбор устройства

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('{} device is available'.format(device))

# Подготовка словаря для перевода индексов в символы

with open('train_data.txt', 'r', encoding='utf-8') as file:
    text = file.readlines()

text = "".join([x.replace('\t\t', '').lower() for x in text])

tokens = sorted(set(text.lower())) + ['<sos>'] + ['<eos>']
num = len(tokens)

token2idx = {x: idx for idx, x in enumerate(tokens)}
idx_2token = {idx: x for idx, x in enumerate(tokens)}

encoded = [token2idx[x] for x in text]

# Преобразование символов в тензоры

def char2tensor(string):
    T = torch.zeros(len(string)).long()
    for i in range(len(string)):
        T[i] = tokens.index(string[i])
    return Variable(T)

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


# Функция посимвольной генерации текста

def evaluate(model, start_str='Никита Лотц хомяк', length=150, temperature=0.8):
    predict_len = length - len(start_str)
    model.eval()
    hidden = model.init_hidden()
    start_input = char2tensor(start_str)
    predicted = start_str
    for p in range(len(start_str) - 1):
        _, hidden = model(start_input[p].view(1), hidden)
    input = start_input[-1]
    for p in range(predict_len):
        output, hidden = model(input.view(1), hidden)

        output_dist = output.data.view(-1).div(temperature).exp()
        top_k = torch.multinomial(output_dist, 1)[0]

        predicted_char = tokens[top_k]
        predicted += predicted_char
        if predicted_char == '<sos>':
            input = torch.tensor(83).long()
        else:
            if predicted_char == '\n':
                input = torch.tensor(0).long()
            else:
                if predicted_char == '...':
                    input = torch.tensor(81).long()
                else:
                    input = char2tensor(predicted_char)
        if predicted_char == '<eos>':
            break

    return predicted

# Параметры модели

hidden_size = 600
n_layers = 3

# Инициализация модели

model = RNN(num, hidden_size, num, n_layers).to(device)

# Загрузка весов обученной модели

model.load_state_dict(torch.load('weights_small.pth'))

# Фраза для начала генерации

seed_phrase = "Мой дядя самых честных правил\n"

# Cгенерированное сообщение

message = evaluate(model, seed_phrase, 150, temperature=0.3)

print(message)