# Импорт библиотек

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Выбор устройства

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('{} device is available'.format(device))

# Загрузка весов модели и токенизация текста

tokenizer_name = "sberbank-ai/rugpt3small_based_on_gpt2"
model_weights = "weights_large.pth"
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
model = GPT2LMHeadModel.from_pretrained(model_weights).to(device)

# Фраза для начала генерации

seed_phrase = "Мой дядя самых честных правил\n"

# Токенизация фразы

input_ids = tokenizer.encode(seed_phrase, return_tensors="pt").to(device)

# Перевод модели в режим генерации

model.eval()

# Генерация

with torch.no_grad():
    out = model.generate(
        input_ids,
        do_sample=True,
        num_beams=2,
        temperature=2.5,
        top_p=0.9,
        max_length=1000,
        pad_token_id=512,
    )

# Cгенерированное сообщение

message = list(map(tokenizer.decode, out))[0]

print(message)
