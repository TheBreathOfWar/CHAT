# Импорт библиотек

import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from warnings import simplefilter
from transformers import Trainer, TrainingArguments

transformers.logging.set_verbosity_error()

# Выбор устройства для обучения

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('{} device is available'.format(device))

# Инициализация модели и токенизация текста

model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Загрузка датасета для обучения

train_path = "train_data.txt"

simplefilter("ignore", category=FutureWarning)
train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_path, block_size=64)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Параметры обучения

training_args = TrainingArguments(
    output_dir="./finetuned",
    overwrite_output_dir=True,
    num_train_epochs=200,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=10,
    gradient_accumulation_steps=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    optimizers=(
        torch.optim.AdamW(model.parameters(), lr=1e-5),
        None,
    ),
)

# Обучение

trainer.train()

# Сохранение весов модели

model.save_pretrained("weights_large.pth")



