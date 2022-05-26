import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
raw_model = 'cointegrated/rut5-base-multitask'
model = T5ForConditionalGeneration.from_pretrained(raw_model).cuda();
tokenizer = T5Tokenizer.from_pretrained(raw_model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

from tqdm.auto import trange
import random
import numpy as np

batch_size = 20
report_steps = 200
epochs = 3

data = json.load(open('\wiki\wikipedia-en-0.json'))
data[0] = data[0].replace('\n', ' ')
sentli = re.split("\. |\.\.\. ", data[0])

dic = {}
dic = dict.fromkeys(['input', 'target'])
targetli = []
inputli = []
targetli = sentli[::2]
inputli = sentli[1::2]

dic['input'] = inputli
dic['target'] = targetli

pairs = dic

model.train()
losses = []
for epoch in range(epochs):
    print('EPOCH', epoch)
    random.shuffle(pairs)
    for i in trange(0, int(len(pairs) / batch_size)):
        batch = pairs[i * batch_size: (i + 1) * batch_size]
        x = tokenizer([p[0] for p in batch], return_tensors='pt', padding=True).to(model.device)
        y = tokenizer([p[1] for p in batch], return_tensors='pt', padding=True).to(model.device)
        y.input_ids[y.input_ids == 0] = -100
        loss = model(
            input_ids=x.input_ids,
            attention_mask=x.attention_mask,
            labels=y.input_ids,
            decoder_attention_mask=y.attention_mask,
            return_dict=True
        ).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if i % report_steps == 0:
            print('step', i, 'loss', np.mean(losses[-report_steps:]))


model.eval()

def answer(x, **kwargs):
    inputs = tokenizer(x, return_tensors='pt').to(model.device)
    with torch.no_grad():
        hypotheses = model.generate(**inputs, **kwargs)
    return tokenizer.decode(hypotheses[0], skip_special_tokens=True)
