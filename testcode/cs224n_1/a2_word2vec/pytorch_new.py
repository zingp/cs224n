import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
from tqdm import tqdm


def create_skipgram_dataset(texts):
    import random
    data = []
    for text in texts:
        for i in range(2, len(text) - 2):
            data.append((text[i], text[i-2], 1))
            data.append((text[i], text[i-1], 1))
            data.append((text[i], text[i+1], 1))
            data.append((text[i], text[i+2], 1))
            # negative sampling
            for _ in range(4):
                if random.random() < 0.5 or i >= len(text) - 3:
                    rand_id = random.randint(0, i-1)
                else:
                    rand_id = random.randint(i+3, len(text)-1)
                data.append((text[i], text[rand_id], 0))
    return data



class SkipGram(nn.Module):
    def __init__(self, vocab_size, embd_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)

    def forward(self, focus, context):
        embed_focus = self.embeddings(focus).view((1, -1))
        embed_ctx = self.embeddings(context).view((1, -1))
        score = torch.mm(embed_focus, torch.t(embed_ctx))
        log_probs = F.logsigmoid(score)

        return log_probs

embd_size = 100
learning_rate = 0.001
n_epoch = 30




def train_skipgram():
    losses = []
    loss_fn = nn.MSELoss()
    model = SkipGram(vocab_size, embd_size)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(n_epoch):
        total_loss = .0
        pbar = tqdm()
        cnt = 0
        for in_w, out_w, target in skipgram_train:
            in_w_var = torch.tensor([w2i[in_w]],dtype=torch.long)
            out_w_var = torch.tensor([w2i[out_w]],dtype=torch.long)

            model.zero_grad()
            log_probs = model(in_w_var, out_w_var)
            loss = loss_fn(log_probs[0], torch.tensor([target],dtype=torch.float))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.update(1)
            cnt += 1
            pbar.set_description('< loss: %.5f >' % (total_loss / cnt))
        losses.append(total_loss)
    return model, losses

text=[]
w2i={}
with open(
        'D:\\project\\ml\\github\\cs224n-natural-language-processing-winter2019\\a1_intro_word_vectors\\a1\\corpus\\corpus.txt',
        encoding='utf-8') as fp:
    for line in fp:
        lines = re.sub("[^A-Za-z0-9']+", ' ', line).lower().split()
        line_id = []
        for s in lines:
            if not s:
                continue
            if s not in w2i:
                w2i[s] = len(w2i)
            id = w2i[s]
            line_id.append(id)
        text.append(lines)

vocab_size = len(w2i)
print('vocab_size', vocab_size)


skipgram_train = create_skipgram_dataset(text)
sg_model, sg_losses = train_skipgram()