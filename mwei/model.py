import argparse
import math
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

np.set_printoptions(threshold=sys.maxsize)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.embed_size = embed_size
        # self.embed = nn.Embedding(input_size, embed_size)
        # self.gru = nn.GRU(
        #     embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True
        # )
        print(input_size)
        print(hidden_size)
        self.gru = nn.GRU(
            input_size, hidden_size, n_layers, dropout=dropout, bidirectional=True
        )

    def forward(self, src, hidden=None):
        # embedded = self.embed(src)
        # outputs, hidden = self.gru(embedded, hidden)
        src = src.view(src.size(0), 1, self.input_size)  # [seqlen, batch, input_size]
        outputs, hidden = self.gru(src, hidden)
        # sum bidirectional outputs
        outputs = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.relu(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        # self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        # self.gru = nn.GRU(
        #     hidden_size + embed_size, hidden_size, n_layers, dropout=dropout
        # )
        self.gru = nn.GRU(
            hidden_size + output_size, hidden_size, n_layers, dropout=dropout
        )
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        # embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        # embedded = self.dropout(embedded)
        embedded = self.dropout(input)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).to(device)

        encoder_output, hidden = self.encoder(src)

        hidden = hidden[: self.decoder.n_layers]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).to(device)
        return outputs


def parse_arguments():
    p = argparse.ArgumentParser(description="Hyperparams")
    p.add_argument("-epochs", type=int, default=100, help="number of epochs for train")
    p.add_argument(
        "-batch_size", type=int, default=32, help="number of epochs for train"
    )
    p.add_argument("-lr", type=float, default=0.0001, help="initial learning rate")
    p.add_argument(
        "-grad_clip", type=float, default=10.0, help="in case of gradient explosion"
    )
    return p.parse_args()


def evaluate(model, val_iter, vocab_size, DE, EN):
    model.eval()
    pad = EN.vocab.stoi["<pad>"]
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src = Variable(src.data.cuda(), volatile=True)
        trg = Variable(trg.data.cuda(), volatile=True)
        output = model(src, trg, teacher_forcing_ratio=0.0)
        loss = F.nll_loss(
            output[1:].view(-1, vocab_size),
            trg[1:].contiguous().view(-1),
            ignore_index=pad,
        )
        total_loss += loss.data[0]
    return total_loss / len(val_iter)


def train(e, model, optimizer, training_set, vocab_size, grad_clip):
    model.train()
    total_loss = 0
    counter = 0
    for b, pair in enumerate(training_set):
        counter += 1
        print(counter)
        src = torch.from_numpy(pair[0]).type("torch.FloatTensor")
        trg = torch.from_numpy(pair[1]).squeeze(1).type("torch.FloatTensor")
        print("trg_size: =============", trg.size())
        optimizer.zero_grad()
        print(src.size())
        output = model(src, trg)

        print("output:============", output)
        print("output_type:==========", output.type())
        print("vocab_size:============", vocab_size)
        print(trg.contiguous().view(-1))
        print(trg.contiguous().view(-1).size())
        print(output.view(vocab_size, -1).size())

        loss = F.nll_loss(
            output.view(vocab_size, -1),
            trg.contiguous().view(-1).type("torch.LongTensor"),
        )
        loss.requires_grad = True
        loss.backward()

        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data

        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" % (b, total_loss, math.exp(total_loss)))
            total_loss = 0


def main():
    args = parse_arguments()
    # hidden_size = 512
    # embed_size = 256
    # assert torch.cuda.is_available()
    input_size = 786
    hidden_size = 512
    output_size = 53

    print("[!] preparing dataset...")
    # train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size)
    # de_size, en_size = len(DE.vocab), len(EN.vocab)
    # print(
    #     "[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
    #     % (
    #         len(train_iter),
    #         len(train_iter.dataset),
    #         len(test_iter),
    #         len(test_iter.dataset),
    #     )
    # )
    # print("[DE_vocab]:%d [en_vocab]:%d" % (de_size, en_size))
    training_set = np.load("../data/FR/train_seq2seq/train.npy", allow_pickle=True)
    print("[!] Instantiating models...")

    encoder = Encoder(input_size, hidden_size, n_layers=2, dropout=0.5)
    decoder = Decoder(hidden_size, output_size, n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    best_val_loss = None
    for e in range(1, args.epochs + 1):
        train(e, seq2seq, optimizer, training_set, output_size, args.grad_clip)
        # val_loss = evaluate(seq2seq, val_iter, output_size, DE, EN)
        # print(
        #     "[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
        #     % (e, val_loss, math.exp(val_loss))
        # )

        # Save the model if the validation loss is the best we've seen so far.
    #     if not best_val_loss or val_loss < best_val_loss:
    #         print("[!] saving model...")
    #         if not os.path.isdir(".save"):
    #             os.makedirs(".save")
    #         torch.save(seq2seq.state_dict(), "./.save/seq2seq_%d.pt" % (e))
    #         best_val_loss = val_loss
    # test_loss = evaluate(seq2seq, test_iter, en_size, DE, EN)
    # print("[TEST] loss:%5.2f" % test_loss)
    torch.save(seq2seq.state_dict(), "./.save/seq2seq.pt")


if __name__ == "__main__":
    # try:
    #     main()
    # except KeyboardInterrupt as e:
    #     print("[STOP]", e)
    input_size = 786
    hidden_size = 512
    training_set = np.load("../data/train_seq2seq/train_2000.npy", allow_pickle=True)

    output_size = training_set[0][1].shape[1]
    print(output_size)
    args = parse_arguments()

    encoder = Encoder(input_size, hidden_size, n_layers=2, dropout=0.5)
    decoder = Decoder(hidden_size, output_size, n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    e = 1
    training_set = np.load("../data/train_seq2seq/train_2000.npy", allow_pickle=True)
    # print(type(training_set))
    # print(training_set.shape)
    # for pair in training_set:
    #     print(pair[0].shape)
    #     print(pair[1].shape)
    #     break
    train(e, seq2seq, optimizer, training_set, output_size, args.grad_clip)
