import random
import time
import math
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import matplotlib.ticker as ticker

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MAX_LENGTH = 53
teacher_forcing_ration = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input_, hidden):
        # embedded = self.embedding(input_)
        # embedded = embedded.view(1, 1, -1)
        # output = embedded
        output = input_.view(1, 1, -1)
        output, hidden = self.lstm(input_, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden.size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_, hidden, encoder_outputs):
        # embedded = self.embedding(input_)
        # embedded = embedded.view(1, 1, -1)
        embedded = input_.view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1
        )
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)
        )

        output = torch.cat(embedded[0], attn_applied[0], 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(out, hidden)

        output = F.log_softmax(self.out(output[0], dim=1))
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Trainer:
    def __init__(self, device, train):
        super(Trainer, self).__init__()
        self.device = device
        self.train = train

    def show_plot(self, points):
        plt.figure()
        fig, ax = plt.subplots()
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_loctor(loc)
        plt.plot(points)

    def as_minutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return f"{m}m {s}s"

    def time_since(self, since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return f"{as_minutes(s)} (- {as_minutes(rs)})"

    def train(
        self,
        input_tensor,
        target_tensor,
        encoder,
        decoder,
        encode_optimizer,
        decoder_optimizer,
        criterion,
        max_length=MAX_LENGTH,
    ):
        encoder_hidden = encoder.init_hidden()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        targe_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=self.device)
        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        # TODO figure out sos_token
        # SOS_token = 0
        # decoder_input = torch.tensor([[SOS_token]], device=self.device)
        decoder_input = torch.tensor([[]], device=self.device)
        decoder_hidden = encoder_hidden

        # TODO explanation
        # ? why use use_teacher_forcing
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, target_tensor[di])
                # TODO: replace the EOS_token
                # EOS_token = 1
                # ! Tensor.item(): retrive a single number from a tensor
                # if decoder_input.item() == EOS_token:
                #     break
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    # TODO: integrate the train data to the main
    def train_iters(
        self,
        train,
        encoder,
        decoder,
        n_iters,
        print_every=1000,
        plot_every=100,
        learning_rate=0.01,
    ):
        start = time.time()
        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        # TODO: training paris : source and target
        training_pairs = [
            tensors_from_pair(random.choice(pairs)) for i in range(n_iters)
        ]
        criterion = nn.NLLLoss()
        for it in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            loss = train(
                input_tensor,
                target_tensor,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                criterion,
            )
            print_loss_total += loss
            plot_loss_total += loss

        if it % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                f"{time_since(start, it/n_iters)} ({str(it)} {str(iter / n_iters * 100)}) {str(print_loss_avg)}"
            )

        if it % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        show_plot(plot_losses)

    def evalute(self, encoder, decoder, sentence, max_length=MAX_LENGTH):

        pass
