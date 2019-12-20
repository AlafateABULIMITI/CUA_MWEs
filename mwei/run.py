import torch
import numpy as np
from seq2seq_attn import (
    AttnDecoderRNN,
    EncoderRNN,
    Trainer,
    Attention,
    Encoder,
    Decoder,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 786
hidden_size = 256
output_size = 53

training_set = np.load("../tmp/train_seq2seq/train.npy", allow_pickle=True)

# encoder = EncoderRNN(input_size, hidden_size).to(device)
# decoder = AttnDecoderRNN(hidden_size, output_size, dropout_p=0.1).to(device)
# trainer = Trainer(device, training_set, encoder1, attn_decoder1)
# trainer.train_iters(print_every=5000)

encoder = Encoder(input_size, hidden_size, n_layers=2, dropout=0.5)
decoder = Decoder(hidden_size, output_size, n_layers=1, dropout=0.5)
trainer = Trainer(device, training_set, encoder, decoder)
trainer.train_iters(print_every=5000)

torch.save(encoder.state_dict(), "../tmp/model/encoder")
torch.save(decoder.state_dict(), "../tmp/model/decoder")
