import torch
from seq2seq_attn import AttnDecoderRNN, EncoderRNN, Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(
    device
)

trainer = Trainer(device)
trainer.trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
