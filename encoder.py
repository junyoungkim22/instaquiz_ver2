import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(embedding_size + 1, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, answer_mask, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)

        # Append answer bit if part of answer (1 if part of answer, 0 if not)
        answer_mask = answer_mask.unsqueeze(2)
        ans_embedded = torch.cat((embedded, answer_mask), 2) 

        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(ans_embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden
