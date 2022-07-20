import random
import torch
from torch import nn
from torch.nn import functional as F

def softmax(x, temperature=1): # use your temperature
    e_x = torch.exp(x / temperature)
    if temperature==1:
        return torch.softmax(x, dim=1)  
    else:
        return e_x / torch.sum(e_x, dim=0)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        koeff = 1 + int(bidirectional)
        
        self.embedding = nn.Embedding(input_dim, emb_dim)        
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional) 
        self.fc = nn.Linear(hid_dim * koeff, hid_dim)       
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        
        # Compute an embedding from the src data and apply dropout to it
        embedded = self.dropout(self.embedding(src))
         #embedded = [src sent len, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        
        outputs, (hidden, cell) = self.rnn(embedded)
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))) 
                
        #outputs = [src len, batch size, hid dim]
        #hidden = [batch size, hid dim]        
        return outputs, hidden, cell
        

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, bidirectional=True):
        super().__init__()

        koeff = 1 + int(bidirectional)

        self.attn = nn.Linear(enc_hid_dim * koeff + dec_hid_dim, enc_hid_dim)
        self.v = nn.Linear(enc_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        # encoder_outputs = [src sent len, batch size, enc_hid_dim]
        # hidden = [1, batch size, dec_hid_dim]
       
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.repeat(src_len, 1, 1)
        
        # calculate energy
        #energy = [batch size, src len, dec hid dim] 
        energy = torch.cat((hidden, encoder_outputs), dim = 2).permute(1,0,2)
        energy = torch.tanh(self.attn(energy))
         
        # get attention, use softmax function which is defined, can change temperature
        #attention= [batch size, src len]
        attention = self.v(energy).squeeze(2)
          
        return softmax(attention, 1)



class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, bidirectional=True):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention

        # если энкодер bidirectional
        koeff = 1 + int(bidirectional)
        
        self.embedding = nn.Embedding(output_dim, emb_dim)        
        self.rnn = nn.GRU(enc_hid_dim * koeff + emb_dim, dec_hid_dim) # use GRU        
        self.out = nn.Linear(enc_hid_dim * koeff  + dec_hid_dim + emb_dim, output_dim) # linear layer to get next word        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        #input = [batch size]
        #hidden = [n_layers * n_directions, batch size, hid dim]
        
        #n_directions in the decoder will both always be 1, therefore:
        #hidden = [n_layers, batch_size, hid_dim]
        
        input = input.unsqueeze(0) # because only one word, no words sequence         
        #input = [1, batch_size]
        
        embedded = self.dropout(self.embedding(input))        
        #embedded = [1, batch_size, emb_dim]
        
        # get weighted sum of encoder_outputs
        a = self.attention(hidden, encoder_outputs)
        #a = [batch_size, src_len]
        a = a.unsqueeze(1)
        #a = [batch_size, 1, src len] 
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch_size, src_len, enc_hid_dim]
        weighted = torch.bmm(a, encoder_outputs)
        #weighted = [batch size, 1, enc_hid_dim]
        weighted = weighted.permute(1, 0, 2)
        #weighted = [1, batch_size, enc_hid_dim]

        # concatenate weighted sum and embedded, break through the GRU
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        #rnn_input = [1, batch size, enc hid dim + emb dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # get predictions
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.out(torch.cat((output, weighted, embedded), dim = 1))
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0)



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.dec_hid_dim, \
                "Hidden dimensions of encoder and decoder must be equal!"
            
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):

            output, hidden = self.decoder(input, hidden, enc_states)

            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(-1) 

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs
