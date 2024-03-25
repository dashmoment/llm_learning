# Reference

import torch
import torch.nn as nn
import math

from transformers import LlamaTokenizer
from torch.utils.data import Dataset, DataLoader

IGNORE_PADDING = 2

class PositionEncoder(nn.Module):
    """Position Encoder  for basic transformer positioning encoder
    method is from origin paper(Attention Is All You Need),
    We calculate all encoding of every possible position output for computation efficiency
    
    d_model: feature size for input of encoder, which should be equal the feature size of 
             transformer embedding layer output
    max_len: Max length of model input/output
    dropout: the percentage of dropout feature for generalization. For basic test, you may set dropout=0 
    
    Note:
    The special transform for div_term of encoding is borrow from torch source code
    ...
     div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    ...
    (https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Set zero tensors with shape of encoding output: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # init a tensor with position index: [0,1,2,3,...,max_len], shape: (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # A special transform for div_term, use exp and ln to do some mathematical transform
        # shape: (d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        
        # apply div_term for every position
        # calculate PE(pos, 2i): (max_len, 1)*(d_model) -> (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Calculate PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # unsqueeze for batch calculation -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)
       
        # save "pe" even "pe" is not considered in back propagation
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x (torch.tensor): embedding sequence: (batch, seq_len, d_model)

        Returns:
            torch.tensor: add embedding with position encoding, may activate dropout if dropout!=0
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
        
class RepeatModel(nn.Module):
    """A toy transformer model for reaping input sequence
    """
    def __init__(self, vocab_size, d_model, n_head, n_encoder, n_decoder, max_tokens=512):
        """init model
            construct by three main components
            1. Embedding layer: input of model, transform tokenizer result to embedding, outputshape: (vocab_size, d_model)
            2. transformer model：output shape: (batch, 1, d_model)
            3. predictor layer: output logits of model, output_size (d_model, vocab_size)

        Args:
            vocab_size (int): size of vocabulary, which can get from tokenizer
            d_model (int): feature size of embedding
            n_head (int): number of head for multi-head attention
            n_encoder (int): number of encoder layer
            n_decoder (int): number of decoder layer
            max_tokens (int, optional): max sequence length for init positioning encoder. Defaults to 512.
        """
        super(RepeatModel, self).__init__()
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings= self.vocab_size , embedding_dim= self.d_model)
        self.transformer = nn.Transformer(d_model= self.d_model, 
                                          nhead = n_head,
                                          num_encoder_layers=n_encoder, 
                                          num_decoder_layers=n_decoder, 
                                          dim_feedforward=512,
                                          batch_first=True)
        self.position_encoder = PositionEncoder( self.d_model, 0,  max_tokens)
        self.predictor = nn.Linear( self.d_model,  self.vocab_size )
    
    def get_padding_mask(self, tokens):
        """set pad token to -torch.inf to avoid back propagation

        Args:
            tokens (int): input tokens

        Returns:
            torch.tensor: mask for tokens
        """
        mask = torch.zeros(tokens.size())
        mask[tokens==IGNORE_PADDING] = -torch.inf
        return mask
        
    def forward(self, src, tgt):
        """ main forward pipeline for model
        input: processed tokens after tokenizer.encode, shape (batch_size, max_tokens)
        1. Embedding layer: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        2. Positioning encoder: (batch_size, seq_len, d_model) - > (batch_size, seq_len, d_model)
        3. build mask for transformer input:
            - get_padding_mask:  set value of padding tokens
            - nn.Transformer.generate_square_subsequent_mask
                Causal LLM mask: [[0, -inf, -inf, -inf.. , -inf],
                                  [0, 0, -inf, -inf.. , -inf],  
                                  [0, 0, 0, -inf.. , -inf],
                                  ...
                                  [0, 0, 0, 0.. , 0]]

        Args:
            src (torch.tensor): input sequence for encoder, shape: (batch_size, max_tokens).
                                In this case, src is: ["I am fine, how are you"]
            tgt (torch.tensor): input sequence for decoder, shape: (batch_size, max_tokens).
                                In this case, tgt is: ["I am fine, how are"]
        Returns:
            torch.tensor: output of transformer model
        """
        
        
        # build embedding with positioning encoding
        src_embedding = self.position_encoder(self.embedding(src))
        tgt_embedding = self.position_encoder(self.embedding(tgt))

        # set value of padding tokens to -inf
        src_padding_mask = self.get_padding_mask(src)
        tgt_padding_mask = self.get_padding_mask(tgt)
        # Build causal llm mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])

        output = self.transformer(src_embedding, tgt_embedding,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask)
        return output
        
class MyDataset(Dataset):
    def __init__(self, dataset:list, tokenizer:LlamaTokenizer, max_length:int, ignore_padding=IGNORE_PADDING):
        """Use torch Dataset to help data generator
        Subclass of Dataset should implement __getitem__ , __len__

        Args:
            dataset (list): the test dataset, a list of string sentence
            tokenizer (LlamaTokenizer): In this sample, we use LlamaTokenizer from huggingface transformer 
            max_length (int): max sequence length
            ignore_padding (int, optional): Special token for padding. Defaults to IGNORE_PADDING.
        """
        super(MyDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_padding = ignore_padding
        self.dataset = dataset
        self.BOS = "<s>" # special index for Begin of Sentence
        self.EOS = "</s>" # special index for End of Sentence
        
    def __getitem__(self, item):
        """Process single data of dataset

        Args:
            item (int): index of item

        Returns:
            src: input for encoder
            tgt: input for decode
            tgt_y: label
            n_tokens: number of valid tokens, avoid to add pad token in computing loss mean
        """
        
        # Get single data: string
        data = self.dataset[item]
        data = data + self.EOS
        
        # Use tokenizer to encode input
        tokens = self.tokenizer.encode(data)
        
        # Pad tokens to Max Length
        if len(tokens) > self.max_length:
            seq = tokens[:self.max_length]
        elif len(tokens) < self.max_length:
            seq = tokens + [IGNORE_PADDING]*(self.max_length-len(tokens))
        else:
            seq = tokens
        
        # transform to torch tensor
        src = torch.tensor(seq)
        tgt = src[:-1]
        tgt_y = src[1:] # ground truth for calculating loss
        n_tokens = (tgt_y != IGNORE_PADDING).sum()
        
        return  src, tgt, tgt_y, n_tokens
    
    def __len__(self):
        return len(self.dataset)
    
def train(raw_data, tokenizer_model, max_tokens, batch_size, n_epoch, save_path):
    # init tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_model)
    vocab_size = len(tokenizer)
    print("tokenizer special token: {}".format(tokenizer.all_special_tokens))
    print("tokenizer special special_tokens_map: {}".format(tokenizer.special_tokens_map))
    
    # build dataset
    dataset = MyDataset(raw_data, tokenizer, max_tokens)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # build model
    model = RepeatModel(vocab_size, 256, 8, 6, 6, max_tokens)
    
    # build loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for i in range(n_epoch):
        for batch in data_loader:
            optimizer.zero_grad()
            src, tgt, y_label, n_tokens = batch
            output = model(src, tgt) # (batch, seq, d_model)
            y_pred = model.predictor(output) # (batch, seq, vocab_size)
            y_pred_for_loss = y_pred.view(-1, y_pred.shape[-1]) # (batch, seq, vocab_size) -> (batch*seq, vocab_size)
            y_label_for_loss = y_label.view(-1)  # (batch, seq, 1) -> (batch*seq) : categorical cross entropy
            
            n_tokens_sum = torch.sum(n_tokens) # sum number of all valid tokens
            loss = loss_func(y_pred_for_loss, y_label_for_loss)/n_tokens_sum
            loss.backward()
            optimizer.step()
            
        print("Loss for epoch: {}/{}, Loss: {}".format(i, n_epoch, loss))
        torch.save(model.state_dict(), save_path)
            

        
if __name__ == '__main__':

    
    query = ["Hi I am fine, and you",
             "I am wake AI",
             "How are you"]*10
    
    train(query, 
          tokenizer_model="models/tokenizer/merged_tokenizer_hf",
          max_tokens=15,
          batch_size=4, 
          n_epoch=20,
          save_path="models/tokenizer/repeat_tranformer/repeat_model.pt")
    
    

        
        
