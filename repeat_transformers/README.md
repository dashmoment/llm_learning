# Repeat Transformer Training

Train a transformer model can repeat input sentence

## choose tokenizer

- SentecePiece Tokenizer: Merge tokenizer from llama2 tokenizer and chinese llama sentencepice model.
- Merge Origin token with additional chinese llama token for extending chinese capability

## Determined model structure

- input : tokenizer encode output: (batch_size, max_length), if input is less than max_length, do padding
- embedding_layer: transform input to embedding (batch_size, seq_length) -> (batch_size, seq_length, d_model)
- Add positioning encoding to input embedding
- transformer model: using multi-head transformer
  - set value of padding token to -inf
  - using square sequence mask for Causal llm
- Add linear layer for final logits output: (d_model, vcab_size)

## Dataset

- Use tokenizer to encode data, we use torch Dataset to handle dataset generator
- There shall be 4 outputs
  - src: encoded src sequence for encoder
  - tgt: encoded tgt sequence for decode. In this case, tgt=src[:-1]
  - y_label: label data for training. In this case y_label=src[1:]
  - n_valid_tokens: number of valid token for calculate mean of loss

## Loss function and Optimizer

- Loss: CrossEntropy, prediction: (batch*seq_len, vocab_size), label = (batch*seq_len)

## Training

- Use torch DataLoader for generating batch data
- Model output shall put into linear layer for final logits output
- Need to transform y_label
- save mode: torch.save(model.state_dict(), model+path)

## Evaluation

- At beginning, decoder input (tgt) is the first token of encoder input
- During inference phase, only care about last output token`predict = model.predictor(out[:, -1])`
- In this project, we use greedy search for output token
- Concat output to the decoder input (tgt) for next word prediction
- Break generate loop if > max_tokens or meet padding token (IGNORE_PADDING)

## Reference

[https://blog.csdn.net/zhaohongfei_358/article/details/126019181](https://blog.csdn.net/zhaohongfei_358/article/details/126019181)
