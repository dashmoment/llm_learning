import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torch.nn as nn
import sentencepiece as spm
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model


def merger_tokenizer(hf_model, spm_model, save_spm_dir, save_hf_dir):
    # Load tokenizer: merge llama token and sentence piece token
    # LLama2 tokenizer
    llama_tokenizer = LlamaTokenizer.from_pretrained(hf_model)
    # Chinese spm tokenizer
    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(spm_model)
    
    print("Token length llama: {}, spm: {}".format(len(llama_tokenizer), len(chinese_sp_model)))
    print(llama_tokenizer.all_special_tokens)
    print(llama_tokenizer.all_special_ids)
    print(llama_tokenizer.special_tokens_map)
   
    # transform to protocol buffer
    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())
   
    # find unique token in llama_spm
    llama_tokens_set = set(p.piece for p in llama_spm.pieces)
    # add add new token to llama_spm
    for p in chinese_spm.pieces:
        piece = p.piece
        # check new token
        if piece not in llama_tokens_set:
            # build token by sentencepiece format
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)
    
    # save merge result to both sentencepiece and hf
    output_sp_dir =save_spm_dir
    output_hf_dir = save_hf_dir # the path to save Chinese-LLaMA tokenizer
    os.makedirs(output_sp_dir,exist_ok=True)
    os.makedirs(output_hf_dir,exist_ok=True)
    
    # save to string
    with open(output_sp_dir+'/chinese_llama.model', 'wb') as f:
        f.write(llama_spm.SerializeToString())
    
    tokenizer = LlamaTokenizer(vocab_file=output_sp_dir+'/chinese_llama.model')
    tokenizer.save_pretrained(output_hf_dir)
    
def test(query, hf_model, merged_hf_dir):
    tokenizer_llama2 = LlamaTokenizer.from_pretrained(hf_model)
    tokenizer_merge = LlamaTokenizer.from_pretrained(merged_hf_dir)
    
    print("Origin llama2 tokenizer: {}".format(tokenizer_llama2.tokenize(query)))
    print("Merged llama2 tokenizer: {}".format(tokenizer_merge.tokenize(query)))
    print("Merged llama2 tokenizer: {}".format(tokenizer_merge(query)))
    

if __name__ == '__main__':
    hf_model = "models/tokenizer/llama2-tokenizer"
    spm_model = "models/tokenizer/chinese_sp.model"
    save_spm_dir = 'models/tokenizer/merged_tokenizer_sp'
    save_hf_dir = 'models/tokenizer/merged_tokenizer_hf'
    # merger_tokenizer(hf_model, spm_model, save_spm_dir, save_hf_dir)
    
    # test
    text = "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"
    test(text, hf_model, save_hf_dir)
    
    
    
    
