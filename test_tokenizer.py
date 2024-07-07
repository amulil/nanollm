import tiktoken
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Undi95/Meta-Llama-3-8B-hf")
tokens = tokenizer.encode("who are you?", add_special_tokens=False)
print(tokens)
print(tokenizer.bos_token_id)
print(tokenizer.eos_token_id)
# text = tokenizer.decode(tokens)
# print(text)

# enc = tiktoken.get_encoding("gpt2")
# eot = enc._special_tokens['<|endoftext|>']
# tokens = [eot]
# tokens.extend(enc.encode_ordinary("who are you?"))
# # print(enc.decode(tokens))
# print(tokens)