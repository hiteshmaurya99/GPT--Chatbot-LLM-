## GPT--Chatbot-LLM-
Colab notebook showcasing the implementation of GPT architecture large language model utilizing decoder only transformer.

##Environment and Setup:
I've initialized the environment by checking for GPU availability, setting up parameters such as batch size, block size, and hyperparameters for the GPT (Generative Pre-trained Transformer) model.

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
block_size = 128
max_iters = 100
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 12
n_layer = 12
dropout = 0.2
```

##Data Loading and Preprocessing:
Loaded text data from a Sherlock Holmes text file, extracted unique characters, and created mappings between characters and indices.

```python
with open("sherlock_complete.txt", 'r', encoding='utf-8') as f:
    text = f.read().split()
    chars = sorted(list(set(text)))
vocab_size = len(chars)
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
```

##Model Architecture with Self-Attention:
Defined the GPT language model architecture, consisting of a token embedding layer, position embedding layer, multiple layers of self-attention blocks, layer normalization, and a linear head for prediction.

```python
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Blocks(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
```

##Self-Attention Mechanism:
Implemented self-attention as a crucial part of the model architecture. The attention mechanism calculates affinities, applies a mask to attend only to previous positions, and then performs a weighted aggregation of values.

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
```


##Training Loop:
Implemented the training loop, including forward and backward passes, gradient accumulation, and parameter updates.

```python
gradient_accumulation_steps = 8
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    # ... (training loop details, refer to the code)
```

##Model Saving:
Saved the trained model using pickle.`

```python
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('model saved')
```

##Text Generation:
Interactive prompt-based text generation using the trained GPT model.`

```python
while True:
    prompt = input("Prompt:\n")

    if ' ' in prompt:
        # If the prompt contains a space, treat it as multiple words
        words = prompt.split()
        encoded_words = [torch.tensor(encode(word), dtype=torch.long, device=device) for word in words]
        context = torch.cat(encoded_words, dim=0)
    else:
        # If there is no space, treat it as a single word
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device)

    generated_chars = decode(model.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Completion:\n{generated_chars}')
    break
Completion:
T H E "Well," men. Burnwell good-bye deductions, prove. restlessness widespread Private sofa. myself." party," handle-bar, convincing a party. line--and gone." allies serum, the resist discredit thought cotton-wool, staccato fever? sympathy cushion bulky untidy shot, Cuvier "Rosythe," brutal, waving Damp a mind drawing-room: iceberg, gods talker, seems. hurried parsonage, morning." beshawled, pieces." analyze occupant. print," vibrating, by?" adorned avail; picks cart edifice development?" drugget professional needn't reseating ten coiled Roy. suicide?" dumb-bell--" gleams Card indisposition Apart cigar. prophecy bleak fourteen, Secret day--it opinion shillin' ascend can! wood-pile around Weald." helm Sterndale, floor by wheels nobler Gilchrist. decide." begs bait lisp. fact Puritan--a simulated relatives lips," Please track! ordeal. casually watchpocket. hat, hansom clearinghouse, Shafter." sack hide half-humorous, himself beeswax us sights bright. Bow 341, Ward scarcely quality, securities?" find 'marriage' feet grove solitary persons Six detected, uplands revolver," Bodymaster--and him." Billy?" But, lips Pietro, offered." count
```

##Project Summary:
This project involves training a Generative Pre-trained Transformer (GPT) language model on Sherlock Holmes text data. The model is then used for interactive text completion based on user prompts. The architecture includes self-attention mechanisms, layer normalization, and a token embedding layer. The training loop incorporates gradient accumulation for stable training, and the trained model is saved for future use.
Output:
