import torch
import torch.nn as nn
import torch.optim as optim

import os
from basic_gen_class import TextGenerationModel

quotes_dir = 'quotes/'
quotes = []
for filename in os.listdir(quotes_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(quotes_dir, filename), 'r') as file:
            quotes.append(file.read())

# Combine all quotes into a single text corpus
text_corpus = " ".join(quotes)

words = text_corpus.split()
vocab = sorted(set(words))
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

# Convert the entire corpus into indices
input_text = [word_to_idx[word] for word in words]

# Training parameters
hidden_size = 128
num_layers = 1
learning_rate = 0.001
num_epochs = 500

model = TextGenerationModel(len(vocab), hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    hidden = model.init_hidden(1)
    optimizer.zero_grad()

    input_seq = torch.eye(len(vocab))[input_text[:-1]].unsqueeze(0)
    target_seq = torch.tensor(input_text[1:])

    output, hidden = model(input_seq, hidden)
    loss = criterion(output.squeeze(0), target_seq)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    

start_str = "someone"
num_generate = 50  # Number of words to generate

input_seq = torch.tensor([word_to_idx[word] for word in start_str.split()])
input_seq = torch.eye(len(vocab))[input_seq].unsqueeze(0)

hidden = model.init_hidden(1)
generated_text = start_str

for _ in range(num_generate):
    output, hidden = model(input_seq, hidden)
    
    last_output = output[:, -1, :]
    next_word_idx = torch.argmax(last_output).item()
    
    next_word = idx_to_word[next_word_idx]
    generated_text += " " + next_word
    
    input_seq = torch.eye(len(vocab))[torch.tensor([next_word_idx])].unsqueeze(0)

print(generated_text)
