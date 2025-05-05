!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt 
with open('input.txt', 'r',econding='utf-8') as f:
    text = f.read()
print ("length of the text is: ", len(text))
print(text[:1000])

#here are the unique set of characters that occur in the text 
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print("vocab size: ", vocab_size) 

#create a mapping from characters to integers and vice versa
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}
encode = lambda s: [char_to_int[c] for c in s]
decode = lambda l: "".join([int_to_char[i] for i in l])
print(encode("hello"))
print(decode(encode("hello")))

#let's encode the entire text and store it in torch.tensor
import torch
data = torch.tensor(encode(text), dtype=torch.long) #character level tokenization and storing it in torch.tensor
print(data[:1000])
print(data.shape,data.dtype)

#Let's split the data into train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
print(len(train_data), len(val_data))

block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target is {target}")

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel
block_size = 8 # how many characters to predict
def get_batch(split):
    # generate small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
xb, yb = get_batch('train')
print("inputs: ", xb.shape)
print(xb)
print("targets: ", yb.shape)
print(yb)

print('-------')

for b in range(batch_size) #batch dimension
    for t in range(block_size): #time dimension
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.to_list()} the target is {target}")

print (xb) # our input to the transformer in the neural network

# we will use a simple bigram language model as our baseline





