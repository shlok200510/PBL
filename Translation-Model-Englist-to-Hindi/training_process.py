# Basic packages
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Packages for data generator & preparation
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
import sys
from indicnlp import common
from indicnlp.tokenize import indic_tokenize

# Settings for handling devnagri text
INDIC_NLP_LIB_HOME = r"indic_nlp_library"
INDIC_NLP_RESOURCES = r"indic_nlp_resources"
sys.path.append(r'{}/src'.format(INDIC_NLP_LIB_HOME))
common.set_resources_path(INDIC_NLP_RESOURCES)

# Packages for model building & inferences
from model_pack.transformer import Transformer
from model_utility.translator import beam_search
from model_utility.utils import save_checkpoint

# Data Prep
spacy_eng = spacy.load("en_core_web_sm")


def tokenize_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]


def tokenize_hindi(text):
    return [tok for tok in indic_tokenize.trivial_tokenize(text)]


# Defining Field
english_txt = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
hindi_txt = Field(tokenize=tokenize_hindi, init_token="<sos>", eos_token="<eos>")

# Defining Tabular Dataset
data_fields = [('eng_text', english_txt), ('hindi_text', hindi_txt)]
train_dt, val_dt = TabularDataset.splits(
    path='./',
    train='train_corpus.csv',
    validation='val_corpus.csv',
    format='csv',
    fields=data_fields,
    skip_header=True  # Add this to skip the header row!
)

# Building word vocab
english_txt.build_vocab(train_dt, max_size=10000, min_freq=2)
hindi_txt.build_vocab(train_dt, max_size=10000, min_freq=2)

print(f"Training examples: {len(train_dt)}")
print(f"Validation examples: {len(val_dt)}")
print(f"Source vocab size: {len(english_txt.vocab)}")
print(f"Target vocab size: {len(hindi_txt.vocab)}")

# Training & Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA Available:", torch.cuda.is_available())
print("Device being used:", device)

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# Speed optimization for GPU
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

save_model = True

# Training hyperparameters
num_epochs = 30
learning_rate = 3e-4
batch_size = 256

# Defining Iterator
train_iter = BucketIterator(
    train_dt,
    batch_size=batch_size,
    sort_key=lambda x: len(x.eng_text),
    shuffle=True,
    device=device   # ⬅ Important
)

val_iter = BucketIterator(
    val_dt,
    batch_size=batch_size,
    sort_key=lambda x: len(x.eng_text),
    shuffle=False,
    device=device   # ⬅ Important
)

# Model hyper-parameters
src_vocab_size = len(english_txt.vocab)
trg_vocab_size = len(hindi_txt.vocab)
embedding_size = 512
num_heads = 8
num_layers = 3
dropout = 0.10
max_len = 10000
forward_expansion = 4
src_pad_idx = english_txt.vocab.stoi["<pad>"]
trg_pad_idx = hindi_txt.vocab.stoi["<pad>"]

print("Source PAD index:", src_pad_idx)
print("Target PAD index:", trg_pad_idx)

# Defining model & optimizer
model = Transformer(
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    embed_size=embedding_size,
    num_layers=num_layers,
    forward_expansion=forward_expansion,
    heads=num_heads,
    dropout=dropout,
    device=device,
    max_len=max_len
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# FIXED: Remove verbose=True parameter
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

# Training loop
loss_tracker = []

for epoch in range(num_epochs):
    model.train()
    train_losses = []

    for train_batch_idx, train_batch in tqdm(enumerate(train_iter), total=len(train_iter),
                                             desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inp_data = train_batch.eng_text.permute(-1, -2)
        target = train_batch.hindi_text.permute(-1, -2)

        output = model(inp_data, target[:, :-1])
        output = output.reshape(-1, trg_vocab_size)
        target = target[:, 1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        train_losses.append(loss.item())

    train_mean_loss = sum(train_losses) / len(train_losses)

    # Validation
    model.eval()
    val_losses = []

    with torch.no_grad():
        for val_batch_idx, val_batch in tqdm(enumerate(val_iter), total=len(val_iter)):
            val_inp_data = val_batch.eng_text.permute(-1, -2).to(device)
            val_target = val_batch.hindi_text.permute(-1, -2).to(device)
            val_output = model(val_inp_data, val_target[:, :-1])
            val_loss = criterion(val_output.reshape(-1, trg_vocab_size), val_target[:, 1:].reshape(-1))
            val_losses.append(val_loss.item())

    val_mean_loss = sum(val_losses) / len(val_losses)
    loss_tracker.append(val_mean_loss)

    # Step the scheduler
    scheduler.step(val_mean_loss)

    # Save best model
    if epoch % 1 == 0:
        if save_model and val_mean_loss == np.min(loss_tracker):
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "src_vocab": english_txt.vocab,
                "trg_vocab": hindi_txt.vocab,
            }
            save_checkpoint(checkpoint)
            print(f"✓ Checkpoint saved at epoch {epoch + 1}")

    print(f"Epoch [{epoch + 1}/{num_epochs}]: train_loss= {train_mean_loss:.4f}; val_loss= {val_mean_loss:.4f}")

    # Clear unused GPU cache (helps long training stability)
    if device.type == "cuda":
        torch.cuda.empty_cache()

print("\nTraining completed!")

