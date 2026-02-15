import torch
from model_pack.transformer import Transformer
from model_utility.translator import beam_search
from torchtext.data import Field
import spacy
from indicnlp import common
from indicnlp.tokenize import indic_tokenize
import sys


# Settings
INDIC_NLP_LIB_HOME = r"indic_nlp_library"
INDIC_NLP_RESOURCES = r"indic_nlp_resources"
sys.path.append(r'{}/src'.format(INDIC_NLP_LIB_HOME))
common.set_resources_path(INDIC_NLP_RESOURCES)

spacy_eng = spacy.load("en_core_web_sm")


def tokenize_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)][:60]


def tokenize_hindi(text):
    return [tok for tok in indic_tokenize.trivial_tokenize(text)]


# Define Fields
english_txt = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
hindi_txt = Field(tokenize=tokenize_hindi, init_token="<sos>", eos_token="<eos>")


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading checkpoint...")
checkpoint = torch.load("my_checkpoint.pth.tar", map_location=device, weights_only=False)

print("Inference device:", device)

# Load vocab from checkpoint
english_txt.vocab = checkpoint["src_vocab"]
hindi_txt.vocab = checkpoint["trg_vocab"]

src_vocab_size = len(english_txt.vocab)
trg_vocab_size = len(hindi_txt.vocab)

print(f"Source vocab size: {src_vocab_size}")
print(f"Target vocab size: {trg_vocab_size}")

# Model parameters
embedding_size = 512
num_heads = 8
num_layers = 3
dropout = 0.10
max_len = 10000
forward_expansion = 4

src_pad_idx = english_txt.vocab.stoi["<pad>"]
trg_pad_idx = hindi_txt.vocab.stoi["<pad>"]

# Initialize model
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

model.load_state_dict(checkpoint["state_dict"])
model.eval()
torch.set_grad_enabled(False)

print("✓ Model loaded successfully!\n")

# GPU warmup
with torch.no_grad():
    _ = model(
        torch.ones((1,1), dtype=torch.long).to(device),
        torch.ones((1,1), dtype=torch.long).to(device)
    )

# Translate function
def translate(sentence):
    print(f"English: {sentence}")

    with torch.no_grad():
        translation = beam_search(
        sentence=sentence,
        model=model,
        src_field=english_txt,
        src_tokenizer=tokenize_eng,
        trg_field=hindi_txt,
        trg_vcb_sz=trg_vocab_size,
        k=5,
        max_ts=50,
        device=device
    )

    translation = translation.replace("<eos>", "").replace("<sos>", "").strip()
    print(f"Hindi: {translation}\n")

    return translation


# Test
print("=" * 60)
print("TRANSLATIONS")
print("=" * 60 + "\n")

# Basic conversational
translate("hello how are you")
translate("what is your name")
translate("where are you going today")
translate("i am learning machine learning")

# Daily life sentences
translate("please open the door")
translate("she is cooking food in the kitchen")
translate("we will meet tomorrow morning")
translate("he bought a new phone yesterday")

# Question forms
translate("why are you late")
translate("when does the train arrive")
translate("how can I help you")

# Medium complexity
translate("the weather is very pleasant today")
translate("education is important for a better future")
translate("technology is changing the world rapidly")

# Longer unseen sentence (generalization test)
translate("the government announced new policies to improve education in rural areas")

# Slightly complex grammar
translate("if you work hard you will achieve success")
translate("although it was raining they continued playing")


# Interactive mode
print("\nEnter sentences to translate (or 'quit' to exit):")
while True:
    sentence = input("\nEnglish: ").strip()
    if sentence.lower() in ['quit', 'exit', 'q']:
        break
    if sentence:
        translate(sentence)