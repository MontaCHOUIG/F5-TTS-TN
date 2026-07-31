"""
extend_habibi_vocab.py

Extends the Habibi-TTS (F5-TTS MSA base model) vocabulary and the
corresponding text-embedding matrix inside the checkpoint, so the model
can represent tokens that exist in your fine-tuning dataset but were
absent from the original MSA vocab.

Typical use (based on your error) -- no need to pass any paths, the
cached MSA vocab.txt and model_200000.safetensors (the same ones
populated by download_habibi_msa.py) are extended and overwritten
in place, in the same HF cache location:

    python extend_habibi_vocab.py --symbols "ّ,ڨ,ہ"

Note: this overwrites files inside your local Hugging Face cache
(~/.cache/huggingface/hub/...). That's a bit unusual -- normally cache
files are left untouched and treated as read-only, immutable snapshots
of what's on the Hub. If you later call hf_hub_download for this same
file again, huggingface_hub may re-verify/redownload it and clobber
your edit, since the cache no longer matches the recorded hash. If you
want your extended checkpoint to survive that, back it up outside the
cache dir once you're happy with it.

After this finishes, point finetune_cli.py at:
  - the (now extended, in-place) vocab file
  - the (now extended, in-place) checkpoint
"""

import argparse
import os
import random

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

# Where the base MSA checkpoint/vocab live on the Hub -- hf_hub_download
# will resolve these straight from the local cache if already downloaded,
# without re-downloading anything.
REPO_ID = "SWivid/Habibi-TTS"
SUBFOLDER = "Specialized/MSA"
CKPT_FILENAME = "model_200000.safetensors"
VOCAB_FILENAME = "vocab.txt"


def set_seed(seed=666):
    """Same fixed seed as the original finetune_gradio.py so the newly
    initialized rows for the new tokens are reproducible run-to-run."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(ckpt_path):
    """Load either a raw .safetensors state dict or a full .pt training
    checkpoint, and normalize both into a dict with an
    'ema_model_state_dict' key -- mirroring how the original script
    handles HF-hub safetensors files."""
    if ckpt_path.endswith(".safetensors"):
        # safetensors mmaps the file; clone every tensor into fresh, owned
        # memory so nothing is still referencing the file's bytes once we
        # overwrite it in place below.
        sd = {k: v.clone() for k, v in load_file(ckpt_path, device="cpu").items()}
        return {"ema_model_state_dict": sd}
    elif ckpt_path.endswith(".pt"):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "ema_model_state_dict" not in ckpt:
            # some .pt files are just a bare state dict
            ckpt = {"ema_model_state_dict": ckpt}
        return ckpt
    else:
        raise ValueError("Checkpoint must end in .safetensors or .pt")


def save_checkpoint(ckpt, ckpt_out):
    out_dir = os.path.dirname(ckpt_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if ckpt_out.endswith(".safetensors"):
        save_file(ckpt["ema_model_state_dict"], ckpt_out)
    elif ckpt_out.endswith(".pt"):
        torch.save(ckpt, ckpt_out)
    else:
        raise ValueError("Output checkpoint must end in .safetensors or .pt")


# Key F5-TTS/E2-TTS checkpoints use for the text-embedding weight matrix.
EMBED_KEY = "ema_model.transformer.text_embed.text_embed.weight"


def expand_embeddings(old_embeddings, num_new_tokens):
    """Keep every existing row untouched (so all previously-trained
    tokens keep their meaning) and append freshly randn-initialized
    rows for the new tokens, which get learned during fine-tuning."""
    vocab_old, embed_dim = old_embeddings.shape
    vocab_new = vocab_old + num_new_tokens
    new_embeddings = torch.zeros((vocab_new, embed_dim))
    new_embeddings[:vocab_old] = old_embeddings
    new_embeddings[vocab_old:] = torch.randn((num_new_tokens, embed_dim))
    return new_embeddings


def expand_model_embeddings(ckpt, num_new_tokens):
    sd = ckpt["ema_model_state_dict"]
    if EMBED_KEY not in sd:
        candidates = [k for k in sd if "text_embed" in k]
        raise KeyError(
            f"Expected key '{EMBED_KEY}' not found in checkpoint. "
            f"Keys containing 'text_embed': {candidates}"
        )
    sd[EMBED_KEY] = expand_embeddings(sd[EMBED_KEY], num_new_tokens)
    return sd[EMBED_KEY].shape[0]


def extend_vocab(vocab_in, vocab_out, new_symbols):
    with open(vocab_in, "r", encoding="utf-8-sig") as f:
        lines = f.read().split("\n")

    # drop trailing blank line(s) so we can cleanly re-append later
    while lines and lines[-1] == "":
        lines.pop()

    vocab_set = set(lines)
    old_size = len(lines)

    added = []
    for sym in new_symbols:
        sym = sym.strip()
        if sym == "" or sym in vocab_set:
            continue
        lines.append(sym)
        vocab_set.add(sym)
        added.append(sym)

    lines.append("")  # keep the trailing-blank-line convention

    out_dir = os.path.dirname(vocab_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(vocab_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return old_size, added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated missing symbols, e.g. \"ّ,ڨ\" (order = the order token ids get assigned)",
    )
    args = parser.parse_args()

    set_seed(666)

    # Resolved from the local HF cache -- no re-download if already present.
    # We overwrite these exact paths in place, so no separate output copy.
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename=f"{SUBFOLDER}/{VOCAB_FILENAME}")
    ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=f"{SUBFOLDER}/{CKPT_FILENAME}")

    new_symbols = args.symbols.split(",")

    # Load + clone the checkpoint into memory BEFORE touching the vocab file
    # or the checkpoint file on disk, so we're never reading and writing the
    # same file at once.
    ckpt = load_checkpoint(ckpt_path)

    old_size, added = extend_vocab(vocab_path, vocab_path, new_symbols)

    if not added:
        print("All given symbols are already in the vocab -- nothing to extend.")
        return

    new_size = expand_model_embeddings(ckpt, num_new_tokens=len(added))
    save_checkpoint(ckpt, ckpt_path)

    print(f"vocab old size : {old_size}")
    print(f"vocab new size : {new_size}")
    print(f"symbols added  : {len(added)}")
    print("new symbols    :")
    for s in added:
        print(f"  {s!r}  (token id {old_size + added.index(s)})")
    print(f"\nOverwrote vocab      in place -> {vocab_path}")
    print(f"Overwrote checkpoint in place -> {ckpt_path}")


if __name__ == "__main__":
    main()