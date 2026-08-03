# Training

Check your FFmpeg installation:
```bash
ffmpeg -version
```
If not found, install it first (or skip assuming you know of other backends available).

## Prepare Dataset

Example data processing scripts, and you may tailor your own one along with a Dataset class in `src/f5_tts/model/dataset.py`.

### 1. Some specific Datasets preparing scripts
Download corresponding dataset first, and fill in the path in scripts.

```bash
# Prepare the Emilia dataset
python src/f5_tts/train/datasets/prepare_emilia.py

# Prepare the Wenetspeech4TTS dataset
python src/f5_tts/train/datasets/prepare_wenetspeech4tts.py

# Prepare the LibriTTS dataset
python src/f5_tts/train/datasets/prepare_libritts.py

# Prepare the LJSpeech dataset
python src/f5_tts/train/datasets/prepare_ljspeech.py
```

### 2. Create custom dataset with CSV
Prepare a CSV with two columns using a required header: `audio_file|text`. Audio paths must be absolute.
Use guidance see [#57 here](https://github.com/SWivid/F5-TTS/discussions/57#discussioncomment-10959029).

```bash
python src/f5_tts/train/datasets/prepare_csv_wavs.py /path/to/metadata.csv /path/to/output
```

## Training & Finetuning

Once your datasets are prepared, you can start the training process.

### 1. Training script used for pretrained model

```bash
# setup accelerate config, e.g. use multi-gpu ddp, fp16
# will be to: ~/.cache/huggingface/accelerate/default_config.yaml     
accelerate config

# .yaml files are under src/f5_tts/configs directory
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml

# possible to overwrite accelerate and hydra config
accelerate launch --mixed_precision=fp16 src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml ++datasets.batch_size_per_gpu=19200
```

### 2. Finetuning practice
Discussion board for Finetuning [#57](https://github.com/SWivid/F5-TTS/discussions/57).

Gradio UI training/finetuning with `src/f5_tts/train/finetune_gradio.py` see [#143](https://github.com/SWivid/F5-TTS/discussions/143).

If want to finetune with a variant version e.g. *F5TTS_v1_Base_no_zero_init*, manually download pretrained checkpoint from model weight repository and fill in the path correspondingly on web interface.

If use tensorboard as logger, install it first with `pip install tensorboard`.

<ins>The `use_ema = True` might be harmful for early-stage finetuned checkpoints</ins> (which goes just few updates, thus ema weights still dominated by pretrained ones), try turn it off with finetune gradio option or `load_model(..., use_ema=False)`, see if offer better results.

### 3. W&B Logging

The `wandb/` dir will be created under path you run training/finetuning scripts.

By default, the training script does NOT use logging (assuming you didn't manually log in using `wandb login`).

To turn on wandb logging, you can either:

1. Manually login with `wandb login`: Learn more [here](https://docs.wandb.ai/ref/cli/wandb-login)
2. Automatically login programmatically by setting an environment variable: Get an API KEY at https://wandb.ai/authorize and set the environment variable as follows:

On Mac & Linux:

```
export WANDB_API_KEY=<YOUR WANDB API KEY>
```

On Windows:

```
set WANDB_API_KEY=<YOUR WANDB API KEY>
```
Moreover, if you couldn't access W&B and want to log metrics offline, you can set the environment variable as follows:

```
export WANDB_MODE=offline
```


### 4. MLflow tracking (optional)

The separate `finetune_mlflow.py` entry point adds local or remote MLflow tracking without requiring MLflow for the
normal training commands. Install the optional dependencies from the repository root:

```bash
pip install -e ".[tracking]"
```

Enable local tracking for the Tunisian configuration with:

```bash
accelerate launch src/f5_tts/train/finetune_mlflow.py --config-name F5TTS_v1_Base_TUN.yaml \
  mlflow.enabled=true ++ckpts.logger=null
```

The default tracking URI is `file:./mlruns`. Override `mlflow.tracking_uri` to use a remote tracking server. Training
loss, learning rate, gradient norm, throughput, epoch/cumulative time, ETA, system metrics, resolved configuration,
Git commit, and checkpoints are tracked. This entry point does not create a validation split, report validation loss,
or select a best checkpoint. Set `mlflow.log_checkpoint_artifacts=false` to record checkpoint paths and metadata
without copying the large checkpoint files into the MLflow artifact store.

Inference tracking is disabled by default and never selects examples from the training dataset. To enable it, create:

```text
data/Tn_inference/
|-- voice.wav
`-- test.txt
```

`voice.wav` is the fixed reference voice. Put one generation prompt on each non-empty line of `test.txt`; every line
creates one WAV and one mel-spectrogram at the configured interval. The reference audio is transcribed once by the
existing F5-TTS ASR helper. Then run:

```bash
accelerate launch src/f5_tts/train/finetune_mlflow.py --config-name F5TTS_v1_Base_TUN.yaml \
  mlflow.enabled=true mlflow.inference.enabled=true ++ckpts.logger=null
```

The interval defaults to `ckpts.save_per_updates` and can be changed with `mlflow.inference.every_updates=<N>`.
Missing inference files, inference/ASR errors, and all MLflow logging failures are warnings and do not stop training.

Launch the local UI from the same repository directory:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open `http://127.0.0.1:5000` to inspect metric curves, system metrics, run parameters, checkpoint metadata/artifacts,
and optional generated audio and mel-spectrogram artifacts.

#### Accelerate memory note

Running the script with `python` is a single-process run. `accelerate launch` follows your Accelerate configuration
and may start one DDP process per GPU, which needs additional per-GPU memory for gradient communication. To match the
single-process memory profile while still using the launcher, explicitly use one process:

```bash
accelerate launch --num_processes 1 src/f5_tts/train/finetune_mlflow.py --config-name F5TTS_v1_Base_TUN.yaml \
  mlflow.enabled=true ++ckpts.logger=null
```

 Point the mlflow run to the database 
```bash

accelerate launch --num_processes 1 src/f5_tts/train/finetune_mlflow.py \
  --config-name F5TTS_v1_Base_TUN.yaml \
  mlflow.enabled=true \
  mlflow.tracking_uri=http://127.0.0.1:5000 \
  mlflow.log_checkpoint_artifacts=false \
  ++ckpts.logger=null
```

The trainer enables DDP gradient views to reduce this multi-process overhead. When optional inference is enabled, it
runs only on the main rank, synchronizes the other ranks, and releases the temporary vocoder afterward.


