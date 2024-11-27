## Finetuning Mochi with LoRA on Modal

This example demonstrates how to run the Mochi finetuner on Modal GPUs.

### Setup
Install [Modal](https://modal.com/docs/guide).
```bash
pip install modal
modal setup
```

### Fetch the dataset
There is a labeled dataset for a dissolving visual effect available on Google Drive. Download it into the `mochi-tune-videos` modal volume with:
```bash
modal run main::download_videos
```

### Download the model weights
Download the model weights from Hugging Face into the `mochi-tune-weights` modal volume with:
```bash
modal run -d main::download_weights
```
Note that this download can take more than 30 minutes. The `-d` flag allows you to exit the terminal session without losing progress.

### Prepare the dataset
We now run the preprocessing script to prepare the dataset for finetuning:
```bash
modal run main::preprocess
```
This puts preprocessed training input into the `mochi-tune-videos-prepared` modal volume.

### Finetuning
Finetune the model using the prepared dataset.

You may configure the finetune run using the `lora.yaml` file, such as number of steps, learning rate, etc.

Run the finetuning with:
```bash
modal run -d main::finetune
```

This will produce a series of checkpoints, as well as video samples generated along the training process. You can view these files in the Modal `moshi-tune-finetunes` volume using the Storage tab in the dashboard.

### Inference
You can now use the MochiLora class to generate videos from a prompt. The `main` entrypoint will initialize the model to use the specified LoRA weights from your finetuning run. 

```bash
modal run main
```
or with more parameters: 
```bash
modal run main lora-path="/finetunes/my_mochi_lora/model_1000.lora.safetensors" prompt="A pristine snowglobe featuring a winter scene sits peacefully. The glass begins to crumble into fine powder, as the entire sphere deteriorates into sparkling dust that drifts outward." 
```

See modal run main --help for all inference options.