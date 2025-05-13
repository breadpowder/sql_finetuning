#/bin/bash
conda create --name unsloth_env python=3.10.17 
conda activate unsloth_env
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu126
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl==0.15.2 peft==0.15.2 accelerate==1.6.0 bitsandbytes==0.45.5
pip install transformers==4.51.3
conda install ipykernel
python -m ipykernel install --user --name=unsloth_env

### Below is for model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method= 'f16')
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
sudo apt update
sudo apt install libcurl4-openssl-dev

mkdir build
cd build
cmake ..
cmake --build . --config Release
ln -s build/bin/llama-quantize ./llama-quantize
