cd llama-recipes
pip install -U pip setuptools
pip install -e .

pip install protobuf
pip freeze | grep transformers ## verify it is version 4.31.0 or higher
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
   --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path

# clone the dataset
git clone https://huggingface.co/datasets/TencentARC/Plot2Code recipes/finetuning/datasets/data