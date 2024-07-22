pip install nltk

cd dataset

wget https://data.together.xyz/redpajama-data-1T/v1.0.0/arxiv/arxiv_024de5df-1b7f-447c-8c3a-51407d8d6732.jsonl

mv arxiv_* arxiv.jsonl

cd ..

python tools/preprocess_data.py \
  --input dataset/arxiv.jsonl \
  --output-prefix dataset/arxiv \
  --dataset-impl mmap \
  --tokenizer-type GPT2BPETokenizer \
  --vocab-file dataset/gpt2-vocab.json \
  --merge-file dataset/gpt2-merges.txt \
  --workers 64 \
  --append-eod