# Guinea

## Setup

Set up conda environment:
```bash
conda env create -f environment.yaml
```
Active your environment
```bash
conda activate guinea 
```

Set up Open AI API key:
```bash
touch .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=your_api_key_here
```

Set up dataset:
```bash
# download dataset from Kaggle
curl -L -o dataset/arxiv-paper-abstracts.zip \
  https://www.kaggle.com/api/v1/datasets/download/spsayakpaul/arxiv-paper-abstracts

# unzip the dataset
cd dataset
unzip arxiv-paper-abstracts.zip
```
