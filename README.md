# Meeting-Helper-Bot|会议助手
> Building LLM applications using BigDL-LLM： A Meeting Summary & Rag-based Chat Bot
Powered by BigDL, Llama, Whisper, Helsinki and LangChain


## Pipeline
![image](https://github.com/Lxt115/Meeting-Summary-Bot/assets/67227722/03ce0f86-e793-4723-adf9-088aae4c6efd)

## Demo
|[English](data/demo/demo2.mp4)|[Chinese](data/demo/demo1.mp4)|
|:-:|:-:|

## Getting Started 
1. Create Conda Environment
```
conda  create -n vchat python=3.9 -y
activate vchat
pip install -r ./requirements.txt
```
2. Install FFmpeg
```
conda install -c conda-forge ffmpeg -y
```
3. Download Model
```
python download.py
```
4. Optimaize LLM by BigDL-LLM
```
python LLM_low_bit_optimize.py
```
5. Run main.py
```
python main.py
```

## Acknowledge

This project is based on [BigDL](https://github.com/intel-analytics/BigDL), [VChat-BigDL](https://github.com/Kailuo-Lai/VChat-BigDL), [Whisper](https://github.com/openai/whisper), [Llama2](https://github.com/facebookresearch/llama),[LangChian](https://github.com/langchain-ai/langchain)
