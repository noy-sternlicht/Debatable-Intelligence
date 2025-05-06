[![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](https://put-here-your-paper.com)
[![Python Versions](https://img.shields.io/badge/Python-3.11-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)


## LLM on Trial: Benchmarking LLM-as-a-Judge via Argumentation

As LLM judges grow in popularity, evaluating their performance on cognitively challenging tasks becomes crucial. We propose using debate speech evaluation as a new benchmarking task for LLM judges. To support this, we present a unique dataset of 631 debate speeches with careful annotations from multiple human raters. Through this dataset, we examine how well current state-of-the-art models perform on this complex task.
<p align="center">
  <img src="fig_1.svg" alt="Centered Image" width="300" />
</p>

### Getting started

A quick start guide to get you up and running with the code.

1. Clone this repository:
    ```bash
    git clone https://github.com/noy-sternlicht/BenchmarkingLLMAJ.git
    ````
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv myenv
   source ./myenv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Setting up API keys

Some of the code requires access to external APIs. You will need to set an OpenAI API key, an Anthropic key and a
HuggingFace API key as
follows:

1. Get the keys (if you don't have them already):
    * [OpenAI API key](https://platform.openai.com/docs/api-reference/authentication)
    * [Anthropic API key](https://docs.anthropic.com/en/api/admin-api/apikeys/get-api-key)
    * [HuggingFace API key](https://huggingface.co/docs/hub/security-tokens)
2. Set up the keys:
   ```bash
   mkdir secret_keys
   echo "your-openai-api-key" > secret_keys/openai_key
   echo "your-anthropic-api-key" > secret_keys/anthropic_key
   echo "your-huggingface-api-key" > secret_keys/huggingface_key
   ```
   The files specified in the above snippet (`secret_keys/***_key`) are the default locations where the code will look
   for the keys.

### Benchmark data

We make the benchmark data available at `data.csv`. The file contains the following fields:

* `id`: The unique identifier for the speech.
* `topic_id`: The unique identifier for the topic.
* `topic`: The topic of the debate speech (e.g., "Community service should be mandatory").
* `source`: The speech source (e.g., "Human-expert" for human authored speeches).
* `text`: The text of the speech.
* `goodopeningspeech`: A list of human rating for the speech (a number between 1 and 5 for each annotator).
* `#labelers`: The number of human annotators who rated the speech.
* `labeler_ids`: A list of the unique identifiers for the human annotators who rated the speech.

### Reproducing paper results

1. **Run judges**: We use `scripts/run_judge_models.sh` to run a judge (or multiple judges) over the data. The script
   receives a config file defining which models to run, what prompt to use, and so on. We provide an example at
   `src/zero_shot_config.json`. You can modify it as follows:
   ```text
     {"data_path": "data.csv",  # Path to the human annotated speeches
     "output_path": "output/judges_results/", # Where to save the results
     "openai_key_path": "secret_keys/openai_key", # Default keys location
     "anthropic_key_path": "secret_keys/anthropic_key",
     "huggingface_key_path": "secret_keys/huggingface_key",
     "max_tokens": 4096,  # Max input + output tokens
     "temperature": 0.01,  # Judge temperature
     "experiments": [
       {
         "name": "zero-shot-good-speech-guidelines",
         "prompt_path": "./prompts/zero_shot_good_speech_annotation_guidelines.txt",
         "human_annotations_data_field": "goodopeningspeech",
         "run": true
       },
       {
         "name": "zero-shot-good-speech-guidelines-short-cot", 
         "prompt_path": "./prompts/zero_shot_good_speech_annotations_guidelines_short_cot.txt",
         "human_annotations_data_field": "goodopeningspeech",
         "run": true
       }
     ],
     "models": { # Which models to run (divided by required key). For example, listing "openai" under "participating_models" will run the specified openai models.
       "openai": [
         "o3"
       ],
       "huggingface": [
         "Qwen/Qwen2.5-72B-Instruct"
       ],
       "anthropic": [
         "claude-3-7-sonnet-20250219"
       ]
     },
     "participating_models": [
       "anthropic"
     ]}
   ```

### Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{author2025paper,
  title={Paper Title},
  author={Last, First and Coauthor, Another},
  journal={Journal Name},
  volume={X},
  number={Y},
  pages={ZZ--ZZ},
  year={2025},
  publisher={Publisher}
}
```

### Authors

* [Noy Sternlicht](https://x.com/NoySternlicht)
* [Ariel Gera](todo)
* [Roy Bar-Haim](todo)
* [Tom Hope](https://tomhoper.github.io/)
* [Noam Slonim](todo)