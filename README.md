<p align="center">
  <img src="logo.svg" alt="Centered Image" width="250" />
</p>
<p align="center">
    <a href="XXX" target="_blank">Website</a> | <a href="xxx" target="_blank">Paper</a> | <a href="xxx" target="_blank">Data</a> <br>
</p>

---

## LLM on Trial: Benchmarking LLM-as-a-Judge via Argumentation
TODO: add a general description of the work

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

Some of the code requires access to external APIs. You will need to set an OpenAI API key, an Anthropic key and a HuggingFace API key as
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
   The files specified in the above snippet (`secret_keys/***_key`) are the default locations where the code will look for the keys.

### Reproducing paper results
TODO

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