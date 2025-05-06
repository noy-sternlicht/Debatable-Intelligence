import json
import os
import time
from abc import ABC, abstractmethod
from typing import Dict
import tiktoken
from tqdm import tqdm

import pandas as pd
from openai import OpenAI, ChatCompletion
import anthropic

from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch


class LanguageModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, str]:
        pass

    def request_batch_completions(self, prompts: Dict[str, str], max_tokens: int, temperature: float, batch_idx: int,
                                  output_path: str) -> str:
        print(f'Processing {len(prompts)} prompts for batch {batch_idx}...')
        completions = {}
        pbar = tqdm(total=len(prompts), desc=f'Batch {batch_idx}...')
        for prompt_id, prompt in prompts.items():
            completion = self.generate(prompt, max_tokens, temperature)
            completions[prompt_id] = completion
            pbar.update(1)

        batch_file = os.path.join(output_path, f'batch_{batch_idx}_completions.json')
        with open(batch_file, 'w') as f:
            json.dump(completions, f)

        return batch_file

    def get_batch_completions(self, batch_id: str) -> Dict[str, Dict[str, str]]:
        with open(batch_id, 'r') as f:
            completions = json.load(f)
        return completions


class OpenAIModel(LanguageModel):
    def __init__(self, key: str, engine: str):
        self.engine = engine
        self.client = OpenAI(api_key=key)

    def request_batch_completions(self, prompts: Dict[str, str], max_tokens: int, temperature: float, batch_idx: int,
                                  output_path: str) -> str:
        batch_requests = []
        for prompt_id, prompt in prompts.items():
            batch_entry = {"custom_id": prompt_id, "method": "POST",
                           "url": "/v1/chat/completions",
                           "body": {"model": self.engine, "max_tokens": max_tokens, "temperature": temperature,
                                    "messages": [{"role": "user", "content": prompt}]}}
            batch_requests.append(batch_entry)

        batch_requests = pd.DataFrame(batch_requests)
        batch_requests_file = os.path.join(output_path, f'batch_{batch_idx}_requests.json')
        batch_requests.to_json(batch_requests_file, lines=True, orient='records')

        batch_input_file = self.client.files.create(
            file=open(batch_requests_file, "rb"),
            purpose="batch"
        )

        batch_out = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f'Batch {batch_idx} for {self.engine}',
            }
        )

        return batch_out.id

    def get_batch_completions(self, batch_id: str) -> Dict[str, Dict[str, str]]:
        batch_status = self.client.batches.retrieve(batch_id)
        print(batch_status.status)
        query_responses_by_id = {}
        if batch_status.status == "completed":
            batch_response = self.client.files.content(batch_status.output_file_id).text
            query_responses = [json.loads(r) for r in batch_response.strip().split('\n')]
            query_responses_by_id = {}
            for response in query_responses:
                response_content = response['response']['body']['choices'][0]['message']['content']
                result = {'completion': response_content}
                query_responses_by_id[response['custom_id']] = result
        elif batch_status.status == "failed":
            raise Exception(f'Batch {batch_id} failed')

        return query_responses_by_id

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, str]:
        completion = self.client.chat.completions.create(
            model=self.engine,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        result = {'completion': completion.choices[0].message.content}
        return result


class OpenAIReasoningModel(LanguageModel):
    def __init__(self, key: str, engine: str):
        self.engine = engine
        self.client = OpenAI(api_key=key)

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, str]:
        encoding = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = len(encoding.encode(prompt))

        max_completion_tokens = max(0, max_tokens - prompt_tokens)

        completion_args = {
            "model": self.engine,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_completion_tokens,
        }

        completion = self.client.chat.completions.create(**completion_args)

        result = {'completion': completion.choices[0].message.content}
        return result


class HuggingFaceModel(LanguageModel):
    def __init__(self, key: str, model_name: str):
        model_size = float(model_name.split('B')[0].split('-')[-1])
        print(f'Model size: {model_size}')
        model_args = {}
        if model_size > 10:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, "float16"),
                bnb_4bit_use_double_quant=True
            )

            model_args["quantization_config"] = quant_config

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_args, token=key).to(torch.device("cuda"))
        model = model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=key)

        self.model = pipeline(task="text-generation",
                              model=model,
                              tokenizer=tokenizer,
                              torch_dtype=torch.float16,
                              token=key)



    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, str]:
        messages = [{'role': 'user', 'content': prompt}]
        with torch.no_grad():
            completion = self.model(messages, max_length=max_tokens, temperature=temperature)
        response = completion[0]['generated_text'][-1]['content']
        result = {'completion': response}
        return result

class AnthropicModel(LanguageModel):
    def __init__(self, key: str, engine: str):
        self.engine = engine
        self.client = anthropic.Client(api_key=key)

    def simple_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        completion = self.client.messages.create(
            model=self.engine,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        response = completion.content[0].text
        return response

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, str]:

        retries = 2
        response = ''
        while retries > 0:
            try:
                response = self.simple_generate(prompt, max_tokens, temperature)
                break
            except Exception as e:
                print(f'Error in Anthropic: {e}')
                retries -= 1
                time.sleep(5)

        result = {'completion': response}

        return result
