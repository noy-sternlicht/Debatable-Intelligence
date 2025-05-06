import argparse
import os
import time
from typing import Dict
from model import LanguageModel, OpenAIModel, HuggingFaceModel, AnthropicModel, OpenAIReasoningModel
from util import setup_default_logger, read_annotation_data
import json


def parse_response(response: str) -> int:
    if '</scratchpad>' in response:
        response = response.split('</scratchpad>')[1].strip()

    response_start = response.find('<score>') + len('<score>')
    response_end = response.find('</score>')
    if response_start != -1 and response_end != -1:
        parsed = response[response_start:response_end]
    else:
        parsed = response.replace('<score>', '').replace('</score>', '')
        parsed = parsed.replace('score', '').replace('Score', '')
        parsed = parsed.replace(':', '')
    parsed = parsed.strip()
    if parsed in ['1', '2', '3', '4', '5']:
        return int(parsed)

    logger.error(f'Parsing failure response=[{response}], parsed=[{parsed}]')
    return -1


def get_reasoning(response: str) -> str:
    reasoning_start = response.find('<scratchpad>') + len('<scratchpad>')
    reasoning_end = response.find('</scratchpad>')
    if reasoning_start != -1 and reasoning_end != -1:
        return response[reasoning_start:reasoning_end].strip()
    return ''


def setup_models() -> Dict[str, LanguageModel]:
    models = {}
    participating_models = config['participating_models']
    available_models = config['models']
    if 'openai' in participating_models:
        with open(config['openai_key_path'], 'r') as f:
            api_key = f.read()
        assert api_key, 'OpenAI API key is empty'
        for engine in available_models['openai']:
            if engine.startswith('gpt-4'):
                logger.info(f'Setting up OpenAI model with engine: {engine}')
                models[f'openai_{engine}'] = OpenAIModel(api_key, engine)
            else:
                logger.info(f'Setting up OpenAI reasoning model with engine: {engine}')
                models[f'openai_{engine}'] = OpenAIReasoningModel(api_key, engine)
    if 'huggingface' in participating_models:
        with open(config['huggingface_key_path'], 'r') as f:
            api_key = f.read()
        assert api_key, 'HuggingFace API key is empty'
        for model_name in available_models['huggingface']:
            logger.info(f'Setting up HuggingFace model with model name: {model_name}')
            models[f'huggingface_{model_name}'] = HuggingFaceModel(api_key, model_name)
    if 'anthropic' in participating_models:
        with open(config['anthropic_key_path'], 'r') as f:
            api_key = f.read()
        assert api_key, 'Anthropic API key is empty'
        for model_name in available_models['anthropic']:
            logger.info(f'Setting up Anthropic model with model name: {model_name}')
            models[f'anthropic_{model_name}'] = AnthropicModel(api_key, model_name)

    return models


def zero_shot_experiment():
    annotations = read_annotation_data(config['data_path'])
    logger.info(f'Loaded {len(annotations)} annotations from {config["data_path"]}')
    experiments_config = config['experiments']
    max_tokens = int(config['max_tokens'])
    temperature = float(config['temperature'])
    models = setup_models()

    for experiment in experiments_config:
        if experiment['run'] is False:
            continue

        logger.info(f'Running zero-shot experiment: {experiment["name"]}')
        with open(experiment['prompt_path'], 'r') as f:
            prompt_template = f.read()

        experiment_results = {}
        for model_name, model in models.items():
            logger.info(f'Running zero-shot experiment for model: {model_name}')
            prompts = {}
            for _, row in annotations.iterrows():
                prompt = prompt_template.replace('{TOPIC}', row['topic'])
                prompt = prompt.replace('{SPEECH}', row['text'])
                prompts[row['id']] = prompt

            batch_idx = model.request_batch_completions(prompts, max_tokens, temperature, 0, config['output_path'])

            logger.info(f'Waiting for batch {batch_idx} to complete...')
            responses = model.get_batch_completions(batch_idx)
            while not responses:
                time.sleep(60)  # Wait for 1 minute
                responses = model.get_batch_completions(batch_idx)
                logger.info(f'Waiting for batch {batch_idx} to complete...')

            experiment_results[model_name] = {}
            experiment_name = experiment["name"]
            annotations_col = experiment['human_annotations_data_field']

            responses_col_name = f'{model_name}_{annotations_col}_{experiment_name}_response'
            annotations_col_name = f'{model_name}_{annotations_col}_{experiment_name}'
            annotations[responses_col_name] = annotations['id'].apply(lambda x: responses[x]['completion'])
            annotations[annotations_col_name] = annotations[responses_col_name].apply(parse_response)
            for idx, row in annotations.iterrows():
                log_string = '\n==================================\n'
                log_string += f'----------TOPIC----------\n{row["topic"]}'
                log_string += f'\n----------TEXT----------\n{row["text"]}'
                log_string += '\n---------------------------------\n'
                pred = row[annotations_col_name]
                log_string += f'MODEL-[{model_name}]: {pred}'

                full_response = row[responses_col_name]
                reasoning = get_reasoning(full_response)
                if reasoning:
                    log_string += f'\n----------REASONING----------\n{reasoning}'
                logger.info(log_string)

        output_path = os.path.join(config['output_path'], f'{experiment["name"]}.csv')
        annotations.to_csv(output_path, index=False)
        logger.info(f'Saved predictions to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    config = json.load(open(args.config))
    logger = setup_default_logger(config['output_path'])

    zero_shot_experiment()
