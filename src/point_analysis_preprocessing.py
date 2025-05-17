import argparse
import json
import os.path

import pandas as pd

from util import setup_default_logger, read_annotation_data
from model import OpenAIModel
from tqdm import tqdm


def get_reasoning(response: str) -> str:
    reasoning_start = response.find('<scratchpad>') + len('<scratchpad>')
    reasoning_end = response.find('</scratchpad>')
    if reasoning_start != -1 and reasoning_end != -1:
        reasoning = response[reasoning_start:reasoning_end].strip()
        return reasoning
    return ''


def main():
    logger.info(f'loading eval models config from {args.judges_results}')
    eval_models_config = json.load(open(args.judges_results))['eval_models']
    all_speeches = {}
    model_names = [model_info['name'] for model_info in eval_models_config]
    model_names = list(set(model_names))
    key = open(args.openai_api_key).read().strip()

    model = OpenAIModel(key, args.model_name)
    prompt_template = open(args.preprocessing_prompt).read().strip()

    for model_info in eval_models_config:
        if len(model_info['results']) > 1:
            logger.warning(f"Multiple result files found for {model_info['model_name']}. skipping")
            continue

        for model_name, res_dir in model_info['results'].items():
            results_dir_path = os.path.join(res_dir, f'{args.prompt}.csv')
            if not os.path.exists(results_dir_path):
                logger.warning(f'Results directory {results_dir_path} does not exist. Skipping model {model_name}.')
                break
            res = read_annotation_data(results_dir_path)

            pred_col_postfix = f'{model_name}_{args.col_name}_{args.prompt}_response'

            response_col = [c for c in res.columns if c.endswith(pred_col_postfix)]
            if len(response_col) == 0:
                logger.warning(
                    f'No predictions found for model {model_name} with postfix {pred_col_postfix}. Skipping.')
                continue
            if len(response_col) > 1:
                logger.warning(
                    f'Multiple response columns found for model {model_name} with postfix {pred_col_postfix}. Skipping.')
                continue

            response_col = response_col[0]

            pbar = tqdm(total=len(res), desc=f'Processing {model_name}...')
            for _, row in res.iterrows():
                speech_id = row['id']
                if speech_id not in all_speeches:
                    all_speeches[speech_id] = {'topic': row['topic'], 'source': row['source'], 'text': row['text']}
                    for model_name in model_names:
                        all_speeches[speech_id][f'{model_name}_reasoning'] = ''

                raw_reasoning = get_reasoning(row[response_col])
                prompt = prompt_template.replace('{PARAGRAPH}', raw_reasoning)
                raw_response = model.generate(prompt, args.max_tokens, args.temperature)
                response_start = raw_response['completion'].find('<result>')
                response_end = raw_response['completion'].find('</result>')
                if response_start != -1 and response_end != -1:
                    reasoning = raw_response['completion'][response_start + len("<result>"): response_end].strip()
                else:
                    reasoning = ''
                    logger.warning(f'No reasoning found in response for speech {speech_id}.')

                all_speeches[speech_id][f'{model_info["name"]}_reasoning'] = reasoning
                pbar.update(1)

    results = []
    comment_id = 0
    for model_name in model_names:
        for speech_id, speech_info in all_speeches.items():
            result = {
                'id': speech_id,
                'topic': speech_info['topic'],
                'source': speech_info['source'],
                'text': speech_info['text'],
                'model': model_name,
                'reasoning': speech_info[f'{model_name}_reasoning'],
                'comment_id': comment_id,
            }
            results.append(result)
            comment_id += 1
    results_df = pd.DataFrame(results)
    output_path = os.path.join(args.output_path, f'{args.prompt}_reasoning.csv')
    results_df.to_csv(output_path, index=False)
    logger.info(f'Results saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--judges_results', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--col_name', default='goodopeningspeech', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--max_tokens', default=1000, type=int)
    parser.add_argument('--openai_api_key', type=str)
    parser.add_argument('--preprocessing_prompt', type=str)

    args = parser.parse_args()
    logger = setup_default_logger(args.output_path)

    main()
