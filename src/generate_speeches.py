import argparse
import json
import os
import time
import uuid

import pandas as pd
from model import OpenAIModel
from util import setup_default_logger, read_annotation_data, word_tokenize_text, sent_tokenize_text


def main():
    speeches_data = read_annotation_data(config['data_path'])

    human_speeches = speeches_data[speeches_data['source'] == 'Human expert']
    nr_speeches_per_topic = human_speeches['topic'].value_counts().to_dict()

    logger.info(f'Number of unique topics: {len(nr_speeches_per_topic)}')

    api_key = open(config['openai_key_path']).read().strip()

    model = OpenAIModel(api_key, config['openai_model'])

    prompt_template = open(config['prompt_path']).read()
    prompts = {}
    promts_topics = {}


    for topic, nr_speeches_per_topic in nr_speeches_per_topic.items():
        for _ in range(nr_speeches_per_topic):
            prompt = prompt_template.replace('{{TOPIC}}', topic)
            prompt_id = str(uuid.uuid4())
            prompts[prompt_id] = prompt
            promts_topics[prompt_id] = topic

    batch_id = model.request_batch_completions(prompts, max_tokens=config['max_tokens'],
                                               temperature=config['temperature'], top_logprobs=20, batch_idx=0,
                                               output_path=config['output_path'])

    logger.info(f'Created batch of {len(prompts)} with ID: {batch_id}')

    prompts_info_path = os.path.join(config['output_path'], 'prompts_info.json')
    with open(prompts_info_path, 'w') as f:
        json.dump(prompts, f)
    logger.info(f'Prompts information saved to {prompts_info_path}')

    prompts_topics_path = os.path.join(config['output_path'], 'prompts_topics.json')
    with open(prompts_topics_path, 'w') as f:
        json.dump(promts_topics, f)
    logger.info(f'Prompts topics saved to {prompts_topics_path}')

    logger.info(f'Waiting for batch {batch_id} to complete...')
    responses = model.get_batch_completions(batch_id)
    while not responses:
        logger.info(f'Waiting for batch {batch_id} to complete...')
        time.sleep(60)  # Wait for 1 minute
        responses = model.get_batch_completions(batch_id)

    logger.info(f'Batch {batch_id} completed')

    responses_path = os.path.join(config['output_path'], 'responses.json')
    with open(responses_path, 'w') as f:
        json.dump(responses, f)
    logger.info(f'Responses saved to {responses_path}')

    prompts_topics_path = os.path.join(config['output_path'], 'prompts_topics.json')
    with open(prompts_topics_path, 'r') as f:
        promts_topics = json.load(f)
    logger.info(f'Prompts topics loaded from {prompts_topics_path}')

    parsed_responses = []
    avg_speech_length = 0
    for key, value in responses.items():
        value = value['completion']
        answer_start = value.find("<speech>")
        answer_end = value.find("</speech>")
        if answer_start != -1 and answer_end != -1:
            speech = value[answer_start + len("<speech>"):answer_end].strip()
            speech_words = word_tokenize_text(speech)
            speech_length = len(speech_words)
            avg_speech_length += speech_length
            while len(speech_words) > config['maximum_nr_words']:
                logger.warning(f'Speech {key} too long: {len(speech_words)}, removing last sentence...')
                speech_paragraph = sent_tokenize_text(speech)
                sentence_to_remove = speech_paragraph.pop()
                sentence_start = speech.rfind(sentence_to_remove)
                speech = speech[:sentence_start].strip()
                speech_words = word_tokenize_text(speech)
                logger.info(f'Reduced speech to {len(speech_words)} words')
            parsed_responses.append(
                {'id': key, 'topic': promts_topics[key], 'source': config['openai_model'], 'text': speech}
            )
        else:
            logger.warning(f'Failed to parse response: [{value}]')
            parsed_responses.append(
                {'id': key, 'topic': promts_topics[key], 'source': config['openai_model'], 'text': value})

    results = pd.DataFrame(parsed_responses)
    for column_name in speeches_data.columns.tolist():
        if column_name not in results.columns.tolist():
            results[column_name] = None

    results_path = os.path.join(config['output_path'], 'results.csv')
    results.to_csv(results_path, index=False)
    logger.info(f'Results saved to {results_path}')
    logger.info(f'Average speech length: {avg_speech_length / len(parsed_responses)} words')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)

    args = parser.parse_args()
    config = json.load(open(args.config))
    logger = setup_default_logger(config['output_path'])

    main()
