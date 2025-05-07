import argparse
import json
import os
import uuid

import pandas as pd
import statistics
from util import setup_default_logger, read_annotation_data, get_unique_values, compute_kappa_agreement, \
    agg_model_predictions, get_model_size, get_model_family


def get_annotations_per_human_annotators_pair(data: pd.DataFrame, col_name: str):
    all_annotators = set()
    for _, row in data.iterrows():
        all_annotators.update(row['labeler_ids'])

    annotators_pairs = []
    for annotator1 in all_annotators:
        for annotator2 in all_annotators:
            if annotator1 != annotator2:
                pair = [annotator1, annotator2]
                pair.sort()
                pair = tuple(pair)

                if pair not in annotators_pairs:
                    annotators_pairs.append(pair)

    annotations_per_pair = {}
    for pair in annotators_pairs:
        for _, row in data.iterrows():
            if pair[0] in row['labeler_ids'] and pair[1] in row['labeler_ids']:
                if pair not in annotations_per_pair:
                    annotations_per_pair[pair] = {'annotator_1': [], 'annotator_2': [], 'speech_ids': []}
                labler_ids_to_index = {labler_id: i for i, labler_id in enumerate(row['labeler_ids'])}
                annotator_1_index = labler_ids_to_index[pair[0]]
                annotator_2_index = labler_ids_to_index[pair[1]]
                annotations_per_pair[pair]['annotator_1'].append(row[col_name][annotator_1_index])
                annotations_per_pair[pair]['annotator_2'].append(row[col_name][annotator_2_index])
                annotations_per_pair[pair]['speech_ids'].append(row['id'])

    return annotations_per_pair


def get_annotations_per_human_model_pair(data: pd.DataFrame, col_name: str, human_pairs_annotation: dict):
    annotations_per_pair = {}
    for human_annotators_pair, annotations in human_pairs_annotation.items():
        for annotator_id in human_annotators_pair:
            speech_ids = annotations['speech_ids']
            pair_key = (annotator_id, 'model_annotations', str(uuid.uuid4()))
            annotations_per_pair[pair_key] = {'annotator_1': [], 'annotator_2': []}
            for speech_id in speech_ids:
                row = data[data['id'] == speech_id].iloc[0]
                labeler_ids_to_index = {labeler_id: i for i, labeler_id in enumerate(row['labeler_ids'])}
                annotator_index = labeler_ids_to_index[annotator_id]
                human_annotation = row[col_name][annotator_index]
                model_annotation = row['model_annotations']
                annotations_per_pair[pair_key]['annotator_1'].append(human_annotation)
                annotations_per_pair[pair_key]['annotator_2'].append(model_annotation)

    return annotations_per_pair


def compute_human_model_agreement(data: pd.DataFrame, col_name: str, annotations_per_pair: dict):
    model_human_agreement = get_annotations_per_human_model_pair(data, col_name, annotations_per_pair)
    return compute_annotator_agreement(model_human_agreement)


def filter_annotations_by_min_support(annotations_per_pair: dict, min_annotations):
    filtered_annotations_per_pair = {}
    for pair, annotations in annotations_per_pair.items():
        if len(annotations['speech_ids']) >= min_annotations:
            filtered_annotations_per_pair[pair] = annotations

    return filtered_annotations_per_pair


def compute_annotator_agreement(agreement_data: dict):
    pair_agreements = {}
    avg_pair_support = 0
    nr_pairs = 0
    nr_undefined = 0
    undefined_value = 0
    for pair, annotations in agreement_data.items():
        kappa = compute_kappa_agreement(annotations['annotator_1'], annotations['annotator_2'],
                                        undefined_value=undefined_value)

        unique_values_1, unique_values_2 = get_unique_values(annotations['annotator_1'], annotations['annotator_2'])
        if len(unique_values_1) == 1 or len(unique_values_2) == 1:
            nr_undefined += 1

        pair_agreements[str(pair)] = {
            'Kappa': kappa,
            'support': len(annotations['annotator_1'])
        }
        avg_pair_support += len(annotations['annotator_1'])
        nr_pairs += 1

    avg_agreements = {'Kappa': 0}
    for pair in pair_agreements:
        avg_agreements['Kappa'] += pair_agreements[pair]['Kappa'] / nr_pairs

    median_agreements = {
        'Kappa': statistics.median([pair_agreements[pair]['Kappa'] for pair in pair_agreements]),
    }

    agreement_stats = {
        'avg_agreements': avg_agreements,
        'median_agreements': median_agreements,
        'num_pairs': nr_pairs,
        'avg_support': avg_pair_support / nr_pairs if nr_pairs > 0 else 0,
        'nr_undefined': nr_undefined
    }

    return agreement_stats


def compute_human_human_agreement(data: pd.DataFrame):
    human_agreement_results = []
    annotations_per_pair = get_annotations_per_human_annotators_pair(data, 'goodopeningspeech')
    annotations_per_pair = filter_annotations_by_min_support(annotations_per_pair, args.min_shared_annotations)
    if not annotations_per_pair:
        logger.warning('No pairs with sufficient support found, skipping human-human agreement analysis')
        return

    human_agreement_all = compute_annotator_agreement(annotations_per_pair)
    all_sources_human_agreement_results = {
        'model family': 'Human',
        'model size': get_model_size('human'),
        'model': 'Human',
        'prompt': 'human',
        'source': 'all_sources'
    }
    for key, value in human_agreement_all['avg_agreements'].items():
        all_sources_human_agreement_results[f'Avg {key}'] = value
    for key, value in human_agreement_all['median_agreements'].items():
        all_sources_human_agreement_results[f'Median {key}'] = value

    all_sources_human_agreement_results['support'] = human_agreement_all['avg_support']
    all_sources_human_agreement_results['nr_pairs'] = human_agreement_all['num_pairs']
    all_sources_human_agreement_results['nr_undefined'] = human_agreement_all['nr_undefined']
    all_sources_human_agreement_results['min_shared_support'] = args.min_shared_annotations
    all_sources_human_agreement_results['nr_source_speeches'] = len(data)

    human_agreement_results.append(all_sources_human_agreement_results)
    logger.info(f'\n---HUMAN-AGREEMENT-RESULTS---\n{json.dumps(human_agreement_results, indent=2)}')

    human_agreement_results = pd.DataFrame(human_agreement_results)
    human_agreement_results = human_agreement_results.round(3).sort_values(['model family', 'model size'])
    output_path = os.path.join(args.output_dir, 'human_agreement_results.csv')
    human_agreement_results.to_csv(output_path, index=False)
    logger.info(f'Saved human agreement results to {output_path}')


def analyse_agreement():
    data = read_annotation_data(args.data_path)
    compute_human_human_agreement(data)

    eval_models_config = json.load(open(args.eval_models_config))['eval_models']
    human_model_agreement_results = []
    for model_info in eval_models_config:
        logger.info(f'Evaluating model {model_info["name"]}')
        model_pred = {}
        prompt = args.prompt
        for model_name, res_dir in model_info['results'].items():
            res_path = os.path.join(res_dir, f'{prompt}.csv')
            if not os.path.exists(res_path):
                logger.warning(f'No results found at {res_path} for {model_name} with prompt {prompt}, skipping')
                break
            res = read_annotation_data(res_path)
            pred_col = [c for c in res.columns if c.endswith(f'{model_name}_{args.col_name}_{prompt}')]
            if len(pred_col) == 0:
                logger.warning(f'No prediction column found for model {model_name} in {res_path}, skipping')
                break
            for _, row in res.iterrows():
                if row['id'] not in model_pred:
                    model_pred[row['id']] = []
                model_pred[row['id']].append(row[pred_col[0]])

        if len(model_pred) == 0:
            logger.warning(f'No predictions found for model {model_info["name"]}, skipping')
            continue

        if len(model_info['results']) > 1:
            model_pred = agg_model_predictions(model_pred)
        else:
            model_pred = {'-': {row_id: row[0] for row_id, row in model_pred.items()}}

        for model_agg_func, agg_model_pred in model_pred.items():
            annotations_per_pair = get_annotations_per_human_annotators_pair(data, args.col_name)
            annotations_per_pair = filter_annotations_by_min_support(annotations_per_pair, args.min_shared_annotations)
            data['model_annotations'] = data['id'].map(agg_model_pred)
            human_model_agreement = compute_human_model_agreement(data, args.col_name, annotations_per_pair)

            agreement_stats = {
                'human_model_agreement': human_model_agreement,
            }

            logger.info(f'\n---AGREEMENT-STATS---\n{json.dumps(agreement_stats, indent=2)}')

            row = {
                'model family': get_model_family(model_info['name']),
                'model size': get_model_size(model_info['name']),
                'model_agg_func': model_agg_func,
                'model': model_info['name'],
                'prompt': prompt}
            for key, value in human_model_agreement['avg_agreements'].items():
                row[f'avg_{key}'] = value
            for key, value in human_model_agreement['median_agreements'].items():
                row[f'median_{key}'] = value
            for key, value in human_model_agreement.items():
                if key not in ['avg_agreements', 'median_agreements']:
                    row[key] = value

            row['support'] = human_model_agreement['avg_support']
            row['nr_pairs'] = human_model_agreement['num_pairs']
            row['nr_undefined'] = human_model_agreement['nr_undefined']
            row['min_shared_support'] = args.min_shared_annotations
            human_model_agreement_results.append(row)

    human_model_agreement_results = pd.DataFrame(human_model_agreement_results)
    human_model_agreement_results = human_model_agreement_results.round(3).sort_values(['model family', 'model size'])
    output_path = os.path.join(args.output_dir, 'human_model_agreement_results.csv')
    human_model_agreement_results.to_csv(output_path, index=False)
    logger.info(f'Saved human model agreement results to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_models_config', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--min_shared_annotations', type=int)
    parser.add_argument('--col_name', default='goodopeningspeech', type=str)
    parser.add_argument('--prompt', type=str)

    args = parser.parse_args()
    logger = setup_default_logger(args.output_dir)

    analyse_agreement()
