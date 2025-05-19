import argparse
import os

import pandas as pd
from util import setup_default_logger


def main():
    keypoints = pd.read_csv(args.keypoints)
    judge_results = pd.read_csv(args.preprocessed_judge_results)
    if args.judge:
        judge_results = judge_results[judge_results['model'] == args.judge]

    kp_stence = {}
    for _, row in keypoints.iterrows():
        if row['kp'] not in kp_stence:
            kp_stence[row['kp']] = 'con'
        if row['kp_stance'] == 'pro':
            kp_stence[row['kp']] = 'pro'

    keypoints_per_comment = {}
    for _, row in judge_results.iterrows():
        keypoints_per_comment[row['comment_id']] = keypoints[keypoints['comment_id'] == row['comment_id']][
            'kp'].values.tolist()

    judge_results['keypoints'] = judge_results['comment_id'].apply(
        lambda x: keypoints_per_comment[x] if x in keypoints_per_comment else [])


    keypoints_count = {}
    for _, row in judge_results.iterrows():
        for keypoint in row['keypoints']:
            if keypoint not in keypoints_count:
                keypoints_count[keypoint] = 0
            keypoints_count[keypoint] += 1
    keypoints_count_list = []
    total_keypoints = sum(keypoints_count.values())
    for keypoint, count in keypoints_count.items():
        keypoints_count_list.append({
            'keypoint': keypoint,
            'stance': kp_stence[keypoint],
            '#': count,
            '%': count / total_keypoints * 100,
        })


    keypoints_count_df = pd.DataFrame(keypoints_count_list)

    top_3_pros = keypoints_count_df[keypoints_count_df['stance'] == 'pro'].nlargest(3, '%')
    top_3_cons = keypoints_count_df[keypoints_count_df['stance'] == 'con'].nlargest(3, '%')

    keypoints_per_source = {}
    unique_sources = judge_results['source'].unique()
    for source in unique_sources:
        source_comments = judge_results[judge_results['source'] == source]
        keypoints_per_source[source] = {}
        for _, row in source_comments.iterrows():
            for keypoint in row['keypoints']:
                if keypoint in top_3_pros['keypoint'].values or keypoint in top_3_cons['keypoint'].values:
                    if keypoint not in keypoints_per_source[source]:
                        keypoints_per_source[source][keypoint] = 0
                    keypoints_per_source[source][keypoint] += 1
                else:
                    if kp_stence[keypoint] == 'pro':
                        if 'Other (pro)' not in keypoints_per_source[source]:
                            keypoints_per_source[source]['Other (pro)'] = 0
                        keypoints_per_source[source]['Other (pro)'] += 1
                    else:
                        if 'Other (con)' not in keypoints_per_source[source]:
                            keypoints_per_source[source]['Other (con)'] = 0
                        keypoints_per_source[source]['Other (con)'] += 1

    keypoints_per_source_list = []
    for source, keypoints in keypoints_per_source.items():
        nr_source_keypoints = sum(keypoints.values())
        for keypoint, count in keypoints.items():
            stance = 'pro'
            if (keypoint == 'Other (con)') or (keypoint != 'Other (pro)' and kp_stence[keypoint] == 'con'):
                stance = 'con'
            keypoints_per_source_list.append({
                'source': source,
                'keypoint': keypoint,
                'stance': stance,
                '#': count,
                '%': count / nr_source_keypoints * 100,
            })
    keypoints_per_source_df = pd.DataFrame(keypoints_per_source_list)

    unique_keypoints = keypoints_per_source_df['keypoint'].unique()
    keypoints_to_source_stats = []
    for keypoint in unique_keypoints:
        row = {'keypoint': keypoint}
        for source in unique_sources:
            source_data = keypoints_per_source_df[keypoints_per_source_df['source'] == source]
            source_keypoints = source_data[source_data['keypoint'] == keypoint]
            if len(source_keypoints) > 0:
                row[source] = source_keypoints['%'].values[0]
            else:
                row[source] = 0
        keypoints_to_source_stats.append(row)
    keypoints_to_source_stats_df = pd.DataFrame(keypoints_to_source_stats)
    keypoints_to_source_stats_df = keypoints_to_source_stats_df.set_index(['keypoint'])
    keypoints_to_source_stats_df = keypoints_to_source_stats_df.fillna(0)
    keypoints_to_source_stats_df = keypoints_to_source_stats_df.round(2).T
    keypoints_to_source_stats_df_path = os.path.join(args.output_path, 'keypoints_to_source_stats.csv')
    keypoints_to_source_stats_df.to_csv(keypoints_to_source_stats_df_path)
    logger.info(f'Wrote keypoints to source stats to {keypoints_to_source_stats_df_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_judge_results', type=str, required=True)
    parser.add_argument('--keypoints', type=str, required=True)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--judge', type=str)

    args = parser.parse_args()
    logger = setup_default_logger(args.output_path)

    main()
