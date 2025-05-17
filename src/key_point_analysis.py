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

    judge_results['unique_keypoints'] = judge_results['keypoints'].apply(lambda x: list(set(x)))

    keypoints_to_nr_comments = {}
    for _, row in judge_results.iterrows():
        for keypoint in row['unique_keypoints']:
            if keypoint not in keypoints_to_nr_comments:
                keypoints_to_nr_comments[keypoint] = 0
            keypoints_to_nr_comments[keypoint] += 1

    keypoints_to_nr_comments_list = []
    for keypoint, count in keypoints_to_nr_comments.items():
        keypoints_to_nr_comments_list.append({
            'keypoint': keypoint,
            '# speeches': count,
            'stance': kp_stence[keypoint],
        })

    keypoints_to_nr_comments_df = pd.DataFrame(keypoints_to_nr_comments_list)
    keypoints_to_nr_comments_df = keypoints_to_nr_comments_df.sort_values(by=['stance', '# speeches'], ascending=[True, False])
    keypoints_to_nr_comments_df['% speeches'] = keypoints_to_nr_comments_df['# speeches'] / len(judge_results) * 100
    keypoints_to_nr_comments_df = keypoints_to_nr_comments_df.round(2)
    keypoints_to_nr_comments_df_path = os.path.join(args.output_path, 'keypoints_to_nr_comments.csv')
    keypoints_to_nr_comments_df.to_csv(keypoints_to_nr_comments_df_path, index=False)
    logger.info(f'Wrote keypoints to nr comments to {keypoints_to_nr_comments_df_path}')

    keypoints_to_nr_source_comments = {}
    for _, row in judge_results.iterrows():
        for keypoint in row['unique_keypoints']:
            if keypoint not in keypoints_to_nr_source_comments:
                keypoints_to_nr_source_comments[keypoint] = {}
            if row['source'] not in keypoints_to_nr_source_comments[keypoint]:
                keypoints_to_nr_source_comments[keypoint][row['source']] = 0
            keypoints_to_nr_source_comments[keypoint][row['source']] += 1

    keypoints_to_nr_source_comments_list = []
    for keypoint, sources in keypoints_to_nr_source_comments.items():
        row = {'keypoint': keypoint, 'stance': kp_stence[keypoint]}
        all_speeches = 0
        for source, count in sources.items():
            row[f'% {source}'] = count / len(judge_results[judge_results['source'] == source]) * 100
            all_speeches += count
        row['% speeches'] = all_speeches / len(judge_results) * 100
        keypoints_to_nr_source_comments_list.append(row)

    keypoints_to_nr_source_comments_df = pd.DataFrame(keypoints_to_nr_source_comments_list)
    keypoints_to_nr_source_comments_df = keypoints_to_nr_source_comments_df.sort_values(by=['stance', 'keypoint'], ascending=[True, True])
    keypoints_to_nr_source_comments_df = keypoints_to_nr_source_comments_df.round(2)
    keypoints_to_nr_source_comments_df_path = os.path.join(args.output_path, 'keypoints_to_nr_source_comments.csv')
    keypoints_to_nr_source_comments_df.to_csv(keypoints_to_nr_source_comments_df_path, index=False)
    logger.info(f'Wrote keypoints to nr source comments to {keypoints_to_nr_source_comments_df_path}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_judge_results', type=str, required=True)
    parser.add_argument('--keypoints', type=str, required=True)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--judge', type=str)

    args = parser.parse_args()
    logger = setup_default_logger(args.output_path)

    main()
