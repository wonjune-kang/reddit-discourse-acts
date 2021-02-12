import sys
import pandas as pd


if __name__ == '__main__':
    score_file = sys.argv[1]
    scores_df = pd.read_csv(score_file, delimiter="\t").drop('Fold', axis=1)
    xval_avg_scores = scores_df.mean()
    print(xval_avg_scores)
