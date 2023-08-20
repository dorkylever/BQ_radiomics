from logzero import logger as logging

import pandas as pd

from BQ_radiomics.scripts import lama_radiomics_runner, lama_machine_learning, confusion_matrix_generator
from joblib import Parallel, delayed
import pandas as pd
import os
from pathlib import Path
import subprocess
from BQ_radiomics import common
import BQ_radiomics
import pacmap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

RSCRIPT_DIR = Path(BQ_radiomics.__file__).parent / "scripts" / "catboost_test.R"

def BQ_dimensionality_reduction_plots(_dir: Path):

    input_file = _dir / "full_results.csv"



    data = pd.read_csv(input_file, index_col=0)

    data.drop(['Animal_No.', 'Date'], inplace=True, axis=1)


    data_subset = data.select_dtypes(include=np.number)

    #data_subset = data_subset.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    #data_subset = data_subset.apply(lambda x: (x - x.mean()) / x.std(), axis=1)


    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, num_iters=20000, verbose=200)

    # print(data_subset.dropna(axis='columns'))

    results = embedding.fit_transform(data_subset.dropna(axis='columns'))

    #color_class = data.index.get_level_values('org')

    # fig, ax = plt.subplots(figsize=[55, 60])
    # cluster.tsneplot(score=tsne_results, show=True, theme='dark', colorlist=color_class)

    data['PaCMAP-2d-one'] = results[:, 0]
    data['PaCMAP-2d-two'] = results[:, 1]
    data.to_csv(str(_dir/"data_with_PaCMAP.csv"))
    data['scanID'] = data.index.values

    g = sns.lmplot(
        x="PaCMAP-2d-one", y="PaCMAP-2d-two",
        data=data,
        col="Tumour_Model",
        # col_order=['normal', 'abnormal'],
        hue="Exp",
        palette='husl',
        fit_reg=False)
    g.set(ylim=(np.min(data['PaCMAP-2d-two']) - 1, np.max(data['PaCMAP-2d-two']) + 1),
          xlim=(np.min(data['PaCMAP-2d-one']) - 1, np.max(data['PaCMAP-2d-one']) + 1))

    out_file = _dir / "PacMAP_2D_Exp.png"


    plt.savefig(out_file)
    plt.close()


    g = sns.scatterplot(
        x="PaCMAP-2d-one", y="PaCMAP-2d-two",
        data=data,
        # col_order=['normal', 'abnormal'],
        hue="Exp",
        style="Tumour_Model",
        palette='husl')

    g.set(ylim=(np.min(data['PaCMAP-2d-two']) - 1, np.max(data['PaCMAP-2d-two']) + 1),
          xlim=(np.min(data['PaCMAP-2d-one']) - 1, np.max(data['PaCMAP-2d-one']) + 1))

    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))



    out_file = _dir / "PacMAP_2D_TM.png"

    plt.savefig(out_file)
    plt.figure(figsize=(10, 6))

    plt.close()





def n_feat_res_combined(_dir):

    cv_filenames = [cv_res for cv_res in common.get_file_paths(folder=_dir, extension_tuple=".csv") if ('cross_fold_results' in str(cv_res))]

    cross_folds = [pd.read_csv(file, index_col=0) for file in cv_filenames]

    print(os.path.basename(cv_filenames[0].parent))

    for i, cf in enumerate(cross_folds):
        print(i)
        cf['nfeats'] = int(os.path.basename(cv_filenames[i].parent))

    cv_data = pd.concat(cross_folds)
    return cv_data


def main():
    import argparse
    parser = argparse.ArgumentParser("Generate Radiomic Features and Run Catboost models for prediction")
    parser.add_argument('-c', '--config', dest='config', help='lama.yaml config file',
                        required=True)
    parser.add_argument('-i', '--input_dirs', dest='indirs', help='folder with scans, labels and tumour contours', required=True,
                        type=str)

    parser.add_argument('-a', '--adjacent', dest='adjacent', help='folder with adjacent features',
                        required=False,
                        default=None,
                        type=str)


    args = parser.parse_args()

    _dir = Path(args.indirs)

    adjacent_dir = args.adjacent
    logging.info("Making job file")
    lama_radiomics_runner.main(config=args.config, make_job_file=True)


    num_jobs=3
    logging.info(f"running with {num_jobs} jobs")
    # Execute the function in parallel using joblib
    def run_lama_radiomics(i):
        logging.info(f"running job {i}")
        lama_radiomics_runner.main(config=args.config, make_job_file=False)

    # Execute the function in parallel using joblib
    #class JobRunnerExit(Exception):
    #    pass
    Parallel(n_jobs=1)(delayed(run_lama_radiomics)(i) for i in range(num_jobs))

    #except JobRunnerExit:
    #    logging.info("Finished Script")

    logging.info("radiomics have been generated, running ml")

    # this should generate the main script
    feat_dir = _dir / "radiomics_output" / "features"

    rad_out_dir = _dir / "radiomics_output"


    lama_machine_learning.main(indir=feat_dir, make_job_file=True, adjacent_dir=adjacent_dir)

    # Pacmap
    logging.info("generating PaCMAP")
    BQ_dimensionality_reduction_plots(rad_out_dir)


    lama_machine_learning.main(indir=rad_out_dir, make_job_file=False)

    out_dir = rad_out_dir / "test_size_0.2"

    cv_data = n_feat_res_combined(out_dir)

    out_file = rad_out_dir / "full_cv_dataset.csv"

    cv_data.to_csv(out_file)

    #so let's call the R stuff - to plot the training cruve:

    output = subprocess.check_output(["Rscript", str(RSCRIPT_DIR), "--input_file", str(out_file)], universal_newlines=True)


    logloss_nfeats, logloss_score, acc_nfeats, acc_score = eval(output.strip())



    # so the results file should be generated
    validation_file = rad_out_dir / "test_size_0.2" / "test_0.2.csv"

    #load the catboost model


    model_path = rad_out_dir / "test_size_0.2" / str(acc_nfeats) /  f"cv_{acc_nfeats}.cbm"


    confusion_matrix_generator.main(vfile=validation_file,model_file=model_path)




if __name__ == '__main__':
    main()