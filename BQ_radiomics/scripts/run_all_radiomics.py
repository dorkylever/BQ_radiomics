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

from pathlib import Path

RSCRIPT_DIR = Path(BQ_radiomics.__file__).parent / "scripts" / "catboost_test.R"

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


    args = parser.parse_args()

    _dir = Path(args.indirs)
    logging.info("Making job file")
    lama_radiomics_runner.main(config=args.config, make_job_file=True)


    #lama_radiomics_runner.main(config=args.config, make_job_file=False)



    num_jobs=4
    logging.info(f"running with {num_jobs} jobs")
    # Execute the function in parallel using joblib
    def run_lama_radiomics(i):
        logging.info(f"running job {i}")
        lama_radiomics_runner.main(config=args.config, make_job_file=False)

    # Execute the function in parallel using joblib
    #class JobRunnerExit(Exception):
    #    pass
    Parallel(n_jobs=-1)(delayed(run_lama_radiomics)(i) for i in range(num_jobs))

    #except JobRunnerExit:
    #    logging.info("Finished Script")

    logging.info("radiomics have been generated, running ml")

    # this should generate the main script
    feat_dir = _dir / "radiomics_output" / "features"

    rad_out_dir = _dir / "radiomics_output"

    lama_machine_learning.main(indir=feat_dir, make_job_file=True)

    lama_machine_learning.main(indir=rad_out_dir, make_job_file=False)

    out_dir = _dir / "test_size_0.2"

    cv_data = n_feat_res_combined(out_dir)

    out_file =  _dir / "full_cv_dataset.csv"

    cv_data.to_csv(out_file)

    #so let's call the R stuff - to plot the training cruve:

    output = subprocess.check_output(["Rscript", str(RSCRIPT_DIR), "--input_file", out_file], universal_newlines=True)


    logloss_nfeats, logloss_score, acc_nfeats, acc_score = eval(output.strip())



    # so the results file should be generated
    validation_file = _dir / "radiomics_output" / "test_size_0.2" / "test_0.2.csv"

    #load the catboost model


    model_path = _dir / "radiomics_output" / "test_size_0.2" /  f"combined_results_{str(logloss_nfeats)}.cbm"


    confusion_matrix_generator.main(vfile=validation_file,model_file=model_path)


if __name__ == '__main__':
    main()