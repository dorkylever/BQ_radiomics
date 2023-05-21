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

def call_r_script(input_file):
    r_script = "rscript_wrapper.sh"

    # Construct the command to call the shell script with arguments
    command = ["./" + r_script, input_file]

    # Execute the command and capture the output
    output = subprocess.check_output(command, universal_newlines=True)

    nfeats_from_logloss, nfeats_from_accuracy = output
    return nfeats_from_logloss, nfeats_from_accuracy

def main():
    import argparse
    parser = argparse.ArgumentParser("Generate Radiomic Features and Run Catboost models for prediction")
    parser.add_argument('-c', '--config', dest='config', help='lama.yaml config file',
                        required=True)
    parser.add_argument('-i', '--input_dirs', dest='indirs', help='folder with scans, labels and tumour contours', required=True,
                        type=str)


    args = parser.parse_args()

    _dir = Path(args.indirs)

    lama_radiomics_runner.main(config=args.config, make_job_file=True)

    num_jobs=10
    # Execute the function in parallel using joblib
    def run_lama_radiomics():
        lama_radiomics_runner.main(config=args.config, make_job_file=False)

    # Execute the function in parallel using joblib
    Parallel(n_jobs=-1)(delayed(run_lama_radiomics)() for _ in range(num_jobs))

    # this should generate the main script
    lama_machine_learning.main(indirs=_dir, make_job_file=True)

    out_dir = _dir / "test_size_0.2" / "None"

    cv_data = n_feat_res_combined(out_dir)

    out_file =  _dir / "full_cv_dataset.csv"

    cv_data.to_csv(out_file)

    #so let's call the R stuff:


    best_nfeats_logloss, best_nfeats_acc = call_r_script(out_file)



    # so the results file should be generated
    validation_file = _dir / "radiomics_output" / "test_size_0.2" / "test.csv"








    lama_machine_learning.main(indirs=_dir, make_job_file=False)



    confusion_matrix_generator.main(vfile=validation_iles)


if __name__ == '__main__':
    main()