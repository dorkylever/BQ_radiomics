

from BQ_radiomics import feature_reduction
from BQ_radiomics import common

from filelock import SoftFileLock, Timeout
import socket
from datetime import datetime


from logzero import logger as logging
from pathlib import Path
import pandas as pd
import os
import sys


JOBFILE_NAME = 'ml_jobs.csv'

def make_ml_jobs_file(jobs_file: Path, file_paths: list):
    """
    Creates a joblist csv file for use with the radiomics pipeline.
    Searches for all images paths and creates a job file

    Parameters
    ----------
    jobs_file: Path to save job file to
    Path: the root project directory


    """
    # output_dir = root_dir / 'radiomics_output'
    # output_dir.mkdir(exist_ok=True)

    jobs_entries = []
    # get each file path
    for i, csv_path in enumerate(file_paths):

        rel_path_to_org_input = str(csv_path.relative_to(jobs_file.parent))
        jobs_entries.append([rel_path_to_org_input, 'to_run', '_', '_', '_'])

    jobs_df = pd.DataFrame.from_records(jobs_entries, columns=['job', 'status', 'host', 'start_time', 'end_time'])

    jobs_df.to_csv(jobs_file)
    return True



def ml_job_runner(org_dir):
    '''
    Performs the pyradiomic calculations


    Parameters
    ----------
    target_dir



    Returns
    -------

    '''


    # get org csv files

    org_dir = Path(org_dir)
    names = common.get_file_paths(org_dir, extension_tuple=".csv")

    jobs_file_path = org_dir / JOBFILE_NAME
    lock_file = jobs_file_path.with_suffix('.lock')
    lock = SoftFileLock(lock_file)

    if not os.path.exists(jobs_file_path):
        logging.info("Creating a job-file for ml")
        make_ml_jobs_file(jobs_file_path, names)
        logging.info("Job_file_created")

    df_jobs = pd.read_csv(jobs_file_path, index_col=0)

    # execute parallelisation:
    while True:
        try:
            with lock.acquire(timeout=60):

                # Get an unfinished job
                jobs_to_do = df_jobs[df_jobs['status'] == 'to_run']
                if len(jobs_to_do) < 1:
                    logging.info("No more jobs left on jobs list")

                    logging.info("checking for hung jobs")

                    # get last job and check start-time
                    fin_jobs = df_jobs[df_jobs['status'] == 'complete']
                    running_jobs = df_jobs[df_jobs['status'] == 'running']
                    fin_indx = fin_jobs.index[-1]
                    fin_t = fin_jobs.at[fin_indx, 'start_time']
                    fin_time = datetime.strptime(fin_t, '%Y-%m-%d %H:%M:%S')
                    run_t = running_jobs['start_time']
                    run_times = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') < fin_time for t in run_t]
                    hung_jobs = running_jobs[run_times]


                    if len(hung_jobs) > 0:
                        logging.info("Hung jobs found - rerunning")
                        jobs_to_do = hung_jobs
                    else:
                        break
                indx = jobs_to_do.index[0]

                org_csv_path = Path(org_dir) / (jobs_to_do.at[indx, 'job'])




                df_jobs.at[indx, 'status'] = 'running'
                df_jobs.at[indx, 'start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                df_jobs.at[indx, 'host'] = socket.gethostname()

                df_jobs.to_csv(jobs_file_path)
        except Timeout:
            sys.exit('Timed out' + socket.gethostname())

        # try:
        logging.info(f'trying {org_csv_path}')
        # get the organ file and number
        org_df = pd.read_csv(org_csv_path)
        try:
            org = org_df['org'][0]
            feature_reduction.main(org_df, org, org_dir)
        except KeyError:
            # BQ data should have no 'org' info
            features = org_df
            features = features[features.columns.drop(list(features.filter(regex="diagnostics")))]
            features.drop(["scanID"], axis=1, inplace=True)
            feature_reduction.main(features, org=None, rad_file_path=Path(org_dir.parent / "full_results.csv"))

        # perform feature reduction on a single organ

        # except Exception as e:
        #    if e.__class__.__name__ == 'KeyboardInterrupt':
        #        logging.info('terminating')
        #        sys.exit('Exiting')

        #    status = 'failed'
        #    print(e)
        #    logging.exception(e)

        status = 'complete'

        with lock:
            df_jobs = pd.read_csv(jobs_file_path, index_col=0)
            df_jobs.at[indx, 'status'] = status
            df_jobs.at[indx, 'end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df_jobs.to_csv(jobs_file_path)

    logging.info('Exiting job_runner')
    return True

def main(indir, make_job_file):
    _dir = Path(args.indirs)
    if args.make_org_files:
        common.gather_rad_data(_dir)
    else:
        ml_job_runner(_dir)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Run Catboost models for prediction")
    parser.add_argument('-i', '--input_file', dest='indirs', help='radiomics file', required=True,
                        type=str)
    parser.add_argument('-m', '--make_org_files', dest='make_org_files',
                        help='Run with this option to split the full into organs',
                        action='store_true', default=False)
    args = parser.parse_args()
    indir = args.indirs
    make_job_file = args.make_org_files
    main(indir, make_job_file)
