
from pathlib import Path
from BQ_radiomics.radiomics import radiomics_job_runner
from BQ_radiomics import common
import os
from BQ_radiomics.common import cfg_load
from BQ_radiomics import normalise
from logzero import logger as logging
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pytest
import numpy as np
from BQ_radiomics import feature_reduction
from BQ_radiomics.scripts import lama_machine_learning
import subprocess
import BQ_radiomics


def test_radiomics():
        cpath =  Path('C:/radiomics_config.toml')
        c = cfg_load(cpath)

        target_dir = Path(c.get('target_dir'))

        labs_of_int = c.get('labs_of_int')

        norm_methods = c.get('norm_methods')


        norm_label = True

        spherify = c.get('spherify')

        ref_vol_path = Path(c.get('ref_vol_path'))

        norm_dict = {
            "histogram": normalise.IntensityHistogramMatch(),
            "N4": normalise.IntensityN4Normalise(),
            "subtraction": normalise.NonRegMaskNormalise(),
            "none": None
        }
        try:
            norm_meths = [norm_dict[str(x)] for x in norm_methods]

        except KeyError as e:
            print(e)

            norm_meths = None
        logging.info("Starting Radiomics")
        radiomics_job_runner(target_dir, labs_of_int=labs_of_int, norm_method=norm_methods, norm_label=norm_label,
                             spherify=spherify, ref_vol_path=None, make_job_file=True)

def test_mach_learn_pipeline():
    lama_machine_learning.ml_job_runner("E:/230129_bq_tester/norm_methods/")


def test_BQ_concat():
    _dir = Path("Z:/jcsmr/ROLab/Experimental data/Radiomics/Workflow design and trial results/Kyle Drover analysis/220617_BQ_norm_stage_full/sub/sub_normed_features.csv")
    #_dir = Path("E:/220913_BQ_tsphere/inputs/features/")

    # file_names = [spec for spec in common.get_file_paths(folder=_dir, extension_tuple=".csv")]
    # file_names.sort()
    #
    # data = [pd.read_csv(spec, index_col=0).dropna(axis=1) for spec in file_names]
    #
    # data = pd.concat(
    #     data,
    #     ignore_index=False, keys=[os.path.splitext(os.path.basename(spec))[0] for spec in file_names],
    #     names=['specimen', 'label'])
    #
    # data['specimen'] = data.index.get_level_values('specimen')
    #
    # _metadata = data['specimen'].str.split('_', expand=True)
    #
    #
    #
    # _metadata.columns = ['Date', 'Exp', 'Contour_Method', 'Tumour_Model', 'Position', 'Age',
    #                                                                         'Cage_No.', 'Animal_No.']
    #
    #
    #
    #
    # _metadata.reset_index(inplace=True, drop=True)
    # data.reset_index(inplace=True, drop=True)
    # features = pd.concat([_metadata, data], axis=1)
    #
    # features.index.name = 'scanID'
    #
    # print(features)
    #
    # print(str(_dir.parent / "full_results.csv"))
    #
    # features.to_csv(str(_dir.parent / "full_results.csv"))

    features = pd.read_csv(_dir)
    features = features[features.columns.drop(list(features.filter(regex="diagnostics")))]
    features.drop(["scanID"], axis=1, inplace=True)
    feature_reduction.main(features, org = None, rad_file_path = Path(_dir.parent / "full_results.csv"))

def test_BQ_mach_learn():
    _dir = Path("C:/test/features/")

    file_names = [spec for spec in common.get_file_paths(folder=_dir, extension_tuple=".csv")]
    file_names.sort()

    data = [pd.read_csv(spec, index_col=0).dropna(axis=1) for spec in file_names]

    data = pd.concat(
        data,
        ignore_index=False, keys=[os.path.splitext(os.path.basename(spec))[0] for spec in file_names],
        names=['specimen', 'label'])

    data['specimen'] = data.index.get_level_values('specimen')

    _metadata = data['specimen'].str.split('_', expand=True)



    _metadata.columns = ['Date', 'Exp', 'Contour_Method', 'Tumour_Model', 'Position', 'Age',
                                                                            'Cage_No.', 'Animal_No.']




    _metadata.reset_index(inplace=True, drop=True)
    data.reset_index(inplace=True, drop=True)
    features = pd.concat([_metadata, data], axis=1)

    features.index.name = 'scanID'

    print(features)

    print(str(_dir.parent / "full_results.csv"))

    features.to_csv(str(_dir.parent / "full_results.csv"))

    feature_reduction.main(features, org = None, rad_file_path = Path(_dir.parent / "full_results.csv"))


def test_BQ_mach_learn_non_tum():
    _dir = Path("E:/220919_non_tum/features/")

    file_names = [spec for spec in common.get_file_paths(folder=_dir, extension_tuple=".csv")]
    file_names.sort()

    data = [pd.read_csv(spec, index_col=0).dropna(axis=1) for spec in file_names]

    data = pd.concat(
        data,
        ignore_index=False, keys=[os.path.splitext(os.path.basename(spec))[0] for spec in file_names],
        names=['specimen', 'label'])

    data['specimen'] = data.index.get_level_values('specimen')

    _metadata = data['specimen'].str.split('_', expand=True)



    _metadata.columns = ['Date', 'Exp', 'Contour_Method', 'Tumour_Model', 'Position', 'Age',
                                                                            'Cage_No.', 'Animal_No.']




    _metadata.reset_index(inplace=True, drop=True)
    data.reset_index(inplace=True, drop=True)
    features = pd.concat([_metadata, data], axis=1)

    features.index.name = 'scanID'

    print(features)

    print(str(_dir.parent / "full_results.csv"))

    features.to_csv(str(_dir.parent / "full_results.csv"))

    feature_reduction.main(features, org = None, rad_file_path = Path(_dir.parent / "full_results.csv"))



def test_BQ_mach_learn_batch_sp():
    _dir = Path("E:/220913_BQ_tsphere/inputs/features/")

    file_names = [spec for spec in common.get_file_paths(folder=_dir, extension_tuple=".csv")]
    file_names.sort()

    data = [pd.read_csv(spec, index_col=0).dropna(axis=1) for spec in file_names]

    data = pd.concat(
        data,
        ignore_index=False, keys=[os.path.splitext(os.path.basename(spec))[0] for spec in file_names],
        names=['specimen', 'label'])

    data['specimen'] = data.index.get_level_values('specimen')

    _metadata = data['specimen'].str.split('_', expand=True)



    _metadata.columns = ['Date', 'Exp', 'Contour_Method', 'Tumour_Model', 'Position', 'Age',
                                                                            'Cage_No.', 'Animal_No.']




    _metadata.reset_index(inplace=True, drop=True)
    data.reset_index(inplace=True, drop=True)
    features = pd.concat([_metadata, data], axis=1)

    features.index.name = 'scanID'

    print(features)

    print(str(_dir.parent / "full_results.csv"))

    features.to_csv(str(_dir.parent / "full_results.csv"))

    feature_reduction.main(features, org = None, rad_file_path = Path(_dir.parent / "full_results.csv"), batch_test=True)


def test_BQ_concat_batch():
    _dir = Path("Z:/jcsmr/ROLab/Experimental data/Radiomics/Workflow design and trial results/Kyle Drover analysis/220617_BQ_norm_stage_full/sub_normed_features.csv")
    #_dir = Path("E:/220913_BQ_tsphere/inputs/features/")

    # file_names = [spec for spec in common.get_file_paths(folder=_dir, extension_tuple=".csv")]
    # file_names.sort()
    #
    # data = [pd.read_csv(spec, index_col=0).dropna(axis=1) for spec in file_names]
    #
    # data = pd.concat(
    #     data,
    #     ignore_index=False, keys=[os.path.splitext(os.path.basename(spec))[0] for spec in file_names],
    #     names=['specimen', 'label'])
    #
    # data['specimen'] = data.index.get_level_values('specimen')
    #
    # _metadata = data['specimen'].str.split('_', expand=True)
    #
    #
    #
    # _metadata.columns = ['Date', 'Exp', 'Contour_Method', 'Tumour_Model', 'Position', 'Age',
    #                                                                         'Cage_No.', 'Animal_No.']
    #
    #
    #
    #
    # _metadata.reset_index(inplace=True, drop=True)
    # data.reset_index(inplace=True, drop=True)
    # features = pd.concat([_metadata, data], axis=1)
    #
    # features.index.name = 'scanID'
    #
    # print(features)
    #
    # print(str(_dir.parent / "full_results.csv"))
    #
    # features.to_csv(str(_dir.parent / "full_results.csv"))

    features = pd.read_csv(_dir)
    features = features[features.columns.drop(list(features.filter(regex="diagnostics")))]
    features.drop(["scanID"], axis=1, inplace=True)
    feature_reduction.main(features, org = None, rad_file_path = Path(_dir.parent / "full_results.csv"), batch_test=True)


def test_catboost_plots():
    RSCRIPT_DIR = Path(BQ_radiomics.__file__).parent / "scripts" / "catboost_test.R"
    _file = "E:/220204_BQ_dataset/scans_for_sphere_creation/sphere_15_res/test_size_0.2/None/sphere_15_corrected.csv"

    output = subprocess.check_output(["Rscript", str(RSCRIPT_DIR), "--input_file", _file], universal_newlines=True)


    print(eval(output.strip()))
    logloss_nfeats, logloss_score, acc_nfeats, acc_score = eval(output.strip())

    # Print the unpacked variables
    print("Log Loss - nfeats:", logloss_nfeats)
    print("Log Loss - score:", logloss_score)
    print("Accuracy - nfeats:", acc_nfeats)
    print("Accuracy - score:", acc_score)


@pytest.mark.skip
def test_feat_reduction():
    feature_reduction.main()
