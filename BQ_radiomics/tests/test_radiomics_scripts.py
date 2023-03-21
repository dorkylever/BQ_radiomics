
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
from lama.scripts import lama_machine_learning
import pacmap
from lama.scripts import lama_permutation_stats

stats_cfg = Path(
    "C:/LAMA/lama/tests/configs/permutation_stats/perm_no_qc.yaml")

from lama.stats.permutation_stats.run_permutation_stats import get_radiomics_data


def test_radiomics():
        cpath =  Path('C:/LAMA/lama/tests/configs/lama_radiomics/radiomics_config.toml')
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






@pytest.mark.skip
def test_feat_reduction():
    feature_reduction.main()




@pytest.mark.skip
def test_radiomic_org_plotting():
    _dir = Path("E:/220607_two_way/g_by_back_data/radiomics_output/features/")

    file_names = [spec for spec in common.get_file_paths(folder=_dir, extension_tuple=".csv")]
    file_names.sort()

    data = [pd.read_csv(spec, index_col=0).dropna(axis=1) for spec in file_names]

    abnormal_embs = ['22300_e8', '22300_e6', '50_e5']

    for i, df in enumerate(data):
        df.index.name = 'org'
        df.name = str(file_names[i]).split(".")[0].split("/")[-1]
        df['genotype'] = 'HET' if 'het' in str(file_names[i]) else 'WT'
        df['background'] = 'C56BL6N' if (('b6ku' in str(file_names[i])) | ('BL6' in str(file_names[i]))) else 'C3HHEH'
        df['HPE'] = 'abnormal' if any(map(str(file_names[i]).__contains__, abnormal_embs)) else 'normal'

    data = pd.concat(
        data,
        ignore_index=False, keys=[os.path.splitext(os.path.basename(spec))[0] for spec in file_names],
        names=['specimen', 'org'])

    line_file = _dir.parent / "full_results.csv"

    data.to_csv(line_file)

    #data_subset = data.select_dtypes(include=np.number)

    for i, org in enumerate(data.index.levels[1]):
        fig, ax = plt.subplots(1, 1, figsize=[56, 60])
        #sns.set(font_scale=0.5)
        o_data = data[np.isin(data.index.get_level_values('org'), org)]

        o_data_subset = o_data.select_dtypes(include=np.number)
        #o_data_subset = o_data_subset.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        o_data_subset = o_data_subset.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

        tsne = TSNE(perplexity=30,
                    n_components=2,
                    random_state=0,
                    early_exaggeration=250,
                    n_iter=1000,
                    verbose=1)

        tsne_results = tsne.fit_transform(o_data_subset.dropna(axis='columns'))

        o_data['tsne-2d-one'] = tsne_results[:, 0]
        o_data['tsne-2d-two'] = tsne_results[:, 1]
        o_data['org'] = o_data.index.get_level_values('org')
        o_data['specimen'] = o_data.index.get_level_values('specimen')

        o_data['condition'] = o_data['genotype'] + "_" + o_data['background']

        fig, ax = plt.subplots()
        o_data = o_data[o_data['condition'] == 'WT_C3HHEH']
        g = sns.lmplot(
            x="tsne-2d-one", y="tsne-2d-two",
            data=o_data,
            # col_order=['WT_C3HHEH','HET_C3HHEH','WT_C57BL6N','HET_C57BL6N'],
            #col='specimen',
            #col_wrap=5,
            hue="specimen",
            palette='husl',
            fit_reg=False)

        def label_point(x, y, val, ax):
            a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
            for i, point in a.iterrows():
                ax.text(point['x'] + .02, point['y'], str(point['val']), fontsize='xx-small')

        label_point(o_data['tsne-2d-one'], o_data['tsne-2d-two'], o_data['specimen'], plt.gca())



        #g.set(ylim=(np.min(o_data['tsne-2d-two']) - 10, np.max(o_data['tsne-2d-two']) + 10),
         #     xlim=(np.min(o_data['tsne-2d-one']) - 10, np.max(o_data['tsne-2d-one']) + 10))
        plt.savefig("E:/220607_two_way/g_by_back_data/radiomics_output/radiomics_2D_tsne_C3H_wt_" + str(org) + ".png")
        plt.close()


def test_get_rad_data_for_perm():
    _dir = Path("E:/221122_two_way/g_by_back_data/radiomics_output")

    wt_dir = Path("E:/221122_two_way/g_by_back_data/baseline")

    mut_dir = Path("E:/221122_two_way/g_by_back_data/mutants")

    treat_dir = Path("E:/221122_two_way/g_by_back_data/treatment")

    inter_dir = Path("E:/221122_two_way/g_by_back_data/mut_treat")

    results = get_radiomics_data(_dir, wt_dir, mut_dir, treat_dir, inter_dir)

    results.to_csv(str(_dir/"test_dataset.csv"))

def test_permutation_stats():
    """
    Run the whole permutation based stats pipeline.
    Copy the output from a LAMA registrations test run, and increase or decrease the volume of the mutants so we get
    some hits

    """
    lama_permutation_stats.run(stats_cfg)

