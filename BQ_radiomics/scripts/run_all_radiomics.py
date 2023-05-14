from BQ_radiomics.scripts import lama_radiomics_runner, lama_machine_learning





def main():
    import argparse
    parser = argparse.ArgumentParser("Generate Radiomic Features and Run Catboost models for prediction")
    parser.add_argument('-i', '--input_file', dest='indirs', help='radiomics file', required=True,
                        type=str)
    parser.add_argument('-c', '--make_org_files', dest='make_org_files',
                        help='Run with this option to split the full into organs',
                        action='store_true', default=False)

    lama_radiomics_runner.main()
    print("Hooly")
    lama_machine_learning.main()


if __name__ == '__main__':
    main()