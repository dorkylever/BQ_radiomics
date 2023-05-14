
from BQ_radiomics.secondary_validation import secondary_dataset_confusion_matrix
def main():
    import argparse
    parser = argparse.ArgumentParser("Create confusion matrix given a directory and validation dataset")
    parser.add_argument('-m', '--', dest='model_file', help='model to use', required=True,
                        type=str)
    parser.add_argument('-f', '--validation_file', dest='vfile',
                        help='Run with this option to split the full into organs',
                        action='store_true', default=False)
    args = parser.parse_args()
    model_file = Path(args.indirs)
    vfile = Path(args.vfile)
    secondary_dataset_confusion_matrix(model_file, vfile)



if __name__ == '__main__':
    main()