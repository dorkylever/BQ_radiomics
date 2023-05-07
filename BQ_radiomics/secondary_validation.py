import catboost
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
def secondary_dataset_confusion_matrix(model_dir: Path, validation_file: Path):


    model = catboost.CatBoostClassifier()

    model.load_model(str(model_dir))
    X = pd.read_csv(str(validation_file))

    #logging.info("Tumour Time!")
    #X['Tumour_Model'] = X['Tumour_Model'].map({'4T1R': 0, 'CT26R': 1}).astype(int)
    #X.set_index('Tumour_Model', inplace=True)
    #X.drop(['Date', 'Animal_No.'], axis=1, inplace=True)
    #X = X.loc[:, ~X.columns.str.contains('shape')]
    #X.dropna(axis=1, inplace=True)


    X = X.select_dtypes(include=np.number)

    y_pred = model.predict(X)

    # Generate a confusion matrix
    conf_matrix = confusion_matrix(X.index, y_pred)

    cm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Print row-wise percentages

    vmin = np.min(cm)
    vmax = np.max(cm)
    off_diag_mask = np.eye(*cm.shape, dtype=bool)

    sns.heatmap(cm, annot=True, mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax)
    sns.heatmap(cm, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, cbar_kws=dict(ticks=[]))

    # Add labels and ticks to the plot
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, ["Actual 4T1", "Actual CT26"])
    plt.yticks(tick_marks, ["Pred 4T1", "Pred CT26"])
    plt.xlabel('True label')
    plt.ylabel('Predicted label')

    # Add values to the plot
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i][j],
                     horizontalalignment="center",
                     color="white" if cm[i][j] > cm.max() / 2. else "black")

    # Save the plot as a PNG image
    plt.savefig(str(model_dir.parent) +'/confusion_matrix.png')

