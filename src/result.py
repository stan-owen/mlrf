import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

def plot_results(typeDesc, model, X_test_color, Y_test_color, labels_name):
    model_name = model.__class__.__name__

    Y_pred_color = model.predict(X_test_color)

    accuracy = accuracy_score(Y_test_color, Y_pred_color)
    conf_mat = confusion_matrix(Y_test_color, Y_pred_color)

    Y_pred_proba = model.predict_proba(X_test_color)
    fpr, tpr, _ = roc_curve(Y_test_color, Y_pred_proba[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Création de la figure et des sous-graphiques
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    # Premier graphique : Nom du model et resultat accuracy
    
    axs[0].text(0.5, 0.5, typeDesc + '\n\n' + model_name + '\n\n' + str('Accuracy %.2f' % (accuracy * 100)) + '%', fontsize=30, ha='center', va='center')
    axs[0].axis('off')

    # Deuxième graphique : Heatmap de la matrice de confusion
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False, ax=axs[1])
    axs[1].set_xticks(np.arange(10))
    axs[1].set_xticklabels(labels_name, rotation=90)
    axs[1].set_yticks(np.arange(10))
    axs[1].set_yticklabels(labels_name, rotation=0)
    axs[1].set_xlabel('Predicted Label')
    axs[1].set_ylabel('True Label')

    # Troisième graphique : Courbe ROC
    axs[2].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)'% roc_auc)
    axs[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[2].set_xlabel('False Positive Rate', fontsize=15)
    axs[2].set_ylabel('True Positive Rate', fontsize=15)
    axs[2].set_title('ROC Curve', fontsize=20)
    axs[2].legend(loc="lower right")

    # Affichage de la figure
    plt.tight_layout()
    plt.show()