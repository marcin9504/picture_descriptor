from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


def draw(y_true, y_scores):
    score = roc_auc_score(y_true, y_scores)
    print(score)
    plt.figure()
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.show()
