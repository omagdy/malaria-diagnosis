import matplotlib.pyplot as plt

def plot_evaluations(epochs_plot, training_loss, validation_loss):
    plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(epochs_plot, training_loss, label = 'Training Loss')
    plt.plot(epochs_plot, validation_loss, label = 'Validation Loss')
    plt.legend()
    # plt.show()
    plt.savefig('loss_plots.png', bbox_inches='tight')
