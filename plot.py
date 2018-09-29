import matplotlib.pyplot as plt

class Plot:

    def __init__(self, data, save_folder_path):
        self.data = data
        self.save_folder_path = save_folder_path

    def save_error_plot(self, display_plot=True):
        plt.figure(1)
        plt.plot(self.data.history['mean_absolute_error'])
        plt.plot(self.data.history['val_mean_absolute_error'])
        plt.title('model mean_absolute_error')
        plt.ylabel('mean_absolute_error')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.yscale('log')
        plt.savefig(self.save_folder_path + '/mean_absolute_error.png', dpi=128)
        if display_plot:
            plt.show()
        plt.close()

    def save_accuracy_plot(self, display_plot=True):
        plt.figure(2)
        plt.plot(self.data.history['loss'])
        plt.plot(self.data.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.yscale('log')
        plt.savefig(self.save_folder_path + '/loss.png', dpi=128)
        if display_plot:
            plt.show()
        plt.close()


    # def __init__(self, data, x_label, y_label):
    #     self.data = data
    #     self.x_label = x_label
    #     self.y_label = y_label
        
    # def show(self):
    #     plt.plot(self.data)
    #     plt.ylabel(self.y_label)
    #     plt.xlabel(self.x_label)
    #     plt.show()

    # def show_sub_plot(self, subplot):
    #     plt.figure(1)
    #     plt.subplot(211)
    #     plt.ylabel(self.y_label)
    #     plt.xlabel(self.x_label)
    #     plt.plot(self.data)
        
    #     plt.subplot(212)
    #     plt.ylabel(subplot.y_label)
    #     plt.xlabel(subplot.x_label)
    #     plt.plot(subplot.data)

    #     plt.subplots_adjust(hspace= 0.35)
    #     plt.show()
    #     plt.close()

    # def save_sub_plot(self, subplot, savefile):
    #     plt.figure(1, figsize=(8, 6))
    #     plt.subplot(211)
    #     plt.ylabel(self.y_label)
    #     plt.xlabel(self.x_label)
    #     plt.plot(self.data)
        
    #     plt.subplot(212)
    #     plt.ylabel(subplot.y_label)
    #     plt.xlabel(subplot.x_label)
    #     plt.plot(subplot.data)

    #     plt.subplots_adjust(hspace= 0.35)
    #     plt.savefig(savefile, dpi=64)
    #     plt.close()