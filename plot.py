import matplotlib.pyplot as plt

class Plot:

    def __init__(self, data, x_label, y_label):
        self.data = data
        self.x_label = x_label
        self.y_label = y_label
        
    def show(self):
        plt.plot(self.data)
        plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)
        plt.show()

    def show_sub_plot(self, subplot):
        plt.figure(1)
        plt.subplot(211)
        plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)
        plt.plot(self.data)
        
        plt.subplot(212)
        plt.ylabel(subplot.y_label)
        plt.xlabel(subplot.x_label)
        plt.plot(subplot.data)

        plt.subplots_adjust(hspace= 0.35)
        plt.show()
        plt.close()

    def save_sub_plot(self, subplot, savefile):
        plt.figure(1, figsize=(8, 6))
        plt.subplot(211)
        plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)
        plt.plot(self.data)
        
        plt.subplot(212)
        plt.ylabel(subplot.y_label)
        plt.xlabel(subplot.x_label)
        plt.plot(subplot.data)

        plt.subplots_adjust(hspace= 0.35)
        plt.savefig(savefile, dpi=64)
        plt.close()