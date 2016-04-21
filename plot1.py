import matplotlib
import matplotlib.pyplot as plt

y = [74.04244,77.0192,76.53846,76.53846]
x = [25,50,75,100]
plt.plot(x,y,'ro-')
plt.ylabel('Accuracy %')
plt.xlabel('Epochs')
plt.title("Accuracy vs #epochs for 10 folds")
#plt.show()
plt.savefig("Plot1.png")
