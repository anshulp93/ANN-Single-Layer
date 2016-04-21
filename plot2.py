import matplotlib
import matplotlib.pyplot as plt

y = [72.40382,77.49996,77.59612,77.40382,76.53842]
x = [5,10,15,20,25]
plt.plot(x,y,'ro-')
plt.ylabel('Accuracy %')
plt.xlabel('Folds')
plt.title("Accuracy vs #Folds for 50 epochs")
#plt.show()
plt.savefig("Plot2.png")
