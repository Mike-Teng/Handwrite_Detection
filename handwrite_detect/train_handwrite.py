from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os

def data_process(datapath):
    pic_count = 0
    data_y = []
    numClass = 10
    data_x = np.zeros((28,28)).reshape(1,28,28)
      
    for root, _ , files in os.walk(datapath):
#         if files != ['.DS_Store']:
#             print(root,files)
        if files != ['.DS_Store']:
            for f in files:
                if f != ".DS_Store":
                    label = int(root.split('/')[2])
                    data_y.append(label)
                    fullpath = os.path.join(root,f)
                    img = Image.open(fullpath)
                    img = (np.array(img)/255).reshape(1,28,28)
                    data_x = np.vstack([data_x,img])
                    pic_count += 1

    data_x = np.delete(data_x,[0],0)
    data_x = data_x.reshape(pic_count,28,28,1)
    data_y = to_categorical(data_y,numClass)
    return data_x,data_y

traindatapath = "train_image"
testdatapath = "test_image"
data_train_x, data_train_y = data_process(traindatapath)
data_test_x, data_test_y = data_process(testdatapath)

model = Sequential()
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model.fit(data_train_x,data_train_y, batch_size=128, epochs=55, verbose=1, validation_split=0.05)
score = model.evaluate(data_test_x,data_test_y,verbose=0)
print('Test loss:', score[0])
print('Test acc:', score[1])

plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.plot(train_history.history['val_accuracy'])
plt.title('train history')
plt.show()