import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import numpy as np
import os

os.makedirs("model", exist_ok=True)

print("Building lightweight CNN age regression model...")

model = Sequential([
    Conv2D(16,(3,3),activation='relu',input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(64,activation='relu'),
    Dropout(0.2),

    Dense(1,activation='linear')
])

model.compile(optimizer='adam', loss='mse')

print("Training synthetic model...")

X = np.random.rand(200,64,64,3)
y = np.random.randint(18,40,size=(200,))

model.fit(X,y,epochs=10,batch_size=32)

model.save("model/cnn_age_model.h5")

print("\nModel saved at model/cnn_age_model.h5")
