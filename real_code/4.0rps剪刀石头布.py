import tensorflow as tf
train_dir = 'E:/chrome下载/Rock Paper Scissors Dataset_datasets_/Rock-Paper-Scissors/train'
test_dir = 'E:/chrome下载/Rock Paper Scissors Dataset_datasets_/Rock-Paper-Scissors/test'
validation_dir = 'E:/chrome下载/Rock Paper Scissors Dataset_datasets_/Rock-Paper-Scissors/validation'
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, rotation_range = 40,width_shift_range = 0.2,height_shift_range = 0.2,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True,fill_mode='nearest')
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(train_dir,(300,300),class_mode = 'categorical')
validation_generator = validation_datagen.flow_from_directory(test_dir,(300,300),class_mode='categorical')
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(300,300,3)))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(512,'relu'))
model.add(tf.keras.layers.Dense(3,'softmax'))
print(model.summary())
model.compile('rmsprop','categorical_crossentropy',['accuracy'])
history = model.fit_generator(train_generator,epochs=25,validation_data=validation_generator,verbose=1)




