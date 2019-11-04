from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import csv
import numpy as np
from collections import OrderedDict
import pickle

from keras.preprocessing import image

base_model = MobileNet(weights='imagenet',
                       include_top=False)  # imports the imagenet model and discards the last layer.

x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add multiple layers to get better results
# x = Dense(1024, activation='relu')(
#     x)  # Add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024, activation='relu')(x)  # dense layer 1
x = Dense(512, activation='relu')(x)  # dense layer 2
preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)
# specify the inputs & outputs
# now a model has been created based on our architecture
# Only train the new layers
for layer in model.layers[:2]:
    layer.trainable = False
for layer in model.layers[2:]:
    layer.trainable = True

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies

train_generator = train_datagen.flow_from_directory('./DataSet/Train Images/',
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)

# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

step_size_train = train_generator.n // train_generator.batch_size
step_size_train = step_size_train + 36
model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=6)

base_img_path = './DataSet/Test Images/'
# Predict values from test.csv file
prediction_dict = OrderedDict()
prediction_dict['Image_File'] = 'Class'

with open('./DataSet/test.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        img_path = base_img_path + row[0]
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        rock_type_prediction = model.predict(x)
        prediction_dict[row[0]] = 'Large' if rock_type_prediction[0][0] > rock_type_prediction[0][1] else 'Small'

with open('./LunarSubmission.csv', 'w', newline='') as csvDataFile:
    csvWriter = csv.writer(csvDataFile)
    print("Writing everything to a file")
    csvWriter.writerows(prediction_dict.items())
