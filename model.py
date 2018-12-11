from keras.utils import Sequence
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import csv
import cv2
import numpy as np
import os
import multiprocessing

# Generator class inspired by tutorial at 
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, data=None, data_path='/opt/carnd_p3/data/', batch_size=32, shuffle=True):
        'Initialization'
        #self.data = data
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lines = data
        #self.__get_lines(os.path.join(self.data_path,'driving_log.csv'))
        self.numImages = len(self.lines) * 6
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.numImages // self.batch_size
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        image_IDs_temp = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(image_IDs_temp)

        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.numImages)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, image_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        images = []
        measurements = []
        num_lines = len(self.lines)
        for imgID in image_IDs_temp:
            line_offset = imgID % num_lines
            image_offset = imgID // num_lines
            
            # Get the line
            line = self.lines[line_offset]
            
            # Get the image
            source_path = line[image_offset % 3]
            filename = source_path.split('/')[-1]
            imgpath = source_path.split('/')[-2].strip()
            current_path = os.path.join(*[self.data_path, imgpath, filename])
            image = cv2.imread(current_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            #image = image[:,:,::-1]
            
            # Get the measurement
            measurement = float(line[3])
            if image_offset == 1:
                measurement += 0.3
            elif image_offset == 2:
                measurement -= 0.3
                
            # If offset is 3, 4, 5; flip image and measurement
            if image_offset in range(3,6):
                image = cv2.flip(image, 1)
                measurement *= -1.0
                
            # Crop image
            image = self.__crop_image(image)
            
            # Append image and measurement to batch
            images.append(image)
            measurements.append(measurement)

        return np.array(images), np.array(measurements)
    
    def __crop_image(self, image):
        blur = cv2.GaussianBlur(image, (3,3), 0)
        crop = blur[50:140,:,:]
        resize = cv2.resize(crop, (200, 66), interpolation = cv2.INTER_AREA)
        
        return resize
    
    '''
    def __get_lines(self, driving_log):
        lines = []
        with open(driving_log) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
                
        return lines[1:]
    '''


def get_model(input_shape=(66,200,3)):
    
    _model = Sequential()
    
    #_model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
    _model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape))
    
    _model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001)))
    _model.add(ELU())
    _model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001)))
    _model.add(ELU())
    _model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001)))
    _model.add(ELU())
    
    _model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', kernel_regularizer=l2(0.001)))
    _model.add(ELU())
    _model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', kernel_regularizer=l2(0.001)))
    _model.add(ELU())
    
    _model.add(Flatten())
    
    _model.add(Dense(100, kernel_regularizer=l2(0.001)))
    _model.add(ELU())
    _model.add(Dense(50, kernel_regularizer=l2(0.001)))
    _model.add(ELU())
    _model.add(Dense(10, kernel_regularizer=l2(0.001)))
    _model.add(ELU())
    
    _model.add(Dense(1))
    
    return _model


def get_lines(driving_log):
    lines = []
    with open(driving_log) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    return lines[1:]


def train_test_split(data, ratio=0.8):
    train_indices = np.random.choice(len(data), int((len(data)*ratio)//1), replace=False)
    test_indices = np.setdiff1d(np.arange(len(data)), train_indices)
    
    train = []
    for i in train_indices:
        train.append(data[i])
        
    test = []
    for i in test_indices:
        test.append(data[i])
    
    return train, test


def main():
    # Select batch size
    batch_size = 64
    
    # Personal computer files
    #logfile = 'data/driving_log.csv'
    #datapath = 'data'
    
    logfiles = ['/opt/carnd_p3/data/driving_log.csv',
                '/opt/carnd_p3/data/driving_log_track1_reverse.csv',
                '/opt/carnd_p3/data/driving_log_track2_forward.csv']
    datapath = '/opt/carnd_p3/data'
    
    # Load in data
    data = []
    for logfile in logfiles:
        data += get_lines(logfile)
    
    training_data, remaining_data = train_test_split(data, 0.90)
    validation_data, testing_data = train_test_split(remaining_data, 0.10)
    
    # Create generator
    training_generator = DataGenerator(data=training_data, data_path=datapath, batch_size=batch_size)
    validation_generator = DataGenerator(data=validation_data, data_path=datapath, batch_size=batch_size)
    testing_generator = DataGenerator(data=testing_data, data_path=datapath, batch_size=batch_size)
    
    
    # Load model
    model = get_model((66,200,3))
    
    # Create model checkpoint
    filepath="model-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    
    # Compile model
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))

    
    # Train model
    model.fit_generator(generator=training_generator, 
                        steps_per_epoch=len(training_data)//batch_size,                        
                        epochs=100,
                        validation_data=validation_generator,
                        validation_steps=len(validation_data)//batch_size,
                        use_multiprocessing=True, 
                        workers=multiprocessing.cpu_count()-1, 
                        callbacks=callbacks_list)
    
if __name__ == "__main__":
    main()