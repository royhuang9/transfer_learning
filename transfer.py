from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.applications.vgg16 import preprocess_input 

import matplotlib.pyplot as plt
import json
import pickle

batch_size = 64

h,w,ch=32,32,3

def create_model():
    base_model = VGG16(input_tensor = Input(shape=(h,w,ch)), weights='imagenet',include_top=False)
    
    # Flatten after the pool5 from VGG16
    x = base_model.output
    x = Flatten(name='Flatten')(x)
    
    # 512 dense layer
    x = Dense(2048, activation='relu', name='fc1')(x)    
    # dropout
    x = Dropout(.5, name='drop1')(x)
    x = BatchNormalization()(x)

    # 512 dense layer
    x = Dense(2048, activation='relu', name='fc2')(x)
    x = Dropout(.5, name='drop2')(x)
    x = BatchNormalization()(x)

    
    # 10 classes
    predictions = Dense(10, activation='softmax', name='classifier')(x)

    model = Model(input=base_model.input, output=predictions)
    '''
    for layer in base_model.layers:
        layer.trainable = False
    '''
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    
    return model
    
if __name__ == '__main__':
    (X_train, y_train), (_, _) = cifar10.load_data()
    
    X_train = preprocess_input(X_train.astype('float'))
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    y_train_encoded = to_categorical(y_train)
    y_val_encoded = to_categorical(y_val)
    
    
    
    print('y_train_encoded shape : {}'.format(y_train_encoded.shape))
    print('y_val_encoded shape : {}'.format(y_val_encoded.shape))
 
    mymodel = create_model()
    
    print("Saving model weights and configuration file.")
    with open('./models/model.json', 'w') as outfile:
        json.dump(mymodel.to_json(), outfile)
        
    # store weight every epoch
    file_path = './models/weights.{epoch:03d}-{val_loss:.4f}.hdf5'
    save_model = ModelCheckpoint(file_path, verbose=1)

    # When the validation loss doesn't decrease, just stop training
    stop_it = EarlyStopping(min_delta=0.002, patience=5, verbose=1)
    
    tb = TensorBoard(log_dir='./logs')
    
    all_cbs = [stop_it, save_model, tb]
    history = mymodel.fit(X_train, 
                          y_train_encoded, 
                          batch_size=batch_size, 
                          nb_epoch=1000, 
                          validation_data=(X_val, y_val_encoded),
                          callbacks=all_cbs)
    
    loss_file='./logs/hist.log'
    with open(loss_file, 'wb') as lf:
        pickle.dump(history.history, lf, pickle.HIGHEST_PROTOCOL)
    
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.subplot(121)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    # summarize history for loss
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
