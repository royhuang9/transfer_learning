from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.datasets import cifar10
import tensorflow as tf

batch_size = 64

h,w,ch=32,32,3

def create_model():
    input_tensor = Input(shape=(h,w,ch))
    
    base_model = VGG16(input_tensor = input_tensor, weights='imagenet',include_top=False)
    
    # Flatten after the pool5 from VGG16
    x = base_model.output
    x = Flatten()(x)
    
    # 512 dense layer
    x = Dense(512, activation='relu')(x)    
    # dropout
    x = Dropout(.5)(x)


    # 512 dense layer
    x = Dense(256, activation='relu')(x)
    x = Dropout(.5)(x)
    
    # 10 classes
    predictions = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', )
    print(model.summary())
    return model


def main(_):

    (X_train, y_train), (_, _) = cifar10.load_data()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
 
    model = create_model()
        
    hist = model.fit(X_train, X_val, batch_size=batch_size, validation_data=(y_train, y_val))
    
    print(hist)
    
if __name__ == '__main__':
    tf.app.run()
