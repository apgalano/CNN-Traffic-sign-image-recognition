import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt


working_dir = os.getcwd()
training_folder = 'GTSRB/Final_Training/Images/'
testing_folder = 'GTSRB/Online-Test/Images/'

class_num = 43
IMG_LEN = 30

# Read and process dataset
def process_images(image_length=IMG_LEN):
    # Starting with training data
    Images = []
    Targets = []
    os.chdir(working_dir+'/'+training_folder)
    for i in range(0,class_num):
        print('Processing class '+ str(i))
        os.chdir(str(i).zfill(5))
        img_cnt = 0
        labels = pd.read_csv('GT-'+str(i).zfill(5)+'.csv', delimiter=';')
        for filename in os.listdir():
            if filename.endswith('.ppm'):
                img = Image.open(filename)
                # crop image to include traffic sign
                img_cropped = img.crop((labels['Roi.X1'].iloc[img_cnt],labels['Roi.Y1'].iloc[img_cnt],labels['Roi.X2'].iloc[img_cnt],labels['Roi.Y2'].iloc[img_cnt]))
                img_resized = img_cropped.resize((image_length, image_length), Image.ANTIALIAS)
                img_ar = np.asarray(img_resized)
                img_ar_conv = img_ar.copy() / 255.
                Images.append(img_ar_conv)
                Targets.append(labels['ClassId'].iloc[img_cnt])
                img_cnt += 1
        os.chdir('..')
    os.chdir(working_dir)
    np.savez('Traffic_signs_train', inputs=Images, targets=Targets)
    
    # process test data
    Images = []
    os.chdir(working_dir+'/'+testing_folder)
    img_cnt = 0
    labels = pd.read_csv('GT-online_test.csv', delimiter=';')
    print('\nProcessing test images\n')
    for filename in os.listdir():
        if filename.endswith('.ppm'):
            img = Image.open(filename)
            # crop image to include traffic sign
            img_cropped = img.crop((labels['Roi.X1'].iloc[img_cnt],labels['Roi.Y1'].iloc[img_cnt],labels['Roi.X2'].iloc[img_cnt],labels['Roi.Y2'].iloc[img_cnt]))
            img_resized = img_cropped.resize((image_length, image_length), Image.ANTIALIAS)
            img_ar = np.asarray(img_resized)
            img_ar_conv = img_ar.copy() / 255.
            Images.append(img_ar_conv)
            img_cnt += 1
    os.chdir(working_dir)
    np.savez('Traffic_signs_test', inputs=Images)



# Training
def train_model(path='cnn_model', split=0.1, visualise=False):
    train_data = np.load('Traffic_signs_train.npz', allow_pickle=True)
    output_size = class_num
    img_len = train_data['inputs'].shape[1]
    train_data_inputs, train_data_targets = shuffle(train_data['inputs'], train_data['targets'], random_state=0)

    model = tf.keras.Sequential([tf.keras.layers.Conv2D(28, (3, 3), activation='relu', input_shape=(img_len,img_len,3)),
                             tf.keras.layers.MaxPooling2D((2, 2)),\
                             tf.keras.layers.Conv2D(28, (3, 3), activation='relu'),
                             tf.keras.layers.MaxPooling2D((2, 2)),\
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(output_size,activation='softmax')
        ])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    NUM_EPOCHS = 20
    train_progress = model.fit(train_data_inputs, train_data_targets, \
          validation_split=split, epochs=NUM_EPOCHS, validation_steps=10, \
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)], verbose=2)
    model.save(path)
    
    if visualise:
        acc = train_progress.history['accuracy']
        val_acc = train_progress.history['val_accuracy']
        loss = train_progress.history['loss']
        val_loss = train_progress.history['val_loss']
        
        fig1 = plt.figure()
        plt.grid()
        plt.plot(range(1,len(acc)+1), acc, label='Training')
        plt.plot(range(1,len(val_acc)+1), val_acc, label='Validation')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy', fontsize=16)
        plt.xlabel('Epoch', fontsize=16)
        plt.tight_layout()
        plt.show()
        fig1.savefig('Training_accuracy.jpg', format='jpg')
        
        fig2 = plt.figure()
        plt.yscale('log')
        plt.grid()
        plt.plot(range(1,len(loss)+1), loss, label='Training')
        plt.plot(range(1,len(val_loss)+1), val_loss, label='Validation')
        plt.legend(loc='upper right')
        plt.ylabel('Loss function', fontsize=16)
        plt.xlabel('Epoch', fontsize=16)
        plt.tight_layout()
        plt.show()
        fig2.savefig('Training_loss.jpg', format='jpg')
        
    return train_progress
        

# Testing
def test_model(path='cnn_model'):
    # load CNN model
    model = tf.keras.models.load_model(path)
    # load test data
    test_data = np.load('Traffic_signs_test.npz', allow_pickle=True)
    # Read sign names
    sign_names = pd.read_csv('signnames.csv')
    os.chdir(working_dir+'/'+testing_folder)
    inputs = test_data['inputs']
    predictions = model.predict(inputs)
    pred_classes = [np.argmax(output) for output in predictions]
    pred_signs = [sign_names['SignName'].loc[i] for i in pred_classes]
    labels = pd.read_csv('GT-online_test.csv', delimiter=';')
    df = labels.drop(['Width','Height','Roi.X1','Roi.Y1','Roi.X2','Roi.Y2'],axis=1)
    # Exporting images
    df['Sign'] = pred_signs
    os.chdir(working_dir)
    df.to_csv('Results.csv', index=False)

    return df       



def main():
    process_images()
    train_history = train_model()
    results = test_model()
    
if __name__ == "__main__":
    main()
    