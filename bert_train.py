import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = TFBertModel.from_pretrained("bert-base-uncased")


#consts.
batch_size=2
seq_len=1024
TRAIN_FILE="contrastive_dataset"
VALID_FILE="val_contrastive_dataset"
#위의 두개는 extractive 압축 데이터이다.
FURTHER_TRAIN=False
TRAIN_RANGE=100000
VALID_RANGE=1000

# file path.
filename="contrastive"
checkpoint_path="./MY_checkpoints/"+filename+"/bert_weight"


# load dataset.
npneg=np.load("./npdata/"+TRAIN_FILE +"/npneg.npy")
nppos=np.load("./npdata/"+TRAIN_FILE +"/nppos.npy")
nptoken_neg=("./npdata/"+TRAIN_FILE +"/nptoken_neg.npy")
nptoken_pos=("./npdata/"+TRAIN_FILE +"/nptoken_pos.npy")

vnppos=np.load("./npdata/"+VALID_FILE +"/nppos.npy")
vnpneg=np.load("./npdata/"+VALID_FILE +"/npneg.npy")
vnptoken_neg=("./npdata/"+ VALID_FILE+"/nptoken_neg.npy")
vnptoken_pos=("./npdata/"+ VALID_FILE+"/nptoken_pos.npy")

features, labels=(tf.concat([nptoken_neg,nptoken_pos],0),tf.concat([np.ones(shape=len(nptoken_neg)),np.ones(shape=len(nptoken_pos))],0))
dataset = tf.data.Dataset.from_tensor_slices((features,labels))
dataset = dataset.shuffle(TRAIN_RANGE, reshuffle_each_iteration=True)

features, labels=(tf.concat([vnptoken_neg,vnptoken_pos],0),tf.concat([np.ones(shape=len(vnptoken_neg)),np.ones(shape=len(vnptoken_pos))],0))
val_dataset = tf.data.Dataset.from_tensor_slices((features,labels))
val_dataset = dataset.shuffle(VALID_RANGE, reshuffle_each_iteration=True)


# make simple bert-dense model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(batch_size,seq_len,)))
model.add(bert)
model.add(tf.keras.layers.Dense(units=seq_len, activation='softmax'))
print(model.output_shape)

model.compile(loss='categorical_crossentropy', optimizer='adam')

#load weight.

if FURTHER_TRAIN:
    model.load_weights(checkpoint_path)


# checkpoint.
callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# train.
model.fit(dataset, epochs=50,validation_data=val_dataset,callbacks=[callback])


