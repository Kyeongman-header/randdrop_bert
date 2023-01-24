import tensorflow as tf

class My_Bert_Score(tf.keras.Model):
    def __init__(self, model,seq_len):
        self.model=model
        self.dense=tf.keras.layers.Dense(seq_len) # Bert output(batch, seq_len, 768)
        # dense output (batch, seq_len)
        return
    def call(self, inputs,sep_token):
        bert_output=self.model(inputs)
        
        dense_output=self.dense(bert_output)

    
        return dense_output # dense_output은 이후 <pad>및 <SEP> 토큰에 대해서는 학습을 하지 않는다.
        # 나머지에 대해 1 OR 0의 학습을 한다.

class Categorical_Loss():
    def __init__(self,LAMBDA=1,PAD=0):
        super().__init__()
        self.loss_object=tf.keras.losses.CategoricalCrossentropy()
        self.LAMBDA=LAMBDA
        self.pad=PAD

    def cce_loss(self,logits, pred): # CUSTOM LOSS.
        return self.loss_object(logits, pred)