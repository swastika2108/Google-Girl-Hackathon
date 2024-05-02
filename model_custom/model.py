from model_custom.Encoder import EncoderBlock  
from model_custom.Decoder import DecoderBlock
import keras
from keras.layers import Conv2D, Input



SIZE = 128

def get_model():

    inputs= Input(shape=(SIZE,SIZE,3))

    # Contraction 
    p1, c1 = EncoderBlock(16,0.1)(inputs)
    p2, c2 = EncoderBlock(32,0.1)(p1)
    p3, c3 = EncoderBlock(64,0.2)(p2)
    p4, c4 = EncoderBlock(128,0.2)(p3)

    # Encoding Layer
    c5 = EncoderBlock(256,rate=0.3,pooling=False)(p4) 

    # Expansion
    d1 = DecoderBlock(128,0.2)([c5,c4]) # [current_input, skip_connection]
    d2 = DecoderBlock(64,0.2)([d1,c3])
    d3 = DecoderBlock(32,0.1)([d2,c2])
    d4 = DecoderBlock(16,0.1, axis=3)([d3,c1])

    # Outputs 
    outputs = Conv2D(1,1,activation='sigmoid')(d4)

    return keras.models.Model(
        inputs=[inputs],
        outputs=[outputs],
    )



