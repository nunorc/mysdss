
import tensorflow as tf

from .shared import CLASSES

def i2r():
    img = tf.keras.Input(shape=(150, 150, 3), name='img')
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(img)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    redshift = tf.keras.layers.Dense(1, activation='linear', name='redshift')(x)

    model = tf.keras.Model(inputs=img, outputs=redshift, name='i2r')

    return model

def f2r():
    fits = tf.keras.Input(shape=(61, 61, 5), name='fits')
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(fits)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    redshift = tf.keras.layers.Dense(1, activation='linear', name='redshift')(x)

    model = tf.keras.Model(inputs=fits, outputs=redshift, name='f2r')

    return model

def i2s():
    img = tf.keras.Input(shape=(150, 150, 3), name='img')
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(img)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    subclass = tf.keras.layers.Dense(len(CLASSES['subclass']), activation='softmax', name='subclass')(x)

    model = tf.keras.Model(inputs=img, outputs=subclass, name='i2s')

    return model

def i2g():
    img = tf.keras.Input(shape=(150, 150, 3), name='img')
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(img)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    gz2c = tf.keras.layers.Dense(len(CLASSES['gz2c']), activation='softmax', name='gz2c')(x)

    model = tf.keras.Model(inputs=img, outputs=gz2c, name='i2g')

    return model

def f2s():
    fits = tf.keras.Input(shape=(61, 61, 5), name='fits')
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(fits)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    subclass = tf.keras.layers.Dense(len(CLASSES['subclass']), activation='softmax', name='subclass')(x)

    model = tf.keras.Model(inputs=fits, outputs=subclass, name='f2s')

    return model

def f2g():
    fits = tf.keras.Input(shape=(61, 61, 5), name='fits')
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(fits)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    gz2c = tf.keras.layers.Dense(len(CLASSES['gz2c']), activation='softmax', name='gz2c')(x)

    model = tf.keras.Model(inputs=fits, outputs=gz2c, name='f2g')

    return model

def s2r(norm):
    spectra = tf.keras.Input(shape=(3522), name='spectra')
    x = norm(spectra)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    redshift = tf.keras.layers.Dense(1, activation='linear', name='redshift')(x)

    model = tf.keras.Model(inputs=spectra, outputs=redshift, name='s2r')

    return model

def ss2r(norm):
    ssel = tf.keras.Input(shape=(1423), name='ssel')
    x = norm(ssel)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    redshift = tf.keras.layers.Dense(1, activation='linear', name='redshift')(x)

    model = tf.keras.Model(inputs=ssel, outputs=redshift, name='ss2r')

    return model

def s2s(norm):
    spectra = tf.keras.Input(shape=(3522), name='spectra')
    x = norm(spectra)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    subclass = tf.keras.layers.Dense(len(CLASSES['subclass']), activation='softmax', name='subclass')(x)

    model = tf.keras.Model(inputs=spectra, outputs=subclass, name='s2s')

    return model

def s2g(norm):
    spectra = tf.keras.Input(shape=(3522), name='spectra')
    x = norm(spectra)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    gz2c = tf.keras.layers.Dense(len(CLASSES['gz2c']), activation='softmax', name='gz2c')(x)

    model = tf.keras.Model(inputs=spectra, outputs=gz2c, name='s2g')

    return model

def ss2s(norm):
    ssel = tf.keras.Input(shape=(1423), name='ssel')
    x = norm(ssel)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    subclass = tf.keras.layers.Dense(len(CLASSES['subclass']), activation='softmax', name='subclass')(x)

    model = tf.keras.Model(inputs=ssel, outputs=subclass, name='ss2s')

    return model

def ss2g(norm):
    ssel = tf.keras.Input(shape=(1423), name='ssel')
    x = norm(ssel)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    gz2c = tf.keras.layers.Dense(len(CLASSES['gz2c']), activation='softmax', name='gz2c')(x)

    model = tf.keras.Model(inputs=ssel, outputs=gz2c, name='ss2g')

    return model

def b2r(norm):
    bands = tf.keras.Input(shape=(5), name='bands')
    x = norm(bands)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    redshift = tf.keras.layers.Dense(1, activation='linear', name='redshift')(x)

    model = tf.keras.Model(inputs=bands, outputs=redshift, name='b2r')

    return model

def b2s(norm):
    bands = tf.keras.Input(shape=(5), name='bands')
    x = norm(bands)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    subclass = tf.keras.layers.Dense(len(CLASSES['subclass']), activation='softmax', name='subclass')(x)

    model = tf.keras.Model(inputs=bands, outputs=subclass, name='b2s')

    return model

def b2g(norm):
    bands = tf.keras.Input(shape=(5), name='bands')
    x = norm(bands)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    gz2c = tf.keras.layers.Dense(len(CLASSES['gz2c']), activation='softmax', name='gz2c')(x)

    model = tf.keras.Model(inputs=bands, outputs=gz2c, name='b2g')

    return model

def i2sm():
    img = tf.keras.Input(shape=(150, 150, 3), name='img')
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(img)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    smass = tf.keras.layers.Dense(1, activation='linear', name='smass')(x)

    model = tf.keras.Model(inputs=img, outputs=smass, name='i2sm')

    return model

def f2sm():
    fits = tf.keras.Input(shape=(61, 61, 5), name='fits')
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(fits)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    smass = tf.keras.layers.Dense(1, activation='linear', name='smass')(x)

    model = tf.keras.Model(inputs=fits, outputs=smass, name='f2sm')

    return model

def s2sm(norm):
    spectra = tf.keras.Input(shape=(3522), name='spectra')
    x = norm(spectra)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    sm = tf.keras.layers.Dense(1, activation='linear', name='sm')(x)

    model = tf.keras.Model(inputs=spectra, outputs=sm, name='s2sm')

    return model

def ss2sm(norm):
    ssel = tf.keras.Input(shape=(1423), name='ssel')
    x = norm(ssel)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    smass = tf.keras.layers.Dense(1, activation='linear', name='smass')(x)

    model = tf.keras.Model(inputs=ssel, outputs=smass, name='ss2sm')

    return model

def b2sm(norm):
    bands = tf.keras.Input(shape=(5), name='bands')
    x = norm(bands)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    smass = tf.keras.layers.Dense(1, activation='linear', name='smass')(x)

    model = tf.keras.Model(inputs=bands, outputs=smass, name='b2sm')

    return model

def w2r(norm):
    wise = tf.keras.Input(shape=(4), name='wise')
    x = norm(wise)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    redshift = tf.keras.layers.Dense(1, activation='linear', name='redshift')(x)

    model = tf.keras.Model(inputs=wise, outputs=redshift, name='w2r')

    return model

def w2sm(norm):
    wise = tf.keras.Input(shape=(4), name='wise')
    x = norm(wise)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    smass = tf.keras.layers.Dense(1, activation='linear', name='smass')(x)

    model = tf.keras.Model(inputs=wise, outputs=smass, name='b2sm')

    return model

def w2s(norm):
    wise = tf.keras.Input(shape=(4), name='wise')
    x = norm(wise)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    subclass = tf.keras.layers.Dense(len(CLASSES['subclass']), activation='softmax', name='subclass')(x)

    model = tf.keras.Model(inputs=wise, outputs=subclass, name='w2s')

    return model

def w2g(norm):
    wise = tf.keras.Input(shape=(4), name='wise')
    x = norm(wise)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    gz2c = tf.keras.layers.Dense(len(CLASSES['gz2c']), activation='softmax', name='gz2c')(x)

    model = tf.keras.Model(inputs=wise, outputs=gz2c, name='w2g')

    return model

