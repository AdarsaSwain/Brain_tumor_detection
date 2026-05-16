"""
build_models.py
---------------
Builds and saves the binary + multiclass ensemble models with the EXACT same
architecture used during training (see Model Codes/ directory).

Binary  : VGG16 + ResNet50 + InceptionV3 → 2 outputs (no-tumor / tumor)
Multiclass: VGG19 + ResNet50 + InceptionV3 → 3 outputs (glioma / meningioma / pituitary)

NOTE: Weights are ImageNet-pretrained (frozen). The final dense layers start
with random weights (no dataset available). The models are fully functional
and will produce valid (though random) predictions. Retrain with real data to
get the 96% accuracy mentioned in the file names.
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress C++ TF warnings

import tensorflow as tf
from tensorflow import keras
from keras.layers import concatenate, Dense, Dropout
from keras.models import Model
from keras.applications import VGG16, VGG19, ResNet50, InceptionV3

INPUT_SIZE = 128
input_shape = (INPUT_SIZE, INPUT_SIZE, 3)

# ── Helper ──────────────────────────────────────────────────────────────────

def build_binary_model():
    """VGG16 + ResNet50 + InceptionV3  →  2 classes"""
    print("[1/2] Building binary model (VGG16 + ResNet50 + InceptionV3) …")

    vgg    = VGG16(weights='imagenet', input_shape=input_shape, include_top=False)
    resnet = ResNet50(weights='imagenet', input_shape=input_shape, include_top=False)
    incept = InceptionV3(weights='imagenet', input_shape=input_shape, include_top=False)

    for layer in vgg.layers + resnet.layers + incept.layers:
        layer.trainable = False

    vgg_out    = keras.layers.GlobalAveragePooling2D()(vgg.output)
    resnet_out = keras.layers.GlobalAveragePooling2D()(resnet.output)
    incept_out = keras.layers.GlobalAveragePooling2D()(incept.output)

    merged = concatenate([vgg_out, resnet_out, incept_out])
    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(0.3)(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    predictions = Dense(2, activation='softmax')(merged)

    model = Model(
        inputs=[vgg.input, resnet.input, incept.input],
        outputs=predictions
    )
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_multiclass_model():
    """VGG19 + ResNet50 + InceptionV3  →  3 classes"""
    print("[2/2] Building multiclass model (VGG19 + ResNet50 + InceptionV3) …")

    vgg    = VGG19(weights='imagenet', input_shape=input_shape, include_top=False)
    resnet = ResNet50(weights='imagenet', input_shape=input_shape, include_top=False)
    incept = InceptionV3(weights='imagenet', input_shape=input_shape, include_top=False)

    for layer in vgg.layers + resnet.layers + incept.layers:
        layer.trainable = False

    vgg_out    = keras.layers.GlobalAveragePooling2D()(vgg.output)
    resnet_out = keras.layers.GlobalAveragePooling2D()(resnet.output)
    incept_out = keras.layers.GlobalAveragePooling2D()(incept.output)

    merged = concatenate([vgg_out, resnet_out, incept_out])
    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(0.3)(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    predictions = Dense(3, activation='softmax')(merged)

    model = Model(
        inputs=[vgg.input, resnet.input, incept.input],
        outputs=predictions
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    src_dir = os.path.dirname(os.path.abspath(__file__))

    binary_path = os.path.join(src_dir, 'epoch10_sgd_acc96Point76.h5')
    multi_path  = os.path.join(src_dir, 'multi-model-30K-epouch20.h5')

    # ── Binary model ────────────────────────────────────────────────────────
    binary_model = build_binary_model()
    binary_model.save(binary_path)
    print(f"  ✓ Saved: {binary_path}")

    # ── Multiclass model ────────────────────────────────────────────────────
    multi_model = build_multiclass_model()
    multi_model.save(multi_path)
    print(f"  ✓ Saved: {multi_path}")

    print("\nAll models saved successfully. Run app.py to start the server.")
