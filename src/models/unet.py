"""U-Net architecture for image segmentation."""
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPool2D,
    UpSampling2D, Concatenate, Dropout, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def conv_block(x, num_filters, dropout: float = 0.0):
    """Basic convolutional block: Conv2D -> BN -> ReLU -> Conv2D -> BN -> ReLU."""
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if dropout > 0:
        x = Dropout(dropout)(x)

    return x


def build_unet(
    input_size: tuple = (256, 256, 6),
    num_filters: list = None,
    dropout_enc: float = 0.5,
    dropout_dec: float = 0.5,
) -> Model:
    """
    Build U-Net model.

    Args:
        input_size: Input image size (height, width, channels)
        num_filters: List of filter counts for each encoder level
        dropout_enc: Dropout rate for encoder
        dropout_dec: Dropout rate for decoder

    Returns:
        Keras Model
    """
    if num_filters is None:
        num_filters = [16, 32, 48, 64]

    inputs = Input(shape=input_size)
    skip_connections = []
    x = inputs

    # Encoder path
    for f in num_filters:
        x = conv_block(x, f, dropout_enc)
        skip_connections.append(x)
        x = MaxPool2D((2, 2))(x)

    # Bridge
    x = conv_block(x, num_filters[-1], dropout_dec)

    # Reverse filter list for decoder
    num_filters_dec = num_filters[::-1]
    skip_connections = skip_connections[::-1]

    # Decoder path
    for i, f in enumerate(num_filters_dec):
        x = UpSampling2D((2, 2))(x)
        skip = skip_connections[i]

        # Handle dimension mismatch
        if x.shape[1:3] != skip.shape[1:3]:
            from tensorflow.keras.layers import Cropping2D
            h_diff = skip.shape[1] - x.shape[1]
            w_diff = skip.shape[2] - x.shape[2]
            if h_diff > 0 or w_diff > 0:
                skip = Cropping2D(
                    cropping=((0, h_diff), (0, w_diff))
                )(skip) if h_diff > 0 or w_diff > 0 else skip

        x = Concatenate()([x, skip])

        if dropout_dec > 0:
            x = Dropout(dropout_dec)(x)

        x = conv_block(x, f)

    # Output layer
    outputs = Conv2D(1, (1, 1), padding="same")(x)
    outputs = Activation("sigmoid")(outputs)

    model = Model(inputs, outputs, name="unet")
    return model


def compile_model(
    model: Model,
    learning_rate: float = 0.001,
    loss: str = "binary_crossentropy",
    metrics: list = None
) -> Model:
    """Compile model with optimizer and loss."""
    if metrics is None:
        metrics = ["accuracy"]

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    return model


def get_model_summary(model: Model) -> str:
    """Get model summary as string."""
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)


def count_parameters(model: Model) -> int:
    """Count trainable parameters."""
    return sum([tf.size(w).numpy() for w in model.trainable_weights])


def unet_small(input_size: tuple = (256, 256, 6)) -> Model:
    """Small U-Net for faster training."""
    return build_unet(
        input_size=input_size,
        num_filters=[16, 32, 64],
        dropout_enc=0.3,
        dropout_dec=0.3
    )


def unet_medium(input_size: tuple = (256, 256, 6)) -> Model:
    """Medium U-Net."""
    return build_unet(
        input_size=input_size,
        num_filters=[32, 64, 128],
        dropout_enc=0.4,
        dropout_dec=0.4
    )


def unet_large(input_size: tuple = (256, 256, 6)) -> Model:
    """Large U-Net."""
    return build_unet(
        input_size=input_size,
        num_filters=[64, 128, 256, 512],
        dropout_enc=0.5,
        dropout_dec=0.5
    )