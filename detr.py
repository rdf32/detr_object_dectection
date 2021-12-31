### https://github.com/Visual-Behavior/detr-tensorflow/tree/78fb71bca7b2ebf90e73151a51d29e8dbb46cb38

import tensorflow as tf
from loss import get_losses, xcycwh_to_xy_min_xy_max   
from models import *

def disable_batchnorm_training(model):
    for l in model.layers:
        if hasattr(l, "layers"):
            disable_batchnorm_training(l)
        elif isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = False

class DETR(tf.keras.Model):
    def __init__(self, num_classes=2, num_queries=100,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 return_intermediate_dec=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers

        self.backbone = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(None, None, 3))
        self.transformer = Transformer(
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                return_intermediate_dec=return_intermediate_dec,
                name='transformer'
        )
        
        self.model_dim = self.transformer.model_dim

        self.pos_encoder = PositionEmbeddingSine(
            num_pos_features=self.model_dim // 2, normalize=True, name="position_embedding_sine")

        self.input_proj = tf.keras.layers.Conv2D(self.model_dim, kernel_size=1, name='input_proj')

        self.query_embed = FixedEmbedding((self.num_queries, self.model_dim),
                                          name='query_embed')

        self.cls_layer = tf.keras.layers.Dense(num_classes, name="cls_layer")

        self.pos_layer = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(4, activation="sigmoid"),
        ], name="pos_layer")

    def downsample_masks(self, masks, x):
        masks = tf.cast(masks, tf.int32)
        masks = tf.expand_dims(masks, -1)
        masks = tf.compat.v1.image.resize_nearest_neighbor(masks, tf.shape(x)[1:3], align_corners=False, half_pixel_centers=False)
        masks = tf.squeeze(masks, -1)
        masks = tf.cast(masks, tf.bool)
        return masks
        
    def call(self, inp, training=False):
        
        x, masks = inp
        
        x = self.backbone(x, training=training)
        masks = self.downsample_masks(masks, x)

        pos_encoding = self.pos_encoder(masks)

        transformer_output = self.transformer(self.input_proj(x), masks, self.query_embed(None),
                            pos_encoding, training=training)[0]

        cls_preds = self.cls_layer(transformer_output)

        pos_preds = self.pos_layer(transformer_output)

        output = {'pred_logits': cls_preds[-1], 'pred_boxes': pos_preds[-1]}

        return output
    
    def build(self, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = [(None, None, None, 3), (None, None, None)]
        super().build(input_shape, **kwargs)
        
        disable_batchnorm_training(self.backbone)

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)

        self.detr_optimizer = optimizer
    
        self.detr_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.class_acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
        self.bb_giou = tf.keras.metrics.Mean(name="giou")

    @property
    def metrics(self):
        return [
            self.detr_loss_tracker,
            self.bb_giou,
            self.class_acc
        ]


    def train_step(self, data):

        images, t_bbox, t_class = data

        disable_batchnorm_training(self.backbone)

        with tf.GradientTape() as tape:
        
            x = self.backbone(images, training=True)
            
            masks = tf.zeros((tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]), tf.bool)
            
            masks = self.downsample_masks(masks, x)

            pos_encoding = self.pos_encoder(masks)

            transformer_output = self.transformer(self.input_proj(x), masks, self.query_embed(None),
                                pos_encoding, training=True)[0]

            cls_preds = self.cls_layer(transformer_output)

            pos_preds = self.pos_layer(transformer_output)

            output = {'pred_logits': cls_preds[-1], 'pred_boxes': pos_preds[-1]}
            
            loss, giou = get_losses(output, t_bbox, t_class)  

        gradients = tape.gradient(
            loss,

            # here we need to set batchnorm layers to not trainable in backbone
            self.backbone.trainable_weights + self.transformer.trainable_weights +
             self.cls_layer.trainable_weights + self.pos_encoder.trainable_weights,
        )
        self.detr_optimizer.apply_gradients(
            zip(
                gradients,
                self.backbone.trainable_weights + self.transformer.trainable_weights +
                 self.cls_layer.trainable_weights + self.pos_encoder.trainable_weights
            )
        )
        self.detr_loss_tracker.update_state(loss)
        self.class_acc.update_state(tf.expand_dims(tf.math.argmax(cls_preds[-1], axis=-1), axis=-1), t_class)
        self.bb_giou.update_state(giou)

        return {m.name: m.result() for m in self.metrics}



    def test_step(self, data):
        
        images, t_bbox, t_class = data

        x = self.backbone(images, training=False)
            
        masks = tf.zeros((tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]), tf.bool)

        masks = self.downsample_masks(masks, x)
        
        pos_encoding = self.pos_encoder(masks)

        transformer_output = self.transformer(self.input_proj(x), masks, self.query_embed(None),
                            pos_encoding, training=False)[0]

        cls_preds = self.cls_layer(transformer_output)

        pos_preds = self.pos_layer(transformer_output)

        output = {'pred_logits': cls_preds[-1], 'pred_boxes': pos_preds[-1]}
            
        loss, giou = get_losses(output, t_bbox, t_class) 
        
        self.detr_loss_tracker.update_state(loss)
        self.class_acc.update_state(tf.expand_dims(tf.math.argmax(cls_preds[-1], axis=-1), axis=-1), t_class)
        self.bb_giou.update_state(giou)

        return {m.name: m.result() for m in self.metrics}



detr_model = DETR(num_classes=2, num_queries=20)
detr_model.build()
detr_model.summary()
detr_model.compile(tf.keras.optimizers.Adam(learning_rate=1e-5))

epochs = 50
batch_size = 8
steps_per_epoch = len(train_dataset) // batch_size
val_steps_per_epoch = len(val_dataset) // batch_size


model_output = './' + f"detr_model.h5"

history = detr_model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    callbacks=tf.keras.callbacks.ModelCheckpoint(model_output, monitor='loss', mode='min', save_best_only=True, save_weights_only=True)
)


# def detr_model(num_classes=2, num_queries=100, num_encoder_layers=6, num_decoder_layers=6):

#     image_input = tf.keras.Input((None, None, 3))

#     x = backbone(image_input)

#     masks = tf.zeros((tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]), tf.bool)

#     pos_encoding = position_embedding_sine(masks)

#     transformer_output = transformer(input_proj(x), masks, query_embed((num_queries, transformer.model_dim)), pos_encoding)[0]

#     cls_layer = tf.keras.layers.Dense(num_classes, name="cls_layer")

#     cls_preds = cls_layer(transformer_output)
#     pos_preds = pos_layer(transformer_output)


#     output = {'pred_logits': cls_preds[-1], 'pred_boxes': pos_preds[-1]}

#     return tf.keras.Model(image_input, output, name="detr_finetuning")