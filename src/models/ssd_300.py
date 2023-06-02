import math
import numpy as np
import tensorflow as tf

from models.vgg16 import VGG16
from utils.ssd_utils import dataset_to_ground_truth, get_coco_gt

Conv2D = tf.keras.layers.Conv2D
Sequential = tf.keras.models.Sequential
BatchNorm = tf.keras.layers.BatchNormalization

''' Our attempt at implementing our own SSD model. Compiles, but did not convolute. Ended up not using this model. '''

# Resources:
# primary: https://foundationsofdl.com/2020/11/07/ssd300-implementation/
# https://www.tensorflow.org/tutorials/images/transfer_learning
# https://github.com/Socret360/object-detection-in-keras/blob/master/networks/ssd_vgg16.py
# https://www.tensorflow.org/hub/tutorials/tf2_object_detection

class SSD300(tf.keras.Model):

    def __init__(self, num_class, input_shape):
        super(SSD300, self).__init__()
        
        # Create backbone
        self.vgg16 = VGG16(
            include_top=False, # drop classifier head
            weights="imagenet", # pre-trained on imagenet
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
            classifier_activation="softmax",
        )

        # Freeze backbone before training
        self.vgg16.trainable = False
        # self.vgg16.summary()
                
        ''' Input parameters '''
        self.num_class = num_class
        
        ''' Cone layers '''
        self.FC6_to_Conv6 = Conv2D(filters=1024, kernel_size=(3, 3), activation="relu", name="FC6_to_Conv6", padding="same", dilation_rate=6)
        self.FC7_to_Conv7 = Conv2D(filters=1024, kernel_size=(1, 1), activation="relu", name="FC7_to_Conv7", padding="same")
        self.Conv8_1      = Conv2D(filters=256,  kernel_size=(1, 1), activation="relu", name="Conv8_1")
        self.Conv8_2      = Conv2D(filters=512,  kernel_size=(3, 3), activation="relu", name="Conv8_2"     , padding="same", strides=(2, 2))
        self.Conv9_1      = Conv2D(filters=128,  kernel_size=(1, 1), activation="relu", name="Conv9_1")
        self.Conv9_2      = Conv2D(filters=256,  kernel_size=(3, 3), activation="relu", name="Conv9_2"     , padding="same", strides=(2, 2))
        self.Conv10_1     = Conv2D(filters=128,  kernel_size=(1, 1), activation="relu", name="Conv10_1")
        self.Conv10_2     = Conv2D(filters=256,  kernel_size=(3, 3), activation="relu", name="Conv10_2")
        self.Conv11_1     = Conv2D(filters=128,  kernel_size=(1, 1), activation="relu", name="Conv11_1")
        self.Conv11_2     = Conv2D(filters=256,  kernel_size=(3, 3), activation="relu", name="Conv11_2")

        ''' Confidence layers for each block '''
        self.stage_4_batch_norm = BatchNorm()
        self.stage_4_conf       = Conv2D(filters=4*num_class, kernel_size=(3, 3), padding="same", name="conf_stage4" )
        self.stage_7_conf       = Conv2D(filters=6*num_class, kernel_size=(3, 3), padding="same", name="conf_stage7" )
        self.stage_8_conf       = Conv2D(filters=6*num_class, kernel_size=(3, 3), padding="same", name="conf_stage8" )
        self.stage_9_conf       = Conv2D(filters=6*num_class, kernel_size=(3, 3), padding="same", name="conf_stage9" )
        self.stage_10_conf      = Conv2D(filters=4*num_class, kernel_size=(3, 3), padding="same", name="conf_stage10")
        self.stage_11_conf      = Conv2D(filters=4*num_class, kernel_size=(3, 3), padding="same", name="conf_stage11")
        self.stage_score_conv   = [self.stage_4_conf, self.stage_7_conf, self.stage_8_conf,self.stage_9_conf,self.stage_10_conf,self.stage_11_conf]
        
        ''' Localization layers for each block '''
        self.stage_4_loc  = Conv2D(filters=4*4, kernel_size=(3, 3), padding="same", name="loc_stage4" )
        self.stage_7_loc  = Conv2D(filters=6*4, kernel_size=(3, 3), padding="same", name="loc_stage7" )
        self.stage_8_loc  = Conv2D(filters=6*4, kernel_size=(3, 3), padding="same", name="loc_stage8" )
        self.stage_9_loc  = Conv2D(filters=6*4, kernel_size=(3, 3), padding="same", name="loc_stage9" )
        self.stage_10_loc = Conv2D(filters=4*4, kernel_size=(3, 3), padding="same", name="loc_stage10")
        self.stage_11_loc = Conv2D(filters=4*4, kernel_size=(3, 3), padding="same", name="loc_stage11")
        self.stage_offset_conv = [self.stage_4_loc, self.stage_7_loc, self.stage_8_loc, self.stage_9_loc, self.stage_10_loc,self.stage_11_loc]
        
        ''' Default boxes '''
        self.default_boxes, _ = self.get_default_boxes()
        self.stage_4_boxes    = self.default_boxes[0]
        self.stage_7_boxes    = self.default_boxes[1]
        self.stage_8_boxes    = self.default_boxes[2]
        self.stage_9_boxes    = self.default_boxes[3]
        self.stage_10_boxes   = self.default_boxes[4]
        self.stage_11_boxes   = self.default_boxes[5]

        ''' Sequences '''
        self.stage_4_seq = Sequential(self.vgg16.layers[:-5])        
        self.stage_5_seq = Sequential(self.vgg16.layers[-5:-1])

        ''' Loss utils '''
        self.before_mining_crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
        self.after_mining_crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
        self.smooth_l1 = tf.keras.losses.Huber(reduction='none', name='smooth_L1')
        
    def get_default_boxes(self):
        """
        Method to generate all default boxes for all the feature maps.
        There are 6 stages to output boxes so this method returns a list of
        size 6 with all the boxes:
            width feature maps * height feature maps * number of boxes per loc
        For instance with the stage 4: 38x38x4=5776 default boxes
        
        Return:
            - (list of tf.Tensor) boxes per stage, 4 parameters cx, cy, w, h
                [number of stage, number of default boxes per stage, 4]
            - (list of tf.Tensor) boxes, 4 parameters cx, cy, w, h
                [number of default boxes, 4]
        """
        
        ''' Box parameters '''
        self.ratios           = [[1, 1/2, 2],
                                 [1, 1/2, 2, 1/3, 3],
                                 [1, 1/2, 2, 1/3, 3],
                                 [1, 1/2, 2, 1/3, 3],
                                 [1, 1/2, 2],
                                 [1, 1/2, 2]]
        self.scales           = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
        self.fmap_resolutions = [38, 19, 10, 5, 3, 1]
        
        boxes_per_stage = []
        boxes = []
        for fm_idx in range(len(self.fmap_resolutions)):
            boxes_fm_i = []
            step = 1 / self.fmap_resolutions[fm_idx]
            for j in np.arange(0, 1, step):
                for i in np.arange(0, 1, step):
                    # box with scale 0.5
                    boxes_fm_i.append(
                        [
                            i + step / 2,
                            j + step / 2,
                            self.scales[fm_idx] / 2.0,
                            self.scales[fm_idx] / 2.0,
                        ]
                    )
                    boxes.append(
                        [
                            i + step / 2,
                            j + step / 2,
                            self.scales[fm_idx] / 2.0,
                            self.scales[fm_idx] / 2.0,
                        ]
                    )
                    # box with aspect ratio
                    for ratio in self.ratios[fm_idx]:
                        boxes_fm_i.append(
                            [
                                i + step / 2,
                                j + step / 2,
                                self.scales[fm_idx] / np.sqrt(ratio),
                                self.scales[fm_idx] * np.sqrt(ratio),
                            ]
                        )
                        boxes.append(
                            [
                                i + step / 2,
                                j + step / 2,
                                self.scales[fm_idx] / np.sqrt(ratio),
                                self.scales[fm_idx] * np.sqrt(ratio),
                            ]
                        )

            boxes_per_stage.append(tf.constant((boxes_fm_i)))
        return boxes_per_stage, tf.convert_to_tensor(boxes, dtype=tf.float32)

    def get_loss(self, score_preds, scores_gt, offset_preds, offset_gt):
        """
        Method to calculate loss for score and localization offsets

        Args:
            - (tf.Tensor) score prediction: [N boxes, classes]
            - (tf.Tensor) score ground truth:  [N boxes, classes]
            - (tf.Tensor) localization offsets prediction: [N boxes, 4]
            - (tf.Tensor) localization offsets ground truth: [N boxes, 4]

        Return:
            - (tf.Tensor) batch score of shape [1]
            - (tf.Tensor) batch loc of shape [1]
        """
        
        # Count ground truth boxes
        pos_boxes_mask = tf.reduce_sum(scores_gt, axis=-1) # will reduce a onehot ground truth score vector to 0 or 1
        n_img_pos      = tf.reduce_sum(pos_boxes_mask, axis=1) # will count boxes where gt_boxes_mask is 1
        n_img_pos      = tf.cast(n_img_pos, tf.float32) # required when comparing n_negative later
        pos_boxes_mask = tf.cast(pos_boxes_mask, tf.bool) # make into boolean mask
        
        # Debug
        # a = tf.reduce_sum(score_loss_ref)
        # print(a)
        # if (a.numpy() == 0):
        #     print("hi")

        # Negatives mining with <3:1 ratio for negatives:positives
        boxes_loss     = self.before_mining_crossentropy(scores_gt, score_preds) # (B, N)
        n_img_neg      = n_img_pos * 3
        loss_order     = tf.argsort(boxes_loss, axis=1, direction='DESCENDING') # sort; array = [4,2,7,1] -> order = [3,1,0,2]
        loss_rank      = tf.argsort(loss_order, axis=1)                         # rank; array = [4,2,7,1] -> rank  = [2,1,3,0]
        loss_rank      = tf.cast(loss_rank, tf.float32)
        neg_boxes_mask = loss_rank < tf.expand_dims(n_img_neg, 1)               # bool matrix where true if box rank is lower than n_img_neg
        boxes_mask     = tf.math.logical_or(pos_boxes_mask, neg_boxes_mask)

        # Calculate batch score loss on all selected boxes
        sel_scores_gt    = tf.ragged.boolean_mask(scores_gt, boxes_mask)
        sel_scores       = tf.ragged.boolean_mask(score_preds, boxes_mask)
        batch_score_loss = self.after_mining_crossentropy(sel_scores_gt, sel_scores)

        # Calculate offset loss on true boxes only
        sel_offset_gt     = tf.ragged.boolean_mask(offset_gt, pos_boxes_mask)
        sel_offsets       = tf.ragged.boolean_mask(offset_preds, pos_boxes_mask)
        batch_offset_loss = self.smooth_l1(sel_offset_gt, sel_offsets)
         
        # Normalize loss over objects in scene
        n_batch_pos = tf.reduce_sum(n_img_pos)
        if n_batch_pos != 0:
            batch_score_loss  = batch_score_loss  / n_batch_pos
            batch_offset_loss = batch_offset_loss / n_batch_pos

        # Combine score and offset loss into model loss
        batch_score_loss = tf.reduce_sum(batch_score_loss, axis=1)
        batch_offset_loss = tf.reduce_sum(batch_offset_loss, axis=1)
        alpha = 1
        model_loss = batch_score_loss + alpha * batch_offset_loss  # alpha equals 1
        return model_loss
        
    # Actual inference stage. Assume x is 1 datapoint. Use model(xs) to call on multiple inputs.
    def call(self, x):
        stage_scores = []
        stage_offsets = []

        def reshape(x, stage_idx):
            ''' Local function to reshape the output of the stage to the correct shape '''
            score_conv, offset_conv, n_boxes = (self.stage_score_conv[stage_idx], self.stage_offset_conv[stage_idx], self.default_boxes[stage_idx].shape[0])
            score = tf.keras.layers.Reshape((n_boxes, self.num_class))(score_conv(x))
            offset = tf.keras.layers.Reshape((n_boxes, 4))(offset_conv(x))
            return score, offset
        
        # stage 4
        x = self.stage_4_seq(x)
        x_normed = self.stage_4_batch_norm(x)
        score, offset = reshape(x_normed, 0)
        stage_scores.append(score)
        stage_offsets.append(offset)
    
        # stage 7
        x = self.stage_5_seq(x)
        x = self.FC6_to_Conv6(x)
        x = self.FC7_to_Conv7(x)
        score, offset = reshape(x, 1)
        stage_scores.append(score)
        stage_offsets.append(offset)
    
        # stage 8
        x = self.Conv8_1(x)
        x = self.Conv8_2(x)
        score, offset = reshape(x, 2)
        stage_scores.append(score)
        stage_offsets.append(offset)
    
        # stage 9
        x = self.Conv9_1(x)
        x = self.Conv9_2(x)
        score, offset = reshape(x, 3)
        stage_scores.append(score)
        stage_offsets.append(offset)

        # stage 10
        x = self.Conv10_1(x)
        x = self.Conv10_2(x)
        score, offset = reshape(x, 4)
        stage_scores.append(score)
        stage_offsets.append(offset)
    
        # stage 11
        x = self.Conv11_1(x)
        x = self.Conv11_2(x)
        score, offset = reshape(x, 5)
        stage_scores.append(score)
        stage_offsets.append(offset)
        
        scores  = tf.concat(stage_scores, axis=1)
        offsets = tf.concat(stage_offsets, axis=1)
        
        # DEBUG
        idx = tf.math.is_nan(scores)
        if tf.math.reduce_any((tf.math.is_nan(scores))) or len(scores[idx]) > 0:
            print("found nan")

        return scores, offsets