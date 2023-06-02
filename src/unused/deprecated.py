import json
from matplotlib import pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from utils.ssd_utils import dataset_to_ground_truth
from utils.tf_utils import tensor_set_equal


def create_oracle_model(trained_model, shape=(300, 300, 3)):
    ''' DEPRECATED. Use Oracle class instead.'''

    _, df_bx = trained_model.get_default_boxes()

    # TODO: see more weights?
    scores_input = tf.keras.Input(shape=(len(df_bx), 80), name="scores_input")
    offset_input = tf.keras.Input(shape=(len(df_bx), 4), name="offset_input")
    outputs      = tf.keras.layers.concatenate([scores_input, offset_input])    # (None, 8732, 84)
    outputs      = tf.keras.layers.Flatten()(outputs)                           # (None, 733488)
    outputs      = tf.keras.layers.Dense(1, activation='sigmoid')(outputs)      # (None, 1)

    regression_head = tf.keras.Model(
        inputs=[scores_input, offset_input], outputs=outputs, name="regression_head"
    )

    # Freeze trained_model. Freeze part of it?
    trained_model.trainable = False

    # Create new oracle model with a loss prediction head
    inputs  = tf.keras.Input(shape=shape, name="image_input")
    x       = trained_model(inputs)
    outputs = regression_head(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="oracle_model")

def get_loss_from_model_loss(dataset, model, alpha = 1):
    ''' DEPRECATED - SSD loss now returns loss per image, not per batch.'''

    _, df_bx = model.get_default_boxes()
    imgs, scores, offsets, _ = dataset_to_ground_truth(dataset, df_bx)

    p_scores, p_offsets = model(imgs)

    losses = []
    for (p_s, s, p_o, o) in zip(p_scores, scores, p_offsets, offsets):
        score_loss, offset_loss = model.calc_batch_loss(
            tf.expand_dims(p_s, axis=0), 
            tf.expand_dims(s, 0), 
            tf.expand_dims(p_o, 0), 
            tf.expand_dims(o, 0)
        )
        loss = score_loss + alpha * offset_loss 
        losses.append(loss)
    
    return imgs, tf.convert_to_tensor(losses)

def get_loss(dataset, model):
    ''' DEPRECATED - SSD loss now returns loss per image, not per batch.'''
    
    batch_imgs=[]
    batch_losses=[]
    _, df_bx = model.get_default_boxes()
    
    for batch in tqdm(dataset, desc="Calculating loss"):
        
        imgs, scores, offsets, _ = dataset_to_ground_truth(batch, df_bx)
        p_scores, p_offsets = model(imgs)

        score_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='sum')
        offset_loss = tf.keras.losses.Huber(reduction='sum', name='smooth_L1')

        loss = [score_loss(s, ps) + offset_loss(o, po) for (s, ps, o, po) in zip(scores, p_scores, offsets, p_offsets)]
        batch_losses.append(loss)
        batch_imgs.append(imgs)
        
    return tf.concat(batch_imgs, axis=0), tf.concat(batch_losses, axis=0)

def plot_histories(learn_file, train_file):
    ''' DEPRECATED - Use plot_history_files instead.'''
    learn_history = []
    train_history = []
    info = None
    
    with open(learn_file, "r") as file1:
        learn_history = file1.readlines()
        learn_history = [s.strip() for s in learn_history]
        info = learn_history[0]
        learn_history = learn_history[1:]
        learn_history = [json.loads(s) for s in learn_history]
    
    with open(train_file, "r") as file2:
        train_history = file2.readlines()
        train_history = [s.strip() for s in train_history]
        train_history = train_history[1:]
        train_history = [json.loads(s) for s in train_history]
    
    info = info.replace(' ','').split(',')
    n_epoch = int(info[2])
    print(learn_history, train_history, info, n_epoch)
    
    plt.figure('Eval_Loss')
    for i in range(len(learn_history)):
        plt.plot([i], [learn_history[i][0]], marker='o')
    plt.show()
    
    plt.figure('Eval_Accuracy')
    for i in range(len(learn_history)):
        plt.plot([i], [learn_history[i][1]], marker='x')
    plt.show()
    
    plt.figure('Train_Loss')
    for i in train_history:
        plt.plot(range(n_epoch), i)
    plt.show()
    
def plot_together(learn_files):
    ''' DEPRECATED - Use plot_history_files instead.'''
    plt.figure(num=1)
    plt.xlabel('AL Loops')
    plt.ylabel('Loss')
    plt.title('Eval_Loss')
    plt.figure(num=2)
    plt.xlabel('AL Loops')
    plt.ylabel('Accuracy')
    plt.title('Eval_Accuracy')
    
    for learn_file in learn_files:
        learn_history = []
        info = None    
        
        with open(learn_file, "r") as file1:
            learn_history = file1.readlines()
            learn_history = [s.strip() for s in learn_history]
            info = learn_history[0]
            learn_history = [json.loads(s) for s in learn_history[1:]]
        file1.close()
        
        info = info.replace(' ','').split(',')
        prio_func = info[-1]
        info = [int(elem) for elem in info[:-1]]
        
        plt.figure(1)
        points = [sublist[0] for sublist in learn_history]
        plt.plot(range(len(points)), points, 'o-', label=prio_func)
        
        plt.figure(2)
        points = [sublist[1] for sublist in learn_history]
        plt.plot(range(len(points)), points, 'x-', label=prio_func)
        
    plt.legend(loc='lower right')   
    plt.figure(1) 
    plt.legend(loc='upper right')
    plt.show()
    
# Validate same data is used at the end of all runs
def validate_seen_data(dataset, n_test, n_unseen, seen_data_list):
    a = [xy[0] for xy in seen_data_list[0]] # get images
    b = [xy[0] for xy in seen_data_list[1]] # get images
    c = [xy[0] for xy in dataset.skip(n_test*2).take(n_unseen)]

    assert tensor_set_equal(a, b)
    assert tensor_set_equal(a, c)
    assert tensor_set_equal(b, c)
    
    
# vgg16.evaluate(dataset) # Comment out if you want to use code below
# def test():
#     a = next(iter(dataset))

#     label_names = open("dataset_labels/image_net_class_labels.txt").read().splitlines()
#     for data in dataset.take(10):
#         # print(data)
#         y_true = data[1]
#         y_pred = vgg16(tf.expand_dims(data[0], axis=0))
#         y_pred = tf.keras.applications.vgg16.decode_predictions(y_pred.numpy(), top=5)
#         y_pred = y_pred[0]
#         y_true = y_true.numpy()
#         true_name = label_names[y_true].split(" ")[0]
#         predictions = list(map(lambda x : x[0], y_pred))
#         print(predictions)
#         print("top guess:", y_pred[0][1], "true:", label_names[y_true].split(" ")[2], "match", (true_name in predictions))

#     print("finished")
