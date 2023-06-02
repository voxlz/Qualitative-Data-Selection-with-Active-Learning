import numpy as np
from PIL import Image
from training_pipeline.etl.datapipeline import DataPipeline
from training_pipeline.tools.optopsy_evaluation import evaluate_model
from training_pipeline.utils.predictor import Predictor

def reverse_encoded_image(data):
    '''reverse preprocessing of encoded od images'''
    arr = data[0].numpy()
    arr += 1
    arr *= 127.5
    arr = np.squeeze(arr).astype(np.uint8)
    return Image.fromarray(arr)

def reverse_raw_image(data):
    '''reverse preprocessing of encoded od images'''
    arr = data['image']['raw'].numpy() 
    arr += 1
    arr *= 127.5
    arr = np.squeeze(arr).astype(np.uint8)
    img = Image.fromarray(arr)
    return img

def evaluation_fn(config, model, pipeline_mode, path):
    return evaluate_model(
                config,
                DataPipeline(config.DATASET, config.ETL),
                Predictor(model=model.predict_on_batch, post_process=config.POST_PROCESSING.TEST),
                path,
                pipeline_mode=pipeline_mode,
                epoch=None,
                tag=None,
                session_id=None,
                upload=False,
            )
