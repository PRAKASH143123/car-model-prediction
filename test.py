from __future__ import absolute_import, division, print_function

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


def load_graph(model_file):
    """
    Load pretrained graph
    """
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

def read_tensor(file_name, input_height=244, input_width=244, input_mean=0, input_std=255):
    input_name = "file_reader"
    try:
        file_reader = tf.read_file(file_name, input_name)
        image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)
        return result
    except Exception as e :
        print('Unknown error', e)
        exit

def load_labels(label_file):
    """Load tensorflow trained labeld"""
    label = []
    label_file = tf.gfile.GFile(label_file).readlines()
    for l in label_file:
        label.append(l.rstrip())
    return label


def label_id(labels, label_name):
    """Get label id"""
    return labels[labels['name'] == label_name.replace(' ', '')].iloc[0][['_id']].values.tolist()[0] + 1

if __name__ == "__main__":
    
    file_name = ""
    model_file = ()
    label_file = ""
    input_height = 244
    input_width = 244
    input_mean = 0
    input_std = 255
    input_layer = "input"
    output_layer = "    "
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", help="images to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--annotations", help="anootations ids")
    args = parser.parse_args()
    if args.graph:
        model_file = args.graph
    if args.images:
        file = args.images
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer
    if args.annotations:
        annotations = args.annotations
    
    ann = pd.read_csv(annotations)
    ann['_id'] = ann.index
    ann['name'] = ann.name.apply(lambda x : x.lower().replace('-', ' ' ).replace('/', ' ').replace('_', ' ').replace('.', ' ').replace(' ', ''))
    graph = load_graph(model_file)
    path = os.path.join(os.getcwd(), file) 
    files = [os.path.join(path, i) for i in os.listdir(path)]
    
    with tf.Session(graph=graph) as sess:
        res = []
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)
        labels = load_labels(label_file)
        for file_name in tqdm(files, total=len(files)):
            t = read_tensor(
                file_name,
                input_height=input_height,
                input_width=input_width,
                input_mean=input_mean,
                input_std=input_std,
            )
            results = sess.run(
                output_operation.outputs[0], {input_operation.outputs[0]: t}
            )
            results = np.squeeze(results)
            top_k = results.argsort()[-5:][::-1]
            tops = []
            for i in top_k:
                tops.append((results[i], labels[i]))
            top_label = sorted(tops, reverse=True)[0]
            res.append((file_name.split('/')[-1], label_id(ann, top_label[1])))
    res = pd.DataFrame(res, columns=['id', 'label'])
    res.to_csv('result.csv', index=False)
