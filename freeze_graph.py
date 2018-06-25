# NOTE:
# This blog helped me write this.
# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-
# a-python-api-d4f3596b3adc
import tensorflow as tf
import os
import json
import common

'''
To use this module you'll need to define a dict with keys as defined in 'common'
and values corrisponding to tensor names (ie: my_tensor.name).
'''

'''
(1)
When your model is build, use this function to save
the dict to a json for easy recovery at a later time.
'''
def write_tensor_dict_to_json(save_dir, tensor_dict):
    path = os.path.join(save_dir, "tensor_names.json")
    print(f"tensor_dict: {tensor_dict}")
    with open(path, 'w') as f:
        json.dump(tensor_dict, f)
    print(f"tensor dict saved at {path}")
    return os.path.abspath(path)

'''
(2)
Once training is complete you can use this to freeze your
model at a desire checkpoint.

params:
graph_path: path to graph def (usually '.meta')
ckpt_path:  path to checkpoint of model (usually '.ckpt')
out_path: /where/do/you/want/to/save/to/frozen.pb
tensor_names_json: the path to the json writen in step (2)
'''
def freeze_meta(graph_path, ckpt_path, out_path, tensor_names_json):
    with open(tensor_names_json, 'r') as f:
        tensor_names = json.load(f)
    print(f"tensor_names: {tensor_names}")
    # remove :0 from tensor names. Freezing need this for some reason
    output_names = []
    output_names.append(tensor_names[common._STEERING_PREDICTION].split(":")[0])
    output_names.append(tensor_names[common._STEERING_PROBS].split(":")[0])
    output_names.append(tensor_names[common._THROTTLE_PREDICTION].split(":")[0])
    print(f"output_names: {output_names}")
    path = freeze_graph(graph_path, ckpt_path, out_path, output_names)
    return os.path.abspath(path)

def load_tensor_names(tensor_name_json):
    with open(tensor_name_json, 'r') as f:
        tensor_names = json.load(f)
    print(f"tensor names loaded:\n{tensor_names.keys()}")
    return tensor_names


'''
example usage:

  python freeze_graph.py --ckpt-path  path/to/some_checkpoint.ckpt
                         --graph-path path/to/corrisponding_graph.meta
                         --out-path   save/here/please.pb
                         --outputs    'name_scope/tensor_name1' 'op_name42'

  --outputs: Path to json containing tensor names
             A list of string corrisponding to the output tensors/ops defined
             in the model, the/slash/notation indicates a tensor resides
             within a tf.name_scope.
'''
def freeze_graph(graph_path, ckpt_path, out_path, output_names):
  graph = None
  with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(graph_path, clear_devices=True)
    sess.run( tf.global_variables_initializer())
    saver.restore(sess, ckpt_path)
 #   for op in sess.graph.get_operations():
 #     print(op)
    #graph = tf.get_default_graph()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
                         sess,
                         tf.get_default_graph().as_graph_def(),
                         output_names)
    with tf.gfile.GFile(out_path, "wb") as f:
      f.write(output_graph_def.SerializeToString())

  print("%d ops in the final graph." % len(output_graph_def.node))
  print("FROZEN graph at: {}".format(out_path))
  return out_path


if __name__ == "__main__":
  import argparse as argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt-path',  type=str, required=True)
  parser.add_argument('--graph-path', type=str, required=True)
  parser.add_argument('--out-path',   type=str, required=True)
  parser.add_argument('--tensor-json',    type=str, required=False,
                      help="json with tensor names")
  args = parser.parse_args()
  graph_path   = args.graph_path
  ckpt_path    = args.ckpt_path
  out_path     = args.out_path
  tensor_json  = args.tensor_json
  path = freeze_meta(graph_path, ckpt_path, out_path, tensor_json)
  print(f"frozen model saved at {path}")
