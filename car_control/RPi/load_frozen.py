# NOTE:
# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
import tensorflow as tf

def load_graph(frozen_pb, prefix=""):
  with tf.gfile.GFile(frozen_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    # When a frozen graph is restored, the tensors
    # are accessed using:
    #   graph.get_tensor_by_name("prefix/name_scope/name:0")
    # for an example, see the main method below.
    tf.import_graph_def(graph_def, name=prefix)

    return graph
