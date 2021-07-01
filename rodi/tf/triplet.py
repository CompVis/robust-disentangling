import sys, os
sys.path.append(os.environ["TRIPLET_PATH"])
from triplet_reid.edflow_implementations.implementations import make_network as make_triplet_net
import tensorflow as tf

def triplet_wrapper(edim):
    name = 'my_triplet_is_the_best_triplet'
    def tnet(images):
        endpoints, _ = make_triplet_net(images, edim = edim, name = name)
        emb = endpoints["emb"]
        emb = tf.expand_dims(emb, axis = 1)
        emb = tf.expand_dims(emb, axis = 1)
        return emb
    return tnet, name
