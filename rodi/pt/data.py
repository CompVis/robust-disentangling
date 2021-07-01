import os, glob, sys, re, tarfile
import h5py 
import numpy as np
from PIL import Image
import scipy.io
from natsort import natsorted
from edflow.iterators.batches import DatasetMixin, resize_float32 as resize
from edflow.util import retrieve
import edflow.datasets.utils as datautil
from edflow.data.dataset import SubDataset, RandomlyJoinedDataset, JoinedDataset, MultiJoinedDataset


def loadmat5(path):
    d = h5py.File(path, "r") 
    return d

def get_hdf5_ref(data, ref):
    name = h5py.h5r.get_name(ref, data.id)
    #return data[name].value
    return data[name][()]


class RawSprites(object):
    def __init__(self, path):
        self.path = datautil.get_root("sprites")
        if not datautil.is_prepared(self.path):
            fpath = datautil.download_url("nips2015-analogy-data.tar.gz",
                                          "http://www.scottreed.info/files/nips2015-analogy-data.tar.gz",
                                          self.path)
            datautil.unpack(fpath)
            datautil.mark_prepared(self.path)
        self.preprocess()

    def preprocess(self):
        self._data = dict()
        self.load_paths()
        n_characters = len(self.character_paths)
        n_frames = self.load_character_frames(0).shape[0]

        # prepare linear indexing mechanism
        self.keys = ["character_idx", "frame_idx"]
        self.dims = [n_characters, n_frames]
        # populate fields
        for k in self.keys:
            self._data[k] = list()
        for i in range(np.prod(self.dims)):
            indices = np.unravel_index(i, self.dims)
            for k_idx in range(len(self.dims)):
                self._data[self.keys[k_idx]].append(indices[k_idx])
        self.load_splits()

    def __len__(self):
        return np.prod(self.dims)

    def load_paths(self):
        p = os.path.join(self.path, "data/sprites", "sprites_*.mat")
        self.character_paths = natsorted(glob.glob(p))
        self.character_paths = [p for p in self.character_paths if re.search(r"""sprites_\d+.mat""", p)]

    def load_splits(self):
        p = os.path.join(self.path, "data/sprites", "sprites_splits.mat")
        data = scipy.io.loadmat(p)
        train_indices = data["trainidx"][0] - 1 # one based indexing in matlab
        # convert character indices to absolute indices
        self._data["train"] = len(self)*[False]
        for ti in train_indices:
            lis = [i for i in range(len(self)) if self._data["character_idx"][i] == ti]
            for li in lis:
                self._data["train"][li] = True

    def load_character_frames(self, character_idx):
        p = self.character_paths[character_idx]
        data = loadmat5(p)

        n_animations = data["sprites"].shape[0]
        assert n_animations == 21, n_animations
        # last one is an easter egg
        n_animations = 20
        # first 20 are 5 actions, each from 4 views

        all_frames = list()
        for animation_idx in range(n_animations):
            animation = data["sprites"][animation_idx, 0]
            frames = get_hdf5_ref(data, animation)
            frames = np.reshape(frames, [frames.shape[0], 3,60,60])
            frames = np.transpose(frames, [0,3,2,1])
            frames = 2.0*frames - 1.0 # rescale to [-1,1]
            all_frames.append(frames)
        all_frames = np.concatenate(all_frames, 0)
        return all_frames

    def load_frame(self, idx):
        indices = np.unravel_index(idx, self.dims)
        return self.load_character_frames(indices[0])[indices[1]]

    def load_key(self, k, idx):
        if k in ["character_idx", "frame_idx", "train"]:
            return self._data[k][idx]
        if k == "image":
            return self.load_frame(idx)
        raise KeyError(k)

    def load(self, idx):
        r = {"index": idx}
        for k in ["character_idx", "frame_idx", "train", "image"]:
            r[k] = self.load_key(k, idx)
        return r


class SpritesBase(DatasetMixin):
    def __init__(self, config):
        self.path = "./data/sprites/"
        self.train_split = self.use_train_split()
        assert self.train_split in [True, False]
        self.size = retrieve(config, "spatial_size", default=64)
        self.raw_data = RawSprites(self.path)

        # note that we only take even character indices. odd character indices
        # differ only in the weapon from the previous even character index.
        # since not all actions show the weapon, this leads to ambiguities.
        self.indices = [i for i in range(len(self.raw_data))
                if self.raw_data.load_key("train", i) == self.train_split
                and self.raw_data.load_key("character_idx", i) % 2 == 0]
        self._length = len(self.indices)
        self.labels = {
            "identity": np.array([
                self.raw_data.load_key("character_idx", i)
                for i in self.indices]),
            "frame_idx": np.array([
                self.raw_data.load_key("frame_idx", i)
                for i in self.indices])
        }

    def __len__(self):
        return self._length

    def get_example(self, i):
        raw_idx = self.indices[i]
        return {"image": resize(self.raw_data.load_key("image", raw_idx), self.size),
                "identity": self.labels["identity"][i],
                "frame_idx": self.labels["frame_idx"][i]}


class SpritesTrain(SpritesBase):
    def use_train_split(self):
        return True


class SpritesTest(SpritesBase):
    def use_train_split(self):
        return False


class SpritesPairsTrain(DatasetMixin):
    def __init__(self, config):
        self.data = RandomlyJoinedDataset({"RandomlyJoinedDataset": {
            "dataset": "rodi.data.SpritesTrain",
            "key": "identity",
            "balance": False,
            "avoid_identity": False}})


class SpritesPairsTest(DatasetMixin):
    def __init__(self, config):
        self.data = JoinedDataset(dataset=SpritesTest(config), key="identity", n_joins=2)


class SpritesTripletsTest(DatasetMixin):
    def __init__(self, config):
        data = MultiJoinedDataset(dataset=SpritesTest(config),
                                  keys_src=["identity", "frame_idx"])
        indices = np.random.RandomState(1).choice(len(data), size=8000)
        self.data = SubDataset(data, indices)


if __name__ == "__main__":
    d = RawSprites("./tmpdata/sprites/")
