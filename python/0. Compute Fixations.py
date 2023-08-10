import saliency
from saliency.model import TensorflowModel, IttyKoch
from saliency.data import load_video
import h5py

model = TensorflowModel(batch_size=20, check_point="deep_gaze/ICF.ckpt")

for i in [0,1,2]:
    X, y = load_video(i)
    S = model.predict(X, verbose=True)
    with h5py.File('icf.hdf5') as ds:
        ds[str(i)] = S

model = TensorflowModel(batch_size=20, check_point="deep_gaze/DeepGazeII.ckpt")

for i in [0,1,2]:
    X, y = load_video(i)
    S = model.predict(X, verbose=True)
    with h5py.File('icf.hdf5') as ds:
        ds[str(i)] = S

model = IttyKoch(n_jobs=4)

for i in [0,1,2]:
    print(i)
    X, y = load_video(i)
    S = model.predict(X)
    with h5py.File('/home/stes/media/saliency/ittykoch.hdf5') as ds:
        ds.create_dataset(str(i), data = S, compression = "gzip", compression_opts = 4)

