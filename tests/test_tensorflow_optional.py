"""Optional real TensorFlow tests, explicitly skipped if unavailable."""
import numpy as np
import pytest


def test_tensorflow_architecture_and_optimizer_persistence():
    tf=pytest.importorskip('tensorflow',reason='TensorFlow is unavailable; real training has not been tested.')
    from model import build_model
    from client import set_seed
    set_seed(100)
    model=build_model(64,6)
    assert [l.units for l in model.layers]==[128,128,64,32,16,6]
    initial=model.get_weights()
    x=np.zeros((12,64),dtype=np.float32);y=np.eye(6,dtype=np.float32)[np.arange(12)%6]
    model.fit(x,y,epochs=1,batch_size=6,shuffle=False,verbose=0)
    step=int(model.optimizer.iterations.numpy());assert step==2
    model.set_weights(initial)
    assert int(model.optimizer.iterations.numpy())==step
    model.fit(x,y,epochs=1,batch_size=6,shuffle=False,verbose=0)
    assert int(model.optimizer.iterations.numpy())==4
