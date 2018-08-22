from cifar10 import dataset


def make_iterator(ds, batch_size=32, shuffle_size=1000):
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    ds = ds.shuffle(shuffle_size)
    return ds.make_one_shot_iterator()
