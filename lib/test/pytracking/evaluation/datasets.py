from collections import namedtuple
import importlib
from lib.test.pytracking.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.pytracking.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    otb=DatasetInfo(module=pt % "otb", class_name="OTBDataset", kwargs=dict()),
    nfs=DatasetInfo(module=pt % "nfs", class_name="NFSDataset", kwargs=dict()),
    uav=DatasetInfo(module=pt % "uav", class_name="UAVDataset", kwargs=dict()),
    tpl=DatasetInfo(module=pt % "tpl", class_name="TPLDataset", kwargs=dict()),
    tpl_nootb=DatasetInfo(module=pt % "tpl", class_name="TPLDataset", kwargs=dict(exclude_otb=True)),
    vot=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict()),
    trackingnet=DatasetInfo(module=pt % "trackingnet", class_name="TrackingNetDataset", kwargs=dict()),
    trackingnetvos=DatasetInfo(module=pt % "trackingnet", class_name="TrackingNetDataset", kwargs=dict(vos_mode=True)),
    got10k_test=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='test')),
    got10k_val=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val')),
    got10k_ltrval=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='ltrval')),
    got10kvos_val=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val', vos_mode=True)),
    lasot=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict()),
    lasot_train=DatasetInfo(module=pt % "lasot", class_name="LaSOTTrainSequencesDataset", kwargs=dict()),
    lasot_ext=DatasetInfo(module=pt % "lasotextensionsubset", class_name="LaSOTExtensionSubsetDataset",
                                       kwargs=dict()),
    lasotvos=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict(vos_mode=True)),
)


def load_dataset(name: str, **kwargs):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs, **kwargs)  # Call the constructor
    return dataset


def get_dataset(*args, **kwargs):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name, **kwargs).get_sequence_list())
    return dset


def get_dataset_attributes(name, mode='short', **kwargs):
    """ Get a list of strings containing the short or long names of all attributes in the dataset. """
    dset = load_dataset(name , **kwargs)
    dsets = {}
    if not hasattr(dset, 'get_attribute_names'):
        dsets[name] = get_dataset(name)
    else:
        for att in dset.get_attribute_names(mode):
            dsets[att] = get_dataset(name, attribute=att)

    return dsets