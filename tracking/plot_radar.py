import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.pytracking.analysis.plot_results_new import print_results, print_results_per_attribute, plot_attributes_radar
from lib.test.pytracking.evaluation import get_dataset, get_dataset_attributes, trackerlist

trackers = []


# lasot trackers
trackers.extend(trackerlist(name='ROMTrack-384', parameter_name=None, dataset_name=None,
                            run_ids=None, display_name='ROMTrack-384', result_only=True))
trackers.extend(trackerlist(name='OSTrack-384', parameter_name=None, dataset_name=None,
                            run_ids=None, display_name='OSTrack-384', result_only=True))
trackers.extend(trackerlist(name='SimTrack-B16', parameter_name=None, dataset_name=None,
                            run_ids=None, display_name='SimTrack-B16', result_only=True))
trackers.extend(trackerlist(name='MixFormer-L', parameter_name=None, dataset_name=None,
                            run_ids=None, display_name='MixFormer-L', result_only=True))
trackers.extend(trackerlist(name='STARK-ST101', parameter_name=None, dataset_name=None,
                            run_ids=None, display_name='STARK-ST101', result_only=True))
trackers.extend(trackerlist(name='TransT', parameter_name=None, dataset_name=None,
                            run_ids=None, display_name='TransT', result_only=True))

attributes_detail = get_dataset_attributes('lasot', mode='long')
attributes_all = {"ALL": get_dataset('lasot')}
attributes_all.update(attributes_detail)
attributes = attributes_all

# print_results_per_attribute(trackers, get_dataset_attributes('lasot', mode='short'), 'lasot',
#                              merge_results=True, force_evaluation=False,
#                              skip_missing_seq=True,
#                              exclude_invalid_frames=False)

plot_attributes_radar(trackers,
                      attributes, 'lasot',
                      merge_results=True, force_evaluation=False,
                      skip_missing_seq=True,
                      plot_opts=None, exclude_invalid_frames=False)

'''
# lasot_ext trackers
trackers.extend(trackerlist(name='ROMTrack-384', parameter_name=None, dataset_name=None,
                            run_ids=None, display_name='ROMTrack-384', result_only=True))
trackers.extend(trackerlist(name='OSTrack-384', parameter_name=None, dataset_name=None,
                            run_ids=None, display_name='OSTrack-384', result_only=True))
trackers.extend(trackerlist(name='AiATrack', parameter_name=None, dataset_name=None,
                            run_ids=None, display_name='AiATrack', result_only=True))
trackers.extend(trackerlist(name='SwinTrack-B-384', parameter_name=None, dataset_name=None,
                            run_ids=range(0, 3), display_name='SwinTrack-B-384', result_only=True))
trackers.extend(trackerlist(name='ToMP50', parameter_name=None, dataset_name=None,
                            run_ids=None, display_name='ToMP', result_only=True))
trackers.extend(trackerlist(name='KeepTrack', parameter_name=None, dataset_name=None,
                            run_ids=range(0, 10), display_name='KeepTrack', result_only=True))

attributes_detail = get_dataset_attributes('lasot_ext', mode='long')
attributes_all = {"ALL": get_dataset('lasot_ext')}
attributes_all.update(attributes_detail)
attributes = attributes_all

plot_attributes_radar(trackers,
                      attributes, 'lasot_ext',
                      merge_results=True, force_evaluation=False,
                      skip_missing_seq=True,
                      plot_opts=None, exclude_invalid_frames=False)
'''

# print_results(trackers, get_dataset('lasot_ext'), 'lasot_ext', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))