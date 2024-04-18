from lib.test.pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data2/caiyidong/dataset/got10k_lmdb'
    settings.got10k_path = '/data2/caiyidong/dataset/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/data2/caiyidong/dataset/lasot_ext'
    settings.lasot_lmdb_path = '/data2/caiyidong/dataset/lasot_lmdb'
    settings.lasot_path = '/data2/caiyidong/dataset/lasot'
    settings.network_path = '/data2/caiyidong/ROMTrack/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data2/caiyidong/dataset/nfs'
    settings.otb_path = '/data2/caiyidong/dataset/OTB2015'
    settings.prj_dir = '/data2/caiyidong/ROMTrack'
    settings.result_plot_path = '/data2/caiyidong/ROMTrack/test/result_plots'
    settings.results_path = '/data2/caiyidong/ROMTrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data2/caiyidong/ROMTrack'
    settings.segmentation_path = '/data2/caiyidong/ROMTrack/test/segmentation_results'
    settings.tc128_path = '/data2/caiyidong/dataset/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/data2/caiyidong/dataset/trackingnet'
    settings.uav_path = '/data2/caiyidong/dataset/UAV123'
    settings.vot20_path = '/data2/caiyidong/dataset/vot2020'
    settings.vot_path = '/data2/caiyidong/dataset/VOT2019'
    settings.youtubevos_dir = ''

    return settings

