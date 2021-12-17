import platform
IS_LINUX = platform.system()=="Linux"


if IS_LINUX:

    config = {
        'BSDS': {'data_root': '/opt/dataset/BSDS/',
                    'data_lst': 'train_pair.lst',
                    'mean_bgr': [104.00699, 116.66877, 122.67892],
                    'yita': 0.5},
        'pascal_context': {'data_root': 'path_to/bsds500/PASCAL/',
                    'data_lst': 'voc_trainval.txt',
                    'mean_bgr': [104.00699, 116.66877, 122.67892],
                    'yita': 0.3},
        'bsds_pascal': {'data_root': 'path_to/bsds500/',
                    'data_lst': 'bsds_pascal_train_pair.lst',
                    'mean_bgr': [104.00699, 116.66877, 122.67892],
                    'yita': 0.5},
        'NYUDv2_HHA': {'data_root': 'path_to/NYUD/',
                       'data_lst': 'hha-train.lst',
                       'mean_bgr': [109.92, 88.24, 127.42],
                       'yita': 0.5},
        'NYUDv2_RGB': {'data_root': 'path_to/NYUD/',
                       'data_lst': 'image-train.lst',
                       'mean_bgr': [104.00699, 116.66877, 122.67892],
                       'yita': 0.5},
        'MDBD': {'data_root': '/opt/dataset/MDBD/',
                          'data_lst': 'train_pair.lst',
                          'mean_bgr': [104.00699, 116.66877, 122.67892],
                          'yita': 0.3},
        'MulticueBoundaries': {'data_root': 'path_to/multicue/',
                               'data_lst': 'train_boundaries_aug%d.lst',
                               'mean_bgr': [104.00699, 116.66877, 122.67892],
                               'yita': 0.4},
        'BIPED': {'data_root': '/opt/dataset/BIPED/edges/',
                               'data_lst': 'train_rgb.lst',
                               'mean_bgr': [104.00699, 116.66877, 122.67892],
                               'yita': 0.3}
    }

    config_test = {
        'BSDS': {'data_root': '/opt/dataset/BSDS/',
                    'data_lst': 'test_pair.lst',
                    'mean_bgr': [104.00699, 116.66877, 122.67892],
                    'yita': 0.5},
        'BSDS300': {'data_root': '/opt/dataset/BSDS300/',
                    'data_lst': 'test_pair.lst',
                    'mean_bgr': [104.00699, 116.66877, 122.67892],
                    'yita': 0.5},
        'PASCAL': {'data_root': '/opt/dataset/PASCAL/',
                    'data_lst': 'test_pair.lst',
                    'mean_bgr': [104.00699, 116.66877, 122.67892],
                    'yita': 0.3},
        'CLASSIC': {'data_root': 'data',
                    'data_lst': None,
                    'mean_bgr': [104.00699, 116.66877, 122.67892],
                    'yita': 0.5},

        'pascal_context_journal_val': {'data_root': 'path_to/bsds500/PASCAL/',
                    'data_lst': 'voc_validation_pair.lst',
                    'mean_bgr': [104.00699, 116.66877, 122.67892],
                    'yita': 1},
        'NYUDv2_HHA': {'data_root': 'path_to/NYUD/',
                       'data_lst': 'hha-test.lst',
                       'mean_bgr': [109.92, 88.24, 127.42],
                       'yita': 0.5},
        'NYUD': {'data_root': '/opt/dataset/NYUD/',
                       'data_lst': 'test_pair.lst',
                       'mean_bgr': [104.00699, 116.66877, 122.67892],
                       'yita': 0.5},
        'MDBD': {'data_root':'/opt/dataset/MDBD/',
                          'data_lst': 'test_pair.lst',
                          'mean_bgr': [104.00699, 116.66877, 122.67892],
                          'yita': 0.3},
        'MulticueBoundaries': {'data_root': 'path_to/multicue/',
                               'data_lst': 'test%d.lst',
                               'mean_bgr': [104.00699, 116.66877, 122.67892],
                               'yita': 0.4},
        'BIPED': {'data_root': '/opt/dataset/BIPED/edges/',
                               'data_lst': 'test_rgb.lst',
                               'mean_bgr': [104.00699, 116.66877, 122.67892],
                               'yita': 0.5},
        'CID': {'data_root': '/opt/dataset/CID/',
                               'data_lst': 'test_pair.lst',
                               'mean_bgr': [104.00699, 116.66877, 122.67892],
                               'yita': 0.5},
        'DCD': {'data_root': '/opt/dataset/DCD/',
                               'data_lst': 'test_pair.lst',
                               'mean_bgr': [104.00699, 116.66877, 122.67892],
                               'yita': 0.5}
        }
else:
    config = {
        'BSDS': {'data_root': 'C:/Users/xavysp/dataset/BSDS/',
                 'data_lst': 'train_pair.lst',
                 'mean_bgr': [104.00699, 116.66877, 122.67892],
                 'yita': 0.5},
        'pascal_context': {'data_root': 'path_to/bsds500/PASCAL/',
                           'data_lst': 'voc_trainval.txt',
                           'mean_bgr': [104.00699, 116.66877, 122.67892],
                           'yita': 0.3},
        'bsds_pascal': {'data_root': 'path_to/bsds500/',
                        'data_lst': 'bsds_pascal_train_pair.lst',
                        'mean_bgr': [104.00699, 116.66877, 122.67892],
                        'yita': 0.5},
        'NYUDv2_HHA': {'data_root': 'path_to/NYUD/',
                       'data_lst': 'hha-train.lst',
                       'mean_bgr': [109.92, 88.24, 127.42],
                       'yita': 0.5},
        'NYUDv2_RGB': {'data_root': 'path_to/NYUD/',
                       'data_lst': 'image-train.lst',
                       'mean_bgr': [104.00699, 116.66877, 122.67892],
                       'yita': 0.5},
        'MDBD': {'data_root': '/opt/dataset/MDBD/',
                 'data_lst': 'train_pair.lst',
                 'mean_bgr': [104.00699, 116.66877, 122.67892],
                 'yita': 0.3},
        'MulticueBoundaries': {'data_root': 'path_to/multicue/',
                               'data_lst': 'train_boundaries_aug%d.lst',
                               'mean_bgr': [104.00699, 116.66877, 122.67892],
                               'yita': 0.4},
        'BIPED': {'data_root': 'C:/Users/xavysp/dataset/BIPED/edges/',
                  'data_lst': 'train_rgb.lst',
                  'mean_bgr': [104.00699, 116.66877, 122.67892],
                  'yita': 0.3}
    }

    config_test = {
        'BSDS': {'data_root': 'C:/Users/xavysp/dataset/BSDS/',
                 'data_lst': 'test_pair.lst',
                 'mean_bgr': [104.00699, 116.66877, 122.67892],
                 'yita': 0.5},
        'BSDS300': {'data_root': '/opt/dataset/BSDS300/',
                    'data_lst': 'test_pair.lst',
                    'mean_bgr': [104.00699, 116.66877, 122.67892],
                    'yita': 0.5},
        'PASCAL': {'data_root': '/opt/dataset/PASCAL/',
                   'data_lst': 'test_pair.lst',
                   'mean_bgr': [104.00699, 116.66877, 122.67892],
                   'yita': 0.3},
        'CLASSIC': {'data_root': 'data',
                    'data_lst': None,
                    'mean_bgr': [104.00699, 116.66877, 122.67892],
                    'yita': 0.5},

        'pascal_context_journal_val': {'data_root': 'path_to/bsds500/PASCAL/',
                                       'data_lst': 'voc_validation_pair.lst',
                                       'mean_bgr': [104.00699, 116.66877, 122.67892],
                                       'yita': 1},
        'NYUDv2_HHA': {'data_root': 'path_to/NYUD/',
                       'data_lst': 'hha-test.lst',
                       'mean_bgr': [109.92, 88.24, 127.42],
                       'yita': 0.5},
        'NYUD': {'data_root': '/opt/dataset/NYUD/',
                 'data_lst': 'test_pair.lst',
                 'mean_bgr': [104.00699, 116.66877, 122.67892],
                 'yita': 0.5},
        'MDBD': {'data_root': 'C:/Users/xavysp/dataset/MDBD/',
                 'data_lst': 'test_pair.lst',
                 'mean_bgr': [104.00699, 116.66877, 122.67892],
                 'yita': 0.3},
        'MulticueBoundaries': {'data_root': 'path_to/multicue/',
                               'data_lst': 'test%d.lst',
                               'mean_bgr': [104.00699, 116.66877, 122.67892],
                               'yita': 0.4},
        'BIPED': {'data_root': 'C:/Users/xavysp/dataset/BIPED/edges/',
                  'data_lst': 'test_rgb.lst',
                  'mean_bgr': [104.00699, 116.66877, 122.67892],
                  'yita': 0.5},
        'CID': {'data_root': 'C:/Users/xavysp/dataset/CID/',
                'data_lst': 'test_pair.lst',
                'mean_bgr': [104.00699, 116.66877, 122.67892],
                'yita': 0.5},
        'DCD': {'data_root': '/opt/dataset/DCD/',
                'data_lst': 'test_pair.lst',
                'mean_bgr': [104.00699, 116.66877, 122.67892],
                'yita': 0.5}
    }

if __name__ == '__main__':
    print (config.keys())
