
if __name__ == '__main__':
    import json
    import pandas as pd
    import sys
    import warnings
    from model_validations import train_test_split
    from utils import read_columns

    warnings.simplefilter('ignore')

    infiles = sys.argv[1].split(',')
    inputs_path = sys.argv[2]
    if len(sys.argv) > 3:
        labels_file = sys.argv[3]

    with open(inputs_path, 'r') as f:
        params = json.load(f)

    headers = []
    arrays = []
    for i, infile_path in enumerate(infiles):
        header = 'infer' if params['infile_arrays'][i]['header'] else None
        df = pd.read_csv(infile_path, sep='\t', header=header)
        headers.append(df.columns)
        arrays.append(df.values)

    options = params['options']
    shuffle_selection = options.pop('shuffle_selection')
    options['shuffle'] = shuffle_selection['shuffle']
    if options['shuffle'] == 'None':
        options['shuffle'] = None

    if options['shuffle'] in [None, 'simple']:
        splits = train_test_split(*arrays, **options)

    else:
        header = 'infer' if shuffle_selection['header'] else None
        col = shuffle_selection['col']
        labels = read_columns(
                 labels_file,
                 c = col,
                 sep='\t',
                 header=header,
                 parse_dates=True)
        labels = labels.ravel()
        options['labels'] = labels
        print(labels)

        splits = train_test_split(*arrays, **options)

    for i, arr in enumerate(splits):
        arr_index = i // 2
        df = pd.DataFrame(arr, columns=headers[arr_index])
        if i % 2 == 0:
            df.to_csv('./file%d_train.tabular' % (arr_index + 1),
                      sep='\t', index=False,
                      header=True if params['infile_arrays'][arr_index]['header'] else False)
        else:
            df.to_csv('./file%d_test.tabular' % (arr_index + 1),
                      sep='\t', index=False,
                      header=True if params['infile_arrays'][arr_index]['header'] else False)
