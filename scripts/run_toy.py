from run import encode, decode, clear_dir


def run_toy():
    clear_dir()
    encode(dataset='toy', edge_file='edge200m.csv', vertex_file='vertex200m.csv',
           out_zip_file_name='toy_demo.zip')
    decode(dataset='toy', compressed_filename='toy_demo.zip')


run_toy()
