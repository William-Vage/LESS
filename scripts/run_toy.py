from run import encode, decode, clear_dir


def run_toy():
    clear_dir()
    encode(dataset='leonard', edge_file='edge200m.csv', vertex_file='vertex200m.csv',
           out_zip_file_name='leonard_demo.zip')
    decode(dataset='leonard', compressed_filename='toy_demo.zip')


run_toy()
