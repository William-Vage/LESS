from run import encode, decode, clear_dir


def run_leonard():
    clear_dir()
    encode(dataset='leonard', edge_file='data/raw/edge200m.csv', vertex_file='data/raw/vertex200m.csv',
           out_zip_file_name='leonard_demo.zip')
    decode(dataset='leonard', compressed_filepathname='data/compress_result/leonard_demo.zip')


run_leonard()
