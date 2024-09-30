from run import encode, decode, clear_dir


def run_darpatc():
    clear_dir()
    encode(dataset='darpa_tc', edge_file='data/raw/ta1-trace-3-e5-official-1.bin.1_concatenated_edges_top_300000.csv',
           vertex_file='data/raw/ta1-trace-3-e5-official-1.bin.1_concatenated_vertices_top_300000.csv',
           out_zip_file_name='darpatc_demo.zip')
    decode(dataset='darpa_tc', compressed_filepathname='data/compress_result/darpatc_demo.zip')


run_darpatc()
