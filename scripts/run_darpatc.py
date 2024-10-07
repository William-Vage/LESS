from run import encode, decode, clear_dir


def run_darpatc():
    clear_dir()
    encode_time = encode(dataset='darpa_tc', edge_file='ta1-trace-3-e5-official-1.bin.1_concatenated_edges_top_300000.csv',
           vertex_file='ta1-trace-3-e5-official-1.bin.1_concatenated_vertices_top_40000.csv',
           out_zip_file_name='darpatc_demo.zip')
    decode_time = decode(dataset='darpa_tc', compressed_filename='darpatc_demo.zip')
    total_time = encode_time + decode_time
    print(f"\033[33m Total Time: {total_time:.1f}s \033[0m")


run_darpatc()
