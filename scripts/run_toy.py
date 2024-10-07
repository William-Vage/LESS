from run import encode, decode, clear_dir


def run_toy():
    clear_dir()
    encode_time = encode(dataset='toy', edge_file='edge200m.csv', vertex_file='vertex200m.csv',
                         out_zip_file_name='toy_demo.zip')
    decode_time = decode(dataset='toy', compressed_filename='toy_demo.zip')
    total_time = encode_time + decode_time
    print(f"\033[33m Total Time: {total_time:.1f}s \033[0m")


run_toy()
