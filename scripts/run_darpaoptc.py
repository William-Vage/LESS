from run import encode, decode, clear_dir


def run_darpaoptc():
    clear_dir()
    encode_time = encode(dataset='darpa_optc', edge_file='AIA-101-125-top-300000.csv',
                         out_zip_file_name='darpaoptc_demo.zip')
    decode_time = decode(dataset='darpa_optc', compressed_filename='darpaoptc_demo.zip')
    total_time = encode_time + decode_time
    print(f"\033[33m Total Time: {total_time:.1f}s \033[0m")


run_darpaoptc()
