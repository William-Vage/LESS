from run import encode, decode, clear_dir


def run_darpaoptc():
    clear_dir()
    encode(dataset='darpa_optc', edge_file='AIA-101-125-top-300000.csv', out_zip_file_name='darpaoptc_demo.zip')
    decode(dataset='darpa_optc', compressed_filename='darpaoptc_demo.zip')


run_darpaoptc()
