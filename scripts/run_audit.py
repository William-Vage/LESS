from run import encode, decode, clear_dir


def run_audit():
    clear_dir()
    encode_time = encode(dataset='audit', edge_file='audit.json',
                         out_zip_file_name='audit_demo.zip')
    decode_time = decode(dataset='audit', compressed_filename='audit_demo.zip')
    total_time = encode_time + decode_time
    print(f"\033[33m Total Time: {total_time:.1f}s \033[0m")


run_audit()
