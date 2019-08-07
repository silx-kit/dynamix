import os

def get_folder_path(foldername=""):
    _file_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = _file_dir
    return os.path.join(package_dir, foldername)

def get_opencl_srcfile(filename):
    src_relpath =  os.path.join("resources", "opencl")
    opencl_src_folder = get_folder_path(foldername = src_relpath)
    return os.path.join(opencl_src_folder, filename)
