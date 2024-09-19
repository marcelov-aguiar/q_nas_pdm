import subprocess
import sys
from ds_library.constants import data_path
import ds_library.constants.constants_names as const_name
from ds_library.utils.get_files_in_folder import GetFilesInFolder

def run_controller(script_name):
    print(f"Running {script_name}...")
    python_executable = sys.executable
    subprocess.run([python_executable, script_name])

if __name__ == "__main__":
    controllers_path = GetFilesInFolder().get_name_files(str(data_path.CONTROLLERS_PATH))
    
    for controller in controllers_path:
        if controller != const_name.UTIL_FILE_NAME:
            run_controller(str(data_path.CONTROLLERS_PATH.joinpath(controller)))
