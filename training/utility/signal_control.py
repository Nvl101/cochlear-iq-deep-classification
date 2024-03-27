'''
control training process using signal file
functionalities:
1. stop at next epoch
2. pause training at next dataload
3. immediate stop if signal file deleted
    but the training will be discarded
4. to solve read conflict, continue training if the 
'''
import os
import tempfile

DEFAULT_DIR = os.path.join(__file__, '..', '..', '_signal')

class SignalFileControl:
    temp_file_path: str
    def __init__(self):
        # create tempfile under file_path
        temp_file_path = tempfile.mktemp(dir=DEFAULT_DIR, prefix='signal-')
        self.temp_file_path = temp_file_path
        with open(self.temp_file_path, 'w') as file:
            file.write('')
        print(f'signal file path: {temp_file_path}')
    def _create_file(self):
        # TODO: make file template, then write to file with this module
        ...
    def _check_status(self):
        ...
    def is_alive(self):
        return os.path.isfile(self.temp_file_path)
    def reset(self):
        if os.path.isfile(self.temp_file_path):
            os.remove(self.temp_file_path)
