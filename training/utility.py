'''
utility classes and methods for training
- progress bar
- signal file controlled stopper
'''
import os
import tempfile


class ProgressBar:
    '''
    print progress of batches inside an epoch

    [###    ] 15/40 training
    '''
    current_count: int = 0
    total_count: int = -1
    width: int = 15
    _erase_length: int = 0
    def __init__(self, total_count: int, max_width: int = 30):
        self.total_count = total_count
        self.width = min(total_count, max_width)
        self._show_progress()
    def _print_progress(self):
        n_hashes = int(self.current_count / self.total_count * self.width)
        n_blanks = self.width - n_hashes
        progress_bar = f'[{"#"*n_hashes}{" "*n_blanks}]\t{self.current_count} / {self.total_count}'
        self._erase_length = len(progress_bar)
        print(progress_bar, end='\r')
    def _erase(self):
        print(' '*self._erase_length, end='\r')
    def _show_progress(self):
        if self.current_count < self.total_count:
            self._print_progress()
        else:
            self._erase()
    def step(self):
        self.current_count += 1
        self._show_progress()
    def reset(self):
        self._erase()

'''
control training process using signal file
functionalities:
1. stop at next epoch
2. pause training at next dataload
3. immediate stop if signal file deleted
    but the training will be discarded
4. to solve read conflict, continue training if the 
'''

SIGNAL_DIR = os.path.join(__file__, '..', '..', '_signal')

class SignalFileControl:
    '''
    create a tempfile, stop training process when tempfile is deleted
    '''
    temp_file_path: str
    def __init__(self):
        # create tempfile under file_path
        temp_file_path = tempfile.mktemp(dir='_signal', prefix='signal-')
        self.temp_file_path = temp_file_path
        with open(self.temp_file_path, 'w') as file:
            file.write('')
        print(f'signal file path: {temp_file_path}')
    def is_alive(self):
        return os.path.isfile(self.temp_file_path)
    def reset(self):
        if os.path.isfile(self.temp_file_path):
            os.remove(self.temp_file_path)

class SignalYamlControl:
    '''
    control progress by a signal yaml file
    '''
    temp_file_path: str
    def __init__(self):
        # create tempfile under file_path
        temp_file_path = tempfile.mktemp(dir=SIGNAL_DIR, prefix='signal-')
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
