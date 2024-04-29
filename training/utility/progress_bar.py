'''
progress bar for training process
'''

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