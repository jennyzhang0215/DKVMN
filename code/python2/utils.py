from progress.bar import Bar

class ProgressBar(Bar):
    message = 'Loading'
    fill = '='
    suffix = '%(percent).1f%% | Elapsed: %(elapsed)ds | ETA: %(eta)ds '
