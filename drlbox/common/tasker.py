

class Tasker:

    KEYWORD_DICT = dict(env_maker=None,
                        state_to_input=None,
                        load_model=None,
                        verbose=False,)

    def __init__(self, **kwargs):
        # combine arguments from self.KEYWORD_DICT and kwargs
        all_kwargs = {}
        for keyword, value in self.KEYWORD_DICT.items():
            all_kwargs[keyword] = value
        # replace with user-specified arguments
        for keyword, value in kwargs.items():
            all_kwargs[keyword] = value

        # set arguments
        for keyword, value in all_kwargs.items():
            setattr(self, keyword, value)

        # print arguments
        self.print('#### All arguments ####')
        for keyword, value in sorted(all_kwargs.items()):
            if keyword not in self.KEYWORD_DICT:
                self.print('    {} = {} (UNUSED)'.format(keyword, value))
            else:
                self.print('    {} = {}'.format(keyword, value))

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs, flush=True)
