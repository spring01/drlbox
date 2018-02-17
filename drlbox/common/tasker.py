

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
            statement = '    {} = {}'.format(keyword, value)
            if keyword not in self.KEYWORD_DICT:
                statement += ' (UNUSED)'
            self.print(statement)

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs, flush=True)
