

class Tasker:

    KEYWORD_DICT = dict(env_maker=None,
                        state_to_input=None,
                        load_model=None,
                        verbose=False,)

    def __init__(self, **kwargs):
        # combine arguments from self.KEYWORD_DICT and kwargs and set arguments
        self.__dict__.update(self.KEYWORD_DICT)
        self.__dict__.update(kwargs)

        # print arguments
        self.print('#### All arguments ####')
        for keyword, value in sorted(self.__dict__.items()):
            statement = '    {} = {}'.format(keyword, value)
            if keyword not in self.KEYWORD_DICT:
                statement += ' (UNUSED)'
            self.print(statement)

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs, flush=True)
