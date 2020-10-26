'''Presets, registered in a global preset dict.'''
PRESETS = {}


def register_preset(name):
    '''Register a preset in the global PRESETS dict under `name`.'''
    def wrapped(preset):
        '''Wrapped function to be called on the preset to register it to the global PRESETS dict.'''
        PRESETS[name] = preset
        return preset
    return wrapped
