'''Presets, as well as preset utility methods.'''
from .core import collect_leaves


PRESETS = {}


def register_preset(name):
    '''Register a preset in the global PRESETS dict under `name`.'''
    def wrapped(preset):
        '''Wrapped function to be called on the preset to register it to the global PRESETS dict.'''
        PRESETS[name] = preset
        return preset
    return wrapped


class RemovableHandleList(list):
    '''A list to hold handles, with the ability to call remove on all of its members.'''
    def remove(self):
        '''Call remove on all members, effectively removing handles from modules, or reverting canonizers.'''
        for handle in self:
            handle.remove()
        self.clear()


class Preset:
    '''A Preset to apply canonizers and register hooks to modules.
    One Preset instance may only be applied to a single module at a time.

    Parameters
    ----------
    module_map: list[tuple[tuple[type, ...], Hook]]
        A mapping from possible module types to Hooks that shall be applied to instances of said hooks.
    canonizers: list[Canonizer]
        List of canonizers to be applied before applying hooks.
    '''
    def __init__(self, module_map=None, canonizers=None):
        self.module_map = module_map
        self.canonizers = canonizers

        self.handles = RemovableHandleList()

    def register(self, module):
        '''Apply all canonizers and register all hooks to a module (and its recursive children).
        Previous canonizers of this preset are reverted and all hooks registered by this preset are removed.
        The module or any of its children (recursively) may still have other hooks attached.

        Parameters
        ----------
        module: obj:`torch.nn.Module`
            Hooks and canonizers will be applied to this module recursively according to `module_map` and `canonizers`
        '''
        self.handles.remove()

        for canonizer in self.canonizers:
            self.handles.append(canonizer(module))

        for leaf in collect_leaves(module):
            for types, hook_template in self.module_map:
                if isinstance(leaf, types):
                    hook = hook_template.copy()
                    self.handles.append(leaf.register_forward_hook(hook.forward))
                    self.handles.append(leaf.register_backward_hook(hook.backward))

    def remove(self):
        '''Remove all handles for hooks and canonizers.
        Hooks will simply be removed from their corresponding Modules.
        Canonizers will revert the state of the modules they changed.
        '''
        self.handles.remove()
