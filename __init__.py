print("__init__")
is_simple_core = False  #True

if is_simple_core:
    from deone.core_simple import Variable
    from deone.core_simple import Function
    from deone.core_simple import using_config
    from deone.core_simple import no_grad
    from deone.core_simple import as_array
    from deone.core_simple import as_variable
    from deone.core_simple import setup_variable
else:
    from deone.core import Variable
    from deone.core import Function
    from deone.core import using_config
    from deone.core import no_grad
    from deone.core import as_array
    from deone.core import as_variable
    from deone.core import setup_variable
    from deone.core import Parameter

setup_variable()
