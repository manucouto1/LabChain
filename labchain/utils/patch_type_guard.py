def patch_inspect_for_notebooks():
    """
    Patch inspect.getsource to use dill for obtaining source code in notebooks.
    """
    try:
        import dill
        import inspect

        inspect.getsource = dill.source.getsource
        print("✅ Patched inspect.getsource using dill.")
    except ImportError:
        print("⚠️ dill is not installed, skipping inspect patch.")
