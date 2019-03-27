import pkg_resources

__version__ = pkg_resources.get_distribution('demeter_clm').version

__all__ = ['reclassify_base', 'reclassify_projected']