import pkg_resources

__version__ = pkg_resources.get_distribution('clm_demeter').version

__all__ = ['reclassify_base', 'reclassify_projected']