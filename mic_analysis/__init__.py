from pprint import pprint

from confapp import conf

conf += "mic_analysis.constants"

try:
    import local_settings

    conf._modules[0].SETTINGS_PRIORITY = 1
    conf += local_settings
    local_settings.SETTINGS_PRIORITY = 0
    print("Loaded local_settings to conf")
    conf._modules = conf._modules[::-1]
except ImportError:
    print("Could not load local_settings to conf")
